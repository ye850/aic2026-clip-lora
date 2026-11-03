# -*- coding: utf-8 -*-
"""区域自注意力适配器（Region Self-Attention Adapter）

--------------------------------------------------------------------
这是什么
--------------------------------------------------------------------
在 CLIP 视觉塔的 **patch token 序列**上做一次区域级自注意力聚合，
插在 `encoder` 之后、`post_layernorm` 之前。

★ **它与我们所有已投的轴都正交**：
| 我们的做法 | 改的是什么 |
|---|---|
| LoRA 注入 q/k/v/out/fc1/fc2 | **单点权重**（"怎么算"）|
| MLP 头 hidden=1024 | **最终分类特征**（"怎么分类"）|
| TTA 15 视图 | **推理时看几次** |
| ★ **本适配器** | ★ **信息聚合结构**（"看哪些区域"）|

--------------------------------------------------------------------
结构依据
--------------------------------------------------------------------
区域级自注意力适配器是公开文献中常见的视觉适配器结构（在 patch token 序列上
做一次轻量自注意力聚合，再回写），本工程按其常规做法实现。
本项目的实测读数见 `experiments/` 下的判死审计记录。

--------------------------------------------------------------------
与原实现的差异（重要）
--------------------------------------------------------------------
常见实现基于**原版 CLIP 代码**（含 `resblocks` / `ln_post`），
★ 而**我们用的是 HF `transformers` 的 CLIPVisionModel**（v5.17）。
⇒ 两者没有共享的"插入点"，故**用 forward hook 实现**，
   **不修改 HF 内部代码**（可随版本升级、随时可摘除）。
⇒ 等价性：HF 的 `last_hidden_state`（= encoder 输出）**正是**
   原版 `resblocks` 之后、`ln_post` 之前的那个 token 序列
   ⇒ ★ 插入位置与原实现**语义一致**。

--------------------------------------------------------------------
合规
--------------------------------------------------------------------
★ 只增加本工程的模块参数；**CLIP 主干权重仍只用官方 OpenAI 那份**
  ⇒ 按本项目合规裁定（"从哪来是官方 CLIP，改成什么样不违规"）**合规**
  （对照：铁律 99）。
⇒ 仍是**单模型**，不构成集成。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    """CLIP 原版用的 QuickGELU（x * sigmoid(1.702x)），非 tanh 近似。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class RegionSelfAttentionAdapter(nn.Module):
    """区域自注意力适配器。

    参数
    ----
    width:视觉塔宽度（ViT-B/32 = 768）
    heads:注意力头数（12）
    pool  :区域池化核（4 ⇒ 把 patch 网格压成 1/4 边长）
    gate_init:门控初值（−2.2 ⇒ sigmoid ≈ 0.10，**接近恒等但梯度非零**）

    ★★ `gate_init=−2.2` 是原实现的关键设计，别改成 0：
       恰好 0 会让注意力权重拿不到梯度（残差分支完全断开）。
    """

    def __init__(self, width: int, heads: int = 12, pool: int = 4,
                 dropout: float = 0.0, gate_init: float = -2.2):
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} 须能被 heads {heads} 整除")
        self.width = int(width)
        self.heads = int(heads)
        self.pool = max(1, int(pool))
        hidden = max(32, width // 16)
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=float(dropout),
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), QuickGELU(), nn.Linear(width * 2, width))
        self.modulator = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden), QuickGELU(),
            nn.Linear(hidden, width))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # ★ 可视化/调试用：记录最近一次前向的网格信息
        self.last_grid: int | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """`tokens`: (B, N, D)，N = 1 个 class token + patch 数。"""
        if tokens.ndim != 3 or tokens.shape[1] <= 1:
            return tokens
        B, N, D = tokens.shape
        if D != self.width:
            raise RuntimeError(f"宽度不符：{D} != {self.width}")
        patch_count = N - 1
        grid = int(round(patch_count ** 0.5))
        if grid * grid != patch_count:
            # ★ 非方阵（极端输入尺寸）⇒ 原实现也是直接跳过
            return tokens
        self.last_grid = grid

        # (B, N, D) -> (B, D, H, W)
        patches = tokens[:, 1:].transpose(1, 2).reshape(B, D, grid, grid)
        pool = self.pool
        if grid % pool != 0:
            divisors = [d for d in (4, 3, 2, 1) if grid % d == 0]
            pool = next((d for d in divisors if d <= self.pool), 1)
        region_map = F.avg_pool2d(patches, kernel_size=pool, stride=pool)
        rh, rw = region_map.shape[-2:]
        regions = region_map.flatten(2).transpose(1, 2)     # (B, R, D)

        normed = self.norm1(regions)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        channel_gate = torch.sigmoid(self.modulator(regions.mean(dim=1, keepdim=True)))
        refined = regions + attended * channel_gate
        refined = refined + self.ffn(self.norm2(refined))

        up = F.interpolate(
            refined.transpose(1, 2).reshape(B, D, rh, rw),
            size=(grid, grid), mode="bilinear", align_corners=False)
        patch_delta = up.flatten(2).transpose(1, 2)
        gate = torch.sigmoid(self.gate).to(dtype=tokens.dtype)
        class_delta = refined.mean(dim=1, keepdim=True).to(dtype=tokens.dtype)
        out_cls = tokens[:, :1] + gate * class_delta
        out_patch = tokens[:, 1:] + gate * patch_delta.to(dtype=tokens.dtype)
        return torch.cat([out_cls, out_patch], dim=1)


# --------------------------------------------------------------------- 挂载
class _RegionVisionModel(torch.nn.Module):
    """包装 `CLIPVisionModel`，在 **pooled 计算之前** 插入区域适配器。

    ★★ **为什么不能用 forward hook**（2026-10-07 实测踩到的坑）：
       transformers v5.17 的 `CLIPVisionModel.forward` 是
       ```
       last_hidden_state = encoder_outputs.last_hidden_state
       pooled_output    = last_hidden_state[:, 0, :]     # ★ 在这里取class token
       pooled_output    = post_layernorm(pooled_output)
       ```
       ① 挂在 `vision_model` 上的 hook **触发得太晚**（forward 已跑完），
          实测挂/不挂的特征余弦 = **1.00000000**、最大差 = **0.0** ⇒ **完全无效**；
       ② 挂在 `encoder` 上同样太晚；
       ③ 若强行改 `out.pooler_output`，`get_image_features` 会
          「读 pooler_output → 原地写回同一个对象」，把**投影后的 512 维**
          再送进 `visual_projection`(768→512) ⇒ 维度错。
    ⇒ ★ 唯一稳妥的做法是**接管 forward**，在正确位置插入。
    """

    def __init__(self, vision_model, adapter: "RegionSelfAttentionAdapter"):
        super().__init__()
        # ★ 用 `object.__setattr__` 直接塞进 __dict__，并让 __getattr__ 不再参与
        #   （见下方说明：走 __getattr__ 会无限递归）
        object.__setattr__(self, "_vm", vision_model)
        self.region_adapter = adapter

    def forward(self, *args, **kwargs):
        vm = self._vm
        pixel_values = kwargs.get("pixel_values", None)
        interpolate = kwargs.get("interpolate_pos_encoding", False)
        # 与 v5.17 原 forward 同序；新增的**只有加粗那一行**
        hidden_states = vm.embeddings(pixel_values,
                                     interpolate_pos_encoding=interpolate)
        hidden_states = vm.pre_layrnorm(hidden_states)
        encoder_outputs = vm.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs.last_hidden_state
        # ★★★ 区域适配器插在这里：resblocks 之后、post_layernorm 之前
        last_hidden_state = self.region_adapter(last_hidden_state)
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = vm.post_layernorm(pooled_output)
        try:                        # 与新版返回类型对齐
            from transformers.modeling_outputs import BaseModelOutputWithPooling
            return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state,
                                              pooler_output=pooled_output)
        except Exception:           # pragma: no cover
            return last_hidden_state, pooled_output

    # ⚠️ **刻意不实现 `__getattr__`**：
    #   nn.Module 的属性访问在 `__getattr__` 里转 `self.vm`，
    #   而 `vm` 是子模块 ⇒ 会再次进入 `__getattr__` ⇒ 无限递归（实测 990+ 层）。
    #   ⇒ 需要的透传属性（config 等）请**显式**加 property。
    @property
    def config(self):
        return self._vm.config

    # ★★★ 2026-10-07 补：显式透传 `encoder`（**这一个是硬需求**）。
    #   `finetune_lora.py` 的骨干解冻逻辑是：
    #       vt    = getattr(clip_model, "vision_model", None) or ...
    #       layers = getattr(getattr(vt, "encoder", None), "layers", None)
    #       if layers is None: raise SystemExit("[unfreeze] 🛑 找不到 encoder.layers")
    #   挂上适配器后 `vision_model` 就是本包装类 ⇒ **没有 `encoder` 属性**
    #   ⇒ 实测报 `[unfreeze] 🛑 视觉塔里找不到 encoder.layers` 并 SystemExit，
    #      **同时丢掉 U 冠军的「解冻最后 4 层」这个关键变量** ⇒ 变成不干净对照。
    #   ⇒ 这里透传它，让解冻逻辑照常工作。
    @property
    def encoder(self):
        return self._vm.encoder

    @property
    def embeddings(self):
        return self._vm.embeddings

    @property
    def pre_layrnorm(self):
        return self._vm.pre_layrnorm

    @property
    def post_layernorm(self):
        return self._vm.post_layernorm


def attach_region_adapter(clip_model, width: int | None = None, heads: int = 12,
                         pool: int = 4, dropout: float = 0.0,
                         gate_init: float = -2.2):
    """把适配器挂到 `CLIPModel` 的视觉塔上（**接管 forward**）。

    ★ 传入的是**顶层 CLIPModel**（不是 vision_model）——因为要替换
      `clip.vision_model` 这个属性，让`get_image_features` 走新实现。
    返回 (adapter)；卸载用 `detach_region_adapter(clip_model)`。
    """
    vision = clip_model.vision_model
    w = int(width or getattr(getattr(vision, "config", None), "hidden_size", 768)
            or getattr(vision, "hidden_size", 768))
    dev = next(clip_model.parameters()).device
    adapter = RegionSelfAttentionAdapter(w, heads=heads, pool=pool,
                                        dropout=dropout,
                                        gate_init=gate_init).to(dev)
    clip_model._region_orig_vision = vision      # 留原引用，便于还原
    clip_model.vision_model = _RegionVisionModel(vision, adapter)
    return adapter


def detach_region_adapter(clip_model):
    """还原为原始视觉塔（用于「挂/不挂」对照与推理时裁剪）。"""
    orig = getattr(clip_model, "_region_orig_vision", None)
    if orig is None:
        return False
    clip_model.vision_model = orig
    if hasattr(clip_model, "_region_orig_vision"):
        del clip_model._region_orig_vision
    return True
