"""Adapter（Pfeiffer 风格瓶颈适配器）—— 给 CLIP 视觉塔的 MLP 后挂串联残差。

背景与动机
----------
官方规则五.2 原文点名「**Adapter** 等参数高效微调方法」，本队此前只做了 LoRA 与
VPT，**Adapter 完全没做**。它和 LoRA 是**正交**的两条范式：

* LoRA 是**并联**低秩旁路（改 attention / MLP 的 Linear 权重）；
* Adapter 是**串联**瓶颈层（在 MLP 输出后加一个 ``x → x + W_up(GELU(W_down x))``）。

二者可单独开、可叠加，便于做「Adapter-only / LoRA-only / 两者」三臂对照。
Adapter 不依赖任何噪声识别（区别于已被否掉的 TrustCLIP/TGP），是纯粹的"换做法"级改动。

★ 零初始化等价性（与 LoRA 同、与 VPT 不同）
    ``W_up`` 零初始化 ⇒ 训练起点 ``Adapter(x) = x``，模型逐位等于原始 CLIP。
    这是本项目最看重的"不破坏 CLIP 先验"保障 —— 见 ``report`` 的 init-identity 检查。
    （VPT 做不到这点，所以 VPT 的自检是"δ 非零且有限"；Adapter 能做到"δ≈0"。）

挂接位置
--------
``vision_model.encoder.layers.N.mlp``（``CLIPMLP``，含 fc1 768→3072 / fc2 3072→768）
**之后**。12 层每层 1 个 Adapter（Pfeiffer 版，先单点；Houlsby 双点留待后续）。

    CLIPEncoderLayer.forward 内部是 ``hidden_states = self.mlp(hidden_states)``，
    单参数 ``hidden_states``，极稳定。我们把 ``layer.mlp`` 换成 ``MLPWithAdapter``
    包装即可，不必改 layer 的 forward —— 包装的 ``forward`` 也只吃一个
    ``hidden_states``，签名天然对齐（用 inspect 探过，不凭猜）。

合规
----
Adapter 旁路是本队自训参数，基础权重仍是 OpenAI 官方那一份，训练全程只有这一套权重
参与。与 LoRA 同样合规（赛规十一（二）3「仅限 OpenAI 官方权重」）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ★ 参数名里带这个串才被认作 Adapter 参数。刻意避开 "lora_"：
#   ``lora.py::mark_only_lora_trainable`` 会把**不含 lora_ 的参数一律冻结**——
#   名字若撞上 "lora_"，两组参数会互相干扰。也避开 "vpt_prompt"（VPT 的 tag），
#   免得参数分组/账目检查里出现重名误判。
ADPT_TAG = "adapter"


# ------------------------------------------------------------------ 单层瓶颈
class Adapter(nn.Module):
    """瓶颈适配器：``forward(x) = x + W_up(GELU(W_down x))``。

    * ``W_down`` : dim → bottleneck（下采样）
    * ``W_up``   : bottleneck → dim（上采样，**零初始化** ⇒ 初始恒等）

    ``down`` 用一个较小的正态初始化（std=1e-2）做"小扰动起步"；``up`` 零初始化
    保证训练起点逐位等于原 CLIP，不会破坏预训练先验。
    """

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError(f"bottleneck 必须为正，收到 {bottleneck}")
        if dim <= 0:
            raise ValueError(f"dim 必须为正，收到 {dim}")
        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        # 下采样：小正态，保证训练初期梯度尺度合理
        nn.init.normal_(self.down.weight, std=1e-2)
        nn.init.zeros_(self.down.bias)
        # ★ 上采样零初始化 ⇒ Adapter(x) ≡ x（恒等），不破坏原模型
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, bottleneck={self.bottleneck}"


class MLPWithAdapter(nn.Module):
    """包装 ``CLIPMLP``：输出 = ``adapter(mlp(x))``。只改 mlp 模块，不动 layer。

    串行放置（Adapter 作用在 MLP 输出上）与 Pfeiffer 原版一致；初始时
    ``adapter(mlp(x)) == mlp(x)``（因 up=0），所以整个模型起点逐位等于原 CLIP。
    """

    def __init__(self, orig_mlp: nn.Module, dim: int, bottleneck: int):
        super().__init__()
        self.mlp = orig_mlp          # 原始 CLIPMLP，保持原接口可见
        self.adapter = Adapter(dim, bottleneck)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.mlp(hidden_states))

    def extra_repr(self) -> str:
        return (f"mlp={type(self.mlp).__name__}, "
                f"adapter({self.adapter.extra_repr()})")


# ------------------------------------------------------- 视觉塔结构定位
def _vision_tower(model: nn.Module) -> nn.Module:
    vt = getattr(model, "vision_model", None) or getattr(model, "visual", None)
    if vt is None:
        raise RuntimeError("[adapter] 找不到视觉塔（vision_model / visual）")
    return vt


def _encoder_layers(model: nn.Module) -> list:
    """返回 encoder 的 layers（ModuleList）。兼容 CLIPModel 外层代理结构。

    CLIPModel.vision_model 是 CLIPVisionModel，它内部再包一层
    CLIPVisionTransformer（含真正的 .encoder.layers）。所以先看 ``vt.encoder``，
    没有再往 ``vt.vision_model.encoder`` 找。
    """
    vt = _vision_tower(model)
    enc = getattr(vt, "encoder", None)
    if enc is None:
        inner = getattr(vt, "vision_model", None)
        enc = getattr(inner, "encoder", None) if inner is not None else None
    if enc is None:
        raise RuntimeError(
            "[adapter] 视觉塔里找不到 encoder（已试 vt.encoder / "
            "vt.vision_model.encoder）。本版本结构可能不同，请核对。")
    layers = getattr(enc, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("[adapter] encoder 里没有 layers（ModuleList 为空/缺失）")
    return list(layers)


def _mlp_of(layer: nn.Module) -> nn.Module | None:
    """取某 encoder 层的 mlp；若已包装则返回其内部原 mlp（便于幂等解包）。"""
    m = getattr(layer, "mlp", None)
    if m is None:
        return None
    if isinstance(m, MLPWithAdapter):
        return m.mlp
    return m


def _detect_dim(layers: list) -> int:
    """从某一层的 CLIPMLP 推出隐藏维度（不写死 768）。"""
    for layer in layers:
        mlp = _mlp_of(layer)
        if mlp is None:
            continue
        fc1 = getattr(mlp, "fc1", None)
        if isinstance(fc1, nn.Linear) and fc1.in_features > 0:
            return int(fc1.in_features)
        fc2 = getattr(mlp, "fc2", None)
        if isinstance(fc2, nn.Linear) and fc2.out_features > 0:
            return int(fc2.out_features)
    raise RuntimeError("[adapter] 无法从任意层的 mlp 推断隐藏维度（fc1/fc2 缺失）")


# ------------------------------------------------------------------ 注入
def inject_adapter(model: nn.Module, bottleneck: int = 64,
                   verbose: bool = True) -> dict:
    """在 ``model`` 的视觉塔每层 mlp 后注入 Pfeiffer Adapter。

    幂等：已注入（``MLPWithAdapter``）则先解包回原 ``CLIPMLP`` 再重注，避免套娃。
    返回统计信息 dict，调用方应当落盘/打印。

    ★ 调用时序约束（与 LoRA/VPT 一致）：必须在 ``inject_lora`` / 全局冻结**之后**
    调用，否则新创建的 Adapter 参数会被 inject_lora 的"整模型 requires_grad=False"
    静默冻住（参数在、梯度为零、训练毫无变化）。本函数内部会再把 Adapter 参数
    显式置 ``requires_grad=True`` 做双保险。
    """
    if bottleneck <= 0:
        return {"injected": False, "n_layers": 0, "params": 0,
                "bottleneck": 0, "dim": 0}

    layers = _encoder_layers(model)
    dim = _detect_dim(layers)

    # ★ 设备 / dtype 对齐（与 lora.py / vpt.py 同一个坑）：
    #   新模块默认在 CPU，模型可能已在 GPU 上 → 前向会抛
    #   RuntimeError: Expected all tensors to be on the same device。
    base_device = next(model.parameters()).device
    base_dtype = next(model.parameters()).dtype

    n_layers = len(layers)
    wrapped_paths: list = []
    for i, layer in enumerate(layers):
        cur = _mlp_of(layer)
        if cur is None:
            raise RuntimeError(
                f"[adapter] 第 {i} 层找不到 mlp（既非 CLIPMLP 也非已包装）——"
                f"拒绝猜测，结构可能不符预期。")
        wrap = MLPWithAdapter(cur, dim, bottleneck)
        wrap = wrap.to(device=base_device, dtype=base_dtype)
        # 双保险：确保 Adapter 旁路参数可训练（orig mlp 的 requires_grad 维持原状）
        for p in wrap.adapter.parameters():
            p.requires_grad = True
        layer.mlp = wrap
        wrapped_paths.append(f"encoder.layers.{i}.mlp")

    # 统计 Adapter 参数量（按名收集，避免把 orig mlp 也算进来）
    adpt_params = sum(p.numel() for p in adapter_parameters(model, False))

    info = {
        "injected": True,
        "n_layers": n_layers,
        "params": adpt_params,
        "bottleneck": int(bottleneck),
        "dim": int(dim),
        "device": str(base_device),
        "dtype": str(base_dtype),
        "expect_params": n_layers * (2 * dim * bottleneck + dim + bottleneck),
        "attr_examples": wrapped_paths[:2],
    }
    if verbose:
        print(f"[adapter] ★ 注入 Pfeiffer Adapter：{n_layers} 层，"
              f"dim={dim}，bottleneck={bottleneck}，新增参数 {adpt_params:,}"
              f"（{base_device}/{base_dtype}，零初始化⇒训练起点=原 CLIP）")
        print(f"[adapter]   期望参数量 {info['expect_params']:,}，"
              f"实际 {adpt_params:,} "
              f"{'✔' if adpt_params == info['expect_params'] else '✘ 不符！'}")
    return info


def find_adapter_modules(model: nn.Module) -> list:
    """返回所有已注入的 ``MLPWithAdapter`` 模块。"""
    return [m for m in model.modules() if isinstance(m, MLPWithAdapter)]


def adapter_parameters(model: nn.Module, only_trainable: bool = True) -> list:
    """按**参数名**精确收集 Adapter 参数。

    ⚠️ 刻意不用 ``[p for p in model.parameters() if p.requires_grad]``：
    那会把分类头、logit_scale、解冻的骨干一起收进来，导致它们在优化器分组里
    被 adapter 组抢走、再用各自的组去重剔掉 ⇒ **静默用了错的学习率**。
    （lora.py / vpt.py 里同一个坑都踩过。）
    """
    out = []
    for name, p in model.named_parameters():
        if ADPT_TAG not in name:
            continue
        if only_trainable and not p.requires_grad:
            continue
        out.append(p)
    return out


def adapter_state_dict(model: nn.Module) -> dict:
    """只抽 Adapter 权重（几十 KB），便于快速比对/单独保存。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ADPT_TAG in k}


def report(model: nn.Module, img_size: int, batch: int = 2,
           device: str | None = None, verbose: bool = True) -> dict:
    """注入后的自检（在真机上跑，判据见各字段说明）。

    返回 dict，其中 ``ok`` 为总判定。任何一项不通过都应当**拒绝开跑** ——
    Adapter 的失败态几乎都是静默的：程序照跑、loss 照降，但学的是别的东西。

    ⚠️ 本函数应在**训练开始前**调用（零初始化等价性只在 up=0 时成立）。
    检查项：
      1. 注入层数 == 12（每层 mlp 都被 MLPWithAdapter 替换）
      2. 零初始化等价 —— 包装输出 vs 原 mlp 输出 max|Δ|≈0（证明不破坏先验）
      3. 梯度流向      —— adapter 参数有非零梯度；未解冻的骨干 grad 为 None
      4. 冻结干净      —— 带"非零梯度"的冻结参数 == 0（无泄漏）
      5. 参数账目      —— 可训练 Adapter 参数 == 12×(2·dim·bottleneck+dim+bottleneck)，
                          且与 lora_ / vpt_prompt 名集合不重叠
    """
    mods = find_adapter_modules(model)
    if not mods:
        raise RuntimeError("[adapter] report() 必须在 inject_adapter() 之后调用")

    dev = torch.device(device or next(model.parameters()).device)
    n_layers = len(mods)
    dim = mods[0].adapter.dim
    bottleneck = mods[0].adapter.bottleneck
    expect_params = n_layers * (2 * dim * bottleneck + dim + bottleneck)

    res = {
        "n_layers": n_layers, "dim": dim, "bottleneck": bottleneck,
        "expect_params": expect_params,
        "attr_examples": [f"encoder.layers.{i}.mlp.adapter" for i in range(2)],
    }

    # --- 1. 层数 ---
    layers = _encoder_layers(model)
    all_wrapped = all(isinstance(getattr(l, "mlp", None), MLPWithAdapter)
                      for l in layers)
    res["all_layers_wrapped"] = all_wrapped
    res["check_layers"] = bool(all_wrapped and n_layers == len(layers))

    # --- 2. 零初始化等价（前向，无梯度）---
    # 取一个随机 hidden_states，比较 包装输出 与 原 mlp 输出。
    # 因 up=0，二者应逐位相等（仅数值舍入）。
    with torch.no_grad():
        hs = torch.randn(batch, 50, dim, device=dev)
        deltas = []
        for m in mods:
            w_out = m(hs)
            o_out = m.mlp(hs)
            deltas.append(float((w_out - o_out).abs().max().item()))
        res["init_identity_max_abs_diff"] = max(deltas) if deltas else float("nan")
        res["check_init_identity"] = bool(
            torch.isfinite(torch.tensor(deltas)).all().item()
            and res["init_identity_max_abs_diff"] < 1e-4)

    # --- 3 + 4. 梯度流向 / 冻结干净（端到端前向+反向）---
    model.train()
    model.to(dev)
    x = torch.zeros(batch, 3, img_size, img_size, device=dev)
    interp = img_size != 224
    try:
        f = _embed(model, x, interp)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"[adapter] report 端到端前向失败：{type(e).__name__}: {e}") from e
    loss = f.float().pow(2).sum()
    loss.backward()

    # adapter 梯度（取第一个模块做代表，并汇总所有 adapter 参数）
    adpt_ps = adapter_parameters(model, False)
    grads = [p.grad for p in adpt_ps if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all().item()) for g in grads) if grads else False
    nonzero = any(bool((g.abs().max().item() > 0.0)) for g in grads) if grads else False
    res["adapter_grad_n_params"] = len(grads)
    res["adapter_grad_absmax"] = float(max(
        (g.abs().max().item() for g in grads), default=0.0))
    res["check_grad_flow"] = bool(grads and finite and nonzero)

    # 冻结参数不得带非零梯度
    leaked = [n for n, p in model.named_parameters()
              if p.requires_grad is False and p.grad is not None
              and not torch.all(p.grad == 0)]
    res["frozen_params_with_grad"] = len(leaked)
    res["frozen_examples"] = leaked[:3]
    res["check_frozen_clean"] = bool(len(leaked) == 0)

    # --- 5. 参数账目 ---
    n_adpt = sum(p.numel() for p in adpt_ps)
    res["adapter_params_found"] = n_adpt
    adpt_names = {n for n, _ in model.named_parameters() if ADPT_TAG in n}
    lora_names = {n for n, _ in model.named_parameters() if "lora_" in n}
    vpt_names = {n for n, _ in model.named_parameters() if "vpt_prompt" in n}
    res["name_overlap_with_lora"] = sorted(adpt_names & lora_names)
    res["name_overlap_with_vpt"] = sorted(adpt_names & vpt_names)
    res["check_param_accounting"] = bool(
        n_adpt == expect_params
        and not res["name_overlap_with_lora"]
        and not res["name_overlap_with_vpt"])

    res["ok"] = all([
        res["check_layers"], res["check_init_identity"],
        res["check_grad_flow"], res["check_frozen_clean"],
        res["check_param_accounting"],
    ])

    if verbose:
        def _m(b):
            return "PASS" if b else "**FAIL**"
        print("[adapter-report] " + "-" * 56)
        print(f"[adapter-report] n_layers={n_layers} dim={dim} "
              f"bottleneck={bottleneck}")
        print(f"[adapter-report]  1 注入层数    {n_layers}（期望 {len(layers)}）  "
              f"{_m(res['check_layers'])}")
        print(f"[adapter-report]  2 零初始化等价 max|Δ|="
              f"{res['init_identity_max_abs_diff']:.3e}  "
              f"{_m(res['check_init_identity'])}  ← up=0 ⇒ 起点=原 CLIP")
        print(f"[adapter-report]  3 梯度流向    adapter grad max="
              f"{res['adapter_grad_absmax']:.3e}（{res['adapter_grad_n_params']} 个）  "
              f"{_m(res['check_grad_flow'])}")
        print(f"[adapter-report]  4 冻结干净    带梯度的冻结参数 "
              f"{len(leaked)} 个  {_m(res['check_frozen_clean'])}")
        print(f"[adapter-report]  5 参数账目    {n_adpt:,} / {expect_params:,}，"
              f"与 lora {len(res['name_overlap_with_lora'])} / vpt "
              f"{len(res['name_overlap_with_vpt'])} 重名  "
              f"{_m(res['check_param_accounting'])}")
        print(f"[adapter-report] 总判定: {'OK' if res['ok'] else 'FAILED'}")
        print("[adapter-report] " + "-" * 56)

    # 收尾：清掉自检留下的梯度，避免污染真实训练的第一次 backward
    model.zero_grad(set_to_none=True)
    model.eval()
    return res


def _embed(model: nn.Module, x: torch.Tensor, interp: bool) -> torch.Tensor:
    """取"图像嵌入"（投射后 512 维），兼容 transformers 新旧返回类型。"""
    from extract import take_embeds
    return take_embeds(model.get_image_features(
        pixel_values=x, interpolate_pos_encoding=interp))


def param_level_count(model: nn.Module) -> int:
    """可训练参数数（用于与 lora.py / vpt.py 的统计口径对齐）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
