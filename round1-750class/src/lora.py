# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""LoRA（Low-Rank Adaptation）低秩旁路，用于解冻 CLIP 视觉骨干。

为什么需要它
------------
本模块的动机来自一条**容量假设**。前一阶段的真实榜分是：

    全套抗噪方法（SCE + CL 降权 + 类平衡） = 52.04
    朴素交叉熵                            = 49.80
    Δ = +2.24

把一整套噪声鲁棒方法加满，相对最朴素的交叉熵只换来 2.24 分。一个自然的解释是
**瓶颈不在损失函数，而在模型容量** —— 骨干完全冻结时只有一个 512×K 的线性头
可调，抗噪方法能腾挪的空间被卡死了。

⚠️ **但这条假设在本阶段的受控对照中并未被证实。** 把两条路线的训练轮数对齐后
（同一子集、同一预算），LoRA 与冻结线性头的差距为 −0.0200，95% 置信区间跨 0，
**统计上不显著**（见 ``scripts/run_ab_harness.py`` 与
``docs/性能与自检记录.md`` 的实测一节）。

⇒ 因此本模块保留为**一条并行的可选路径，不承担提交主链路**；主链路仍是
「冻结骨干 + 鲁棒线性头」。保留它的理由是：受控对照只覆盖 150 类（全量为 750 类），
任务偏简单、冻结线已接近饱和，且为公平起见两边只训了 8 轮 ——
**结论不能外推到全量规模**。

作为对照，全参数微调 ViT-B/32 在本机（RTX 4050 Laptop / 6GB 显存）不现实：

    全参数微调   87,456,000 参数 → 训练态约 1335 MB（fp16权重2B + Adam m,v 8B + fp32 master 4B）
    LoRA r=16     1,179,648 参数 → 训练态约   18 MB
    省 74 倍

LoRA 的做法是不动原权重，只在注意力的若干投影层旁边挂一条低秩旁路：

    h = W0 x + (alpha / r) * B A x

其中 A ∈ R^{r×d_in}、B ∈ R^{d_out×r}，r << d。初始化 A ~ N(0, sigma)、B = 0，
因此**训练开始时旁路输出恰为 0，模型与原始 CLIP 完全等价** —— 这是 LoRA 的
关键性质：不会因为「加了模块」而破坏预训练权重带来的起点。

规则合规性
----------
赛题要求「必须使用 CLIP ViT-B/32 且**仅使用 OpenAI 官方公开权重**，不得使用
其他预训练模型」。LoRA 旁路是**本队自训参数**，不对应任何外部预训练权重；
基础权重仍是 OpenAI 官方那一份，训练全程只有这一套权重参与。合规。

若对「不得使用其他预训练模型」从严解释，可把 ``--lora-scale 0`` 退化为冻结骨干
基线；两者共用同一份代码，便于对照。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import torch
import torch.nn as nn


# ------------------------------------------------------------------ 单层旁路
class LoRALinear(nn.Module):
    """给一个已存在的 ``nn.Linear`` 挂低秩旁路，原权重冻结。

    ``forward`` 输出：

        y = W0 x + scale * B(A(dropout(x)))

    其中 ``scale = alpha / r``。``B`` 初始化为零 → 初始时 y == W0 x。
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float | None = None,
                 dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA 秩必须为正，收到 r={r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        # alpha 默认取 2r：这是 LoRA 论文与社区常用做法，使得改 r 时
        # 缩放 scale = alpha/r 大致保持不变，便于横向比较不同秩。
        self.alpha = float(2 * r if alpha is None else alpha)
        self.scale = self.alpha / self.r

        d_out, d_in = base.out_features, base.in_features
        self.lora_A = nn.Parameter(torch.empty(self.r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, self.r))
        # A 用 Kaiming 均匀初始化（等价于论文里的高斯初始化）：
        # 只要 B 为零，A 的尺度在初始时不影响输出，但会影响后续梯度尺度，
        # 用 Kaiming 可让不同 r 下的更新幅度量级一致。
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        h = self.dropout(x)
        # 写成 (x @ A^T) @ B^T，与 base 的 Linear 口径一致
        delta = (h @ self.lora_A.t()) @ self.lora_B.t()
        return y + self.scale * delta

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


@dataclass
class LoRAStats:
    injected: int
    trainable: int
    total: int

    @property
    def ratio(self) -> float:
        return self.trainable / max(self.total, 1)

    def __str__(self) -> str:
        return (f"LoRA 注入 {self.injected} 层，可训练 {self.trainable:,} / "
                f"{self.total:,} 参数（{self.ratio*100:.3f}%）")


# ------------------------------------------------------- 视觉塔结构定位
# CLIP ViT-B/32 的注意力投影命名在不同 transformers 版本/实现下不一致：
#   HuggingFace CLIPVisionModel : encoder.layers.N.self_attn.{q,k,v,out}_proj
#   open_clip                   : ...attn.{q,k,v,out}_proj 或 .query/.key/.value
# 因此按「短名 -> 可能的叶子名」建映射，再对所有 Linear 的叶子名做匹配。
#
# ⚠️ 踩过的坑：早先写成 `("out_proj", "out")` 然后 `if full != short: want.append(full)`，
#    结果 "out" 那项的 short 与 full 恰好同名，`out_proj` 被去重逻辑吃掉，
#    导致 out_proj 匹配不到（实测 0 命中）。
_ATTN_ALIASES: dict[str, tuple[str, ...]] = {
    "q":   ("q_proj", "query", "q"),
    "k":   ("k_proj", "key", "k"),
    "v":   ("v_proj", "value", "v"),
    "out": ("out_proj", "out", "proj", "o_proj"),
    # ★ 2026-09-22 新增：CLIP ViT 里参数最多的是 MLP 块（fc1 768→3072、
    #   fc2 3072→768，合计约为注意力的 2 倍），而此前只注了注意力。
    #   匹配逻辑本来就是「按 Linear 的叶子名全塔扫」，所以加两个别名即可命中。
    #   ⚠️ 名字里带 fc1/fc2 的模块**不止 MLP 块**（有些实现里视觉塔还有别的
    #   同名 Linear），所以必须按返回的数量核对：ViT-B/32 应为 12 层 × 2 = 24 个。
    "fc1": ("fc1",),
    "fc2": ("fc2",),
}


def find_attention_projections(vision_tower: nn.Module,
                               targets: tuple[str, ...] = ("q", "k", "v", "out"),
                               verbose: bool = True) -> list[tuple[str, nn.Linear]]:
    """在视觉塔里找出注意力投影层，返回 [(完整名, Linear)]。

    实现方式是按**参数名叶子**匹配，兼容 ``self_attn.q_proj`` /
    ``attention.query`` 等不同命名习惯。
    """
    want_suffixes: list[str] = []
    for t in targets:
        key = t.strip().lower()
        if key not in _ATTN_ALIASES:
            raise ValueError(
                f"未知的注入目标 {t!r}；可选 {sorted(_ATTN_ALIASES)}")
        want_suffixes.extend(_ATTN_ALIASES[key])
    want_suffixes = list(dict.fromkeys(want_suffixes))   # 去重且保序

    found: list[tuple[str, nn.Linear]] = []
    for name, mod in vision_tower.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in want_suffixes:
            found.append((name, mod))

    if verbose:
        # 按层号统计，便于确认「12 层 x 4 投影」是否都命中
        import re
        layers = set()
        for n, _ in found:
            m = re.search(r"layers?\.(\d+)", n)
            if m:
                layers.add(int(m.group(1)))
        # ⚠️ 原文写的是「注意力投影」，但 2026-09-22 起 targets 里可含
        #   fc1/fc2（MLP 块），再叫"注意力"就是错的信息了。
        print(f"[lora] 定位到 {len(found)} 个投影层，"
              f"覆盖 {len(layers)} 个 encoder 层"
              f"（目标 {','.join(targets)}）")
        if not found:
            leaves = sorted({n.rsplit('.', 1)[-1]
                             for n, m in vision_tower.named_modules()
                             if isinstance(m, nn.Linear)})
            print(f"[lora] ⚠️ 未命中。视觉塔内 Linear 的叶子名实际为：{leaves}")
            print(f"[lora]    匹配集合为 {want_suffixes}，请据此补充 _ATTN_ALIASES")
    return found


# ------------------------------------------------------------------ 注入
def inject_lora(model, r: int = 16, alpha: float | None = None,
                dropout: float = 0.0,
                targets: tuple[str, ...] = ("q", "k", "v", "out"),
                last_n_layers: int = 0,
                verbose: bool = True) -> LoRAStats:
    """给 CLIP 的**视觉塔**注入 LoRA 旁路，并冻结其余全部参数。

    只改视觉塔：本任务是图像分类，文本塔不参与（分类头直接接在图像嵌入上）。
    冻掉文本塔还能顺带省下它的优化器状态。

    返回 :class:`LoRAStats`。
    """
    vt = getattr(model, "vision_model", None) or getattr(model, "visual", None)
    if vt is None:
        raise AttributeError(
            "找不到视觉塔（vision_model / visual）。"
            f"顶层可用属性：{[n for n, _ in model.named_children()]}")

    found = find_attention_projections(vt, targets, verbose=verbose)
    if not found:
        raise RuntimeError(
            "没有匹配到任何注意力投影层，LoRA 无处注入。"
            f"请检查 targets={targets} 与模型命名。")

    # ★★★ 2026-10-10 注入位置限制（`--lora-last-n`）：只给**最后 N 个 encoder 层**注入。
    #   动机：全仓 93 处 `--targets` 从未变过（注入范围恒＝全 12 层），
    #   而 `docs/官方规则/骨干魔改合规核对_20261006.md` 点名的"魔改第一枪"
    #   ＝"**同样容量放到不同层**"。默认 0 ⇒ 全层注入（与历次臂逐字一致，零影响）。
    #   ⚠️ 出包侧（tta_lora_eval）必须**按 ckpt 的 args 重放**同一过滤，
    #      否则 ckpt 里的 lora_* 键没有落点 ⇒ `strict=False` 静默丢弃 ⇒ 预测塌缩。
    if int(last_n_layers or 0) > 0:
        import re as _re
        _lid = {}
        for _n, _m in found:
            _mm = _re.search(r"layers?\.(\d+)", _n)
            if _mm:
                _lid[_n] = int(_mm.group(1))
        if _lid:
            _maxl = max(_lid.values())
            _from = _maxl - int(last_n_layers) + 1
            _n_before = len(found)
            found = [(_n, _m) for _n, _m in found if _lid.get(_n, -1) >= _from]
            if verbose:
                _kept = sorted({_lid[_n] for _n, _ in found if _n in _lid})
                print(f"[lora] ★ 注入位置限制：仅后 {int(last_n_layers)} 层"
                      f"（层号 {_kept}，投影层 {len(found)}/{_n_before}）")
            if not found:
                raise RuntimeError(
                    f"注入位置过滤后为空（last_n_layers={last_n_layers}）")

    # 先整体冻结，再逐个替换成 LoRA 版本（替换时旁路参数默认 requires_grad=True）
    for p in model.parameters():
        p.requires_grad = False

    total_before = sum(p.numel() for p in model.parameters())

    # ★ 新创建的 lora_A / lora_B 默认落在 CPU，而模型可能已经在 GPU 上。
    # 若不对齐设备，前向会直接抛
    #   RuntimeError: Expected all tensors to be on the same device
    # （实测踩到）。这里取基础权重所在设备作为旁路设备。
    # dtype 一并对齐：半精度推理时若旁路是 fp32，会触发隐式类型提升。
    base_device = next(model.parameters()).device
    base_dtype = next(model.parameters()).dtype

    # 按父模块 + 叶子名逐个 setattr 替换
    for full_name, linear in found:
        parent_path, _, leaf = full_name.rpartition(".")
        parent = vt
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        new_mod = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        new_mod = new_mod.to(device=base_device, dtype=base_dtype)
        setattr(parent, leaf, new_mod)

    stats = LoRAStats(
        injected=len(found),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total=total_before,
    )
    if verbose:
        print(f"[lora] {stats}")
        est_mb = stats.trainable * 14 / 1024 / 1024   # fp16权重2 + Adam m,v 8 + fp32 master 4
        print(f"[lora] 训练态参数显存估算 ≈ {est_mb:.1f} MB"
              f"（fp16 权重 + Adam 两阶矩 + fp32 主副本）")
    return stats


def lora_parameters(model, strict: bool = True):
    """返回可训练参数。

    ``strict=True``（默认）只返回**真正的 LoRA 旁路参数**，即名字里带 ``lora_``
    的项。这是优化器该要的东西 —— 分类头与 ``logit_scale`` 有各自的学习率，
    不能混进这一组。

    ⚠️ 踩过的坑：早先的实现是 ``[p for p in model.parameters() if p.requires_grad]``，
    看似等价，实际上会把**分类头与 logit_scale 也一并收进来**（它们同样
    requires_grad）。当时调用方按「lora / head / logit_scale」分三组建优化器，
    结果 head 与 logit_scale 先被 lora 组抢走、再从自己组里被去重剔掉，
    **两类参数最终用了 LoRA 的小学习率**而不是设计的 10 倍 —— 训练能跑但
    与设计不符，属于静默错误。现在改为按名字精确判定。

    ``strict=False`` 保留旧的「所有可训练参数」语义，便于需要时取全集。
    """
    if not strict:
        return [p for p in model.parameters() if p.requires_grad]
    out = []
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" in name:
            out.append(p)
    return out


def mark_only_lora_trainable(model) -> None:
    """确保除 LoRA 旁路外全部冻结（幂等，可在任意时刻调用校正）。"""
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name


def lora_state_dict(model) -> dict:
    """抽取纯 LoRA 权重，便于单独保存（几十 MB 而不是整模型 600MB）。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_" in k}


def merge_lora(model) -> int:
    """把 LoRA 旁路吸收进基础权重：``W0' = W0 + scale * B @ A``。

    推理时合并不改变输出（是恒等变换），但能省掉每次前向的额外矩阵乘，
    也让「最终交付的就是一份标准 CLIP 权重」这件事更直观。
    返回合并的层数。
    """
    n = 0
    for mod in model.modules():
        if not isinstance(mod, LoRALinear):
            continue
        with torch.no_grad():
            base = mod.base
            delta = (mod.lora_B @ mod.lora_A) * mod.scale
            base.weight.add_(delta.to(base.weight.dtype))
            if base.bias is not None:
                pass  # LoRA 不产生 bias，无需处理
        n += 1
    return n


# ------------------------------------------------------------- 显存预算
def estimate_vram_mb(vit_params_m: float = 87.5, r: int = 16, batch: int = 96,
                     grad_ckpt: bool = False, num_layers: int = 12,
                     dim: int = 768, img_tokens: int = 50,
                     bytes_per_param_act: int = 2) -> dict:
    """按公式估算一次前向+反向的峰值显存（MB）。

    刻意做成**纯算术**而不是实测：这样换 batch / 换 r 时可以立刻给出量级判断，
    不必每次真跑一遍。所有项都给出明细，便于核对。

    各项含义：
      * 权重        ：fp16 存 ViT-B/32 全部参数（87.5M × 2B）
      * 优化器状态  ：仅 LoRA 参数（fp16权重2B + Adam m,v 8B + fp32 master 4B = 14B/param）
      * 梯度        ：LoRA 参数的 fp16 梯度
      * 激活        ：每层需要保存的中间张量，与 batch × token 数成正比。
                      开启梯度检查点后只存层边界，约降为 1/sqrt(L)。
      * patch embed ：输入图像张量本身（batch × 3 × 224 × 224）
    """
    P = vit_params_m * 1e6
    lora_params = num_layers * 4 * (r * dim + dim * r)   # q,k,v,out 四个投影

    w_weights = P * bytes_per_param_act / 1024 / 1024
    w_opt = lora_params * 14 / 1024 / 1024
    w_grad = lora_params * bytes_per_param_act / 1024 / 1024

    # 激活：每层约 若干中间张量 × batch × tokens × dim
    per_layer_tensors = 10 if not grad_ckpt else 2
    act_per_layer = (batch * img_tokens * dim * per_layer_tensors
                     * bytes_per_param_act) / 1024 / 1024
    w_act = act_per_layer * (num_layers if not grad_ckpt else 2)
    w_input = batch * 3 * 224 * 224 * bytes_per_param_act / 1024 / 1024

    items = {
        "权重（fp16 全模型）": w_weights,
        "优化器状态（仅 LoRA）": w_opt,
        "梯度（仅 LoRA）": w_grad,
        "激活": w_act,
        "输入图像张量": w_input,
    }
    items["合计"] = sum(items.values())
    items["_lora_params"] = lora_params
    return items


def print_vram_budget(r: int = 16, batches: tuple[int, ...] = (32, 64, 96, 128, 256),
                      limit_mb: float = 6000.0) -> None:
    """打印不同 batch 下的显存预算表，标出哪些能塞进 6GB。"""
    print(f"\n{'batch':>6} | {'激活':>10} | {'合计':>10} | 6GB 可用性")
    print("-" * 46)
    for b in batches:
        d = estimate_vram_mb(r=r, batch=b)
        act = d["激活"]
        tot = d["合计"]
        # 留出约 1.2GB 给 CUDA 上下文 / 显存碎片
        ok = "✅ 可容纳" if tot < limit_mb - 1200 else "❌ 超限"
        print(f"{b:>6} | {act:>8.0f}MB | {tot:>8.0f}MB | {ok}")
    lora_params = estimate_vram_mb(r=r)["_lora_params"]
    print(f"\nLoRA r={r} 可训练参数 {lora_params:,.0f}，"
          f"训练态约 {lora_params * 14 / 1024 / 1024:.0f} MB")
