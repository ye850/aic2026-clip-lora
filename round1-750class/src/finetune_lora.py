# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""端到端 LoRA 微调：解冻 CLIP 视觉骨干 + 分类头联合训练。

与 ``train_head.py`` 的关系
---------------------------
``train_head.py`` 走的是「冻结骨干 + 缓存特征 + 只训线性头」路线，用于**低成本
消融**（每轮秒级，可跑几十组）。本脚本走的是「LoRA 解冻骨干 + 端到端」路线，
用于**验证容量假设**（每轮约 5 分钟量级，实测值）。

两者共用同一套损失（``noise.compute_loss``）与同一套权重（CL 降权 x 类平衡），
因此结果可以直接对比，回答的问题是：

    把「容量」这一项从线性头换成 LoRA 骨干，能否带来可观测的收益？

动机：一条待检验的容量假设
--------------------------
前一阶段的真实榜分是：全套抗噪 52.04  vs  朴素交叉熵 49.80，**Δ 仅 +2.24**。
一整套抗噪方法只值 2.24 分，一个自然的解释是骨干冻结时「抗噪」没有施展空间 ——
一个线性头既要去拟合数百类的细粒度边界、又要抵抗约 38% 的标签噪声，参数不够。
LoRA 给骨干 0.77% 的可调参数（118 万），既保留预训练的强特征，又留出细调余地。

⚠️ **但这条假设在本阶段的受控对照中并未被证实。** 关键修正：第一版对照让冻结线
多训了 20 倍轮数，于是「LoRA 落后且显著」（z=2.40）；**把两边轮数对齐后**，
差距为 −0.0200，95% 置信区间跨 0，**不显著**。
⇒ 本脚本作为**并行探索**保留，不承担提交主链路；提交主链路仍是
「冻结骨干 + 鲁棒线性头」。详见 ``scripts/run_ab_harness.py``。

规则合规性
----------
* 只用 CLIP ViT-B/32 + OpenAI 官方权重：基础权重来自本地 ``models/`` 下那一份官方
  快照，LoRA 旁路是本队自训参数，不是任何外部预训练模型。
* 单一模型，无集成：全程只有一个模型、一条推理路径。
* 测试数据不参与训练：训练只读 ``train/``；``test/`` 仅在最后推理时读一次。
* 自动清洗，非人工：噪声处理由 CL 权重 + 鲁棒损失自动完成，无人工标注环节。

用法
----
    # 先做小规模冒烟（2000 张 / 1 轮，约 1 分钟），确认能跑通
    python src/finetune_lora.py --limit 2000 --epochs 1

    # 正式训练
    python src/finetune_lora.py --epochs 6 --r 16 --batch 96

    # 用训练好的权重推理并生成提交
    python src/finetune_lora.py --predict-only --ckpt outputs/cache/lora_r16.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import lora as LO
import noise as NZ
import vpt as VPT
from data import build_test_manifest, build_train_manifest, load_image
from extract import build_transform, load_clip, take_embeds


# ------------------------------------------------------------------ Dataset
class _QualityDegrade:
    """随机画质退化 —— 把训练图压成「官方测试集那种糊图」。

    ★ 为什么加这一层（有实测依据，不是"通用增强"）
    ---------------------------------------------------------------
    官方测试集被**统一压到长边 500px 并重压缩**（体积中位 38KB，训练是 85KB），
    而训练侧原先**完全没有画质增强**。实测代价（`scripts/probe_quality_drop.py`，
    同一份留出集、同一份权重）：

        变体                    LoRA Δacc    冻结头 Δacc
        压到长边 500px           −0.0001      −0.0024   ← 尺寸统一几乎不花钱
        压到 500 + JPEG q70      −0.0195      −0.0259
        只 JPEG q50              −0.0196      −0.0324   ← 代价**全在重压缩**上

    ⇒ 这是**训练/测试的画质不匹配**，对齐它就是把那 2 分收回来。
      注意"长边 500"本身不值钱（CLIP 预处理是短边缩 224 + 中心裁剪，
      **裁剪框在归一化坐标下不变**），所以**关键动作是重压缩**。

    ★ 两个设计选择
    - **按概率施加**（默认 p=0.5），不是全部退化：保留一半干净样本，
      免得把已经学到的原图画质能力换掉（`scripts/quality_align.py` 里
      专门有一列就是查这个副作用）。
    - **质量随机**（默认 q∈[40,90]）：官方用的具体质量未知，
      扫一个区间比赌一个值稳。
    - **只在长边 > max_side 时才降采样**：本就有 21.7% 的训练图短边 <300，
      再压一次就成了色块，反而制造坏样本。

    ★ 必须是**模块级类**（Windows 下 DataLoader 用 spawn，transform 要能 pickle），
      并自带 `random.Random(seed)` 实例：这样在 `--workers 0` 下
      **增强序列完全确定**，与本项目"位级可复现"的要求一致。
      ⚠️ **现状（2026-09-23 更正）**：`--workers` 已经不是 0 了 ——
        run2（288）用 4、run4 用 4、早前的 run3 用过 8。此时**每个 worker 各持一份
        从同一 seed 起步的副本**，于是"第 k 个样本的退化决策"在**各 worker 之间是相关的**
        （worker0 的第 1 张、worker1 的第 1 张…会得到同一个决策）。
        这**不是正确性问题**（每张图仍有 p 的概率被退化、q 仍在一个区间里随机），
        但严格说混入了一层 worker 间的相关性。
        ⇒ 要真正干净，得把种子按 (worker_id, 样本号) 打散；本次**没改**，
          因为一改就等于换掉 run2 的增强轨迹、与已跑完的对照不可横比。
    """

    def __init__(self, p: float = 0.5, max_side: int = 500,
                 q_lo: int = 40, q_hi: int = 90, seed: int = 0):
        self.p = float(p)
        self.max_side = int(max_side)
        self.q_lo = int(q_lo)
        self.q_hi = int(q_hi)
        self.rng = __import__("random").Random(seed)
        self.n_applied = 0
        self.n_seen = 0

    @classmethod
    def parse(cls, spec: str) -> "_QualityDegrade":
        """`spec` 形如 ``p=0.5,max_side=500,q=40-90``（都可省略）。"""
        kw: dict[str, float] = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition("=")
            k, v = k.strip(), v.strip()
            if k == "q":
                lo, _, hi = v.partition("-")
                kw["q_lo"], kw["q_hi"] = int(lo), int(hi or lo)
            elif k in ("p", "max_side", "seed"):
                kw[k] = float(v) if k == "p" else int(v)
            else:
                raise ValueError(f"[degrade] 无法识别参数 {k!r}（{spec!r}）")
        return cls(**kw)

    def __call__(self, im):
        import io

        from PIL import Image as _I

        self.n_seen += 1
        if self.rng.random() >= self.p:
            return im
        w, h = im.size
        if self.max_side and max(w, h) > self.max_side:
            s = self.max_side / max(w, h)
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                           _I.BICUBIC)
        q = self.rng.randint(self.q_lo, self.q_hi)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        self.n_applied += 1
        return _I.open(buf).convert("RGB")


class _SizeRandomize:
    """随机尺寸归一化 —— 把训练图的**长边**随机对齐到若干档位。

    ★ 为什么加这一层（H2，2026-10-04）
    ---------------------------------------------------------------
    全量实测（`scripts/_gen_stats.py`，训练 148,695 张 / 测试 37,444 张）：

        集合      短边中位   长边中位   长边==500   长边>500   长边 max
        测试集      375       500       99.4%      0.0%       500
        训练集      480       640         —       66.59%      8688

    ⇒ 官方对**测试集做了统一的长边归一化（压到 500）**，而训练侧从来没做过这件事。

    ★★ 但**不能简单"统一缩到 500"** —— 那是 `q75pack` 的同构做法，已被榜分否证：
         `q75pack`（全部固定压到 q75，收窄到一个点） 榜分 **−1.88pp** ❌
         `deg`（随机 q70~80，扩宽成一个区间）       榜分 **+0.26pp** ✅
       ⇒ 规律是「**域对齐要扩宽覆盖面，而不是收窄到一个点**」。
       而且源码上方的 `_QualityDegrade` 注释里也写明了：「长边 500 本身不值钱
       （−0.0001），代价全在重压缩上」。

    ⇒ 所以本类**不做**"统一压到 500"，而是**在多个档位间随机**，让模型见过
      "从原图到测试档位"的整个尺寸区间，把测试集那个 500 变成**区间内的一个点**
      而不是一个需要外推的边界值。

    ★ 设计选择（与 `_QualityDegrade` 逐条对齐，便于对照）
    - **按概率施加**（默认 p=0.5）：留一半原样，不把已学到的原生尺寸能力换掉。
    - **档位随机**（默认 [500,640,800]）：500 = 测试档位，640 = 训练集长边中位，
      800 = 训练集上四分位附近 ⇒ 覆盖"测试档位 ～ 训练主区间"。
    - **只缩不放**：`s = L / max(w,h)`，`s < 1` 才缩放。理由与 `deg` 相同 ——
      本就有相当比例的训练图偏小，放大只会得到插值糊图，反而制造坏样本。
    - **只动几何尺寸、不动画质**：不重压缩、不转 JPEG。这样它与 `--degrade`
      是**正交的两维**，可以单独判读，也可以叠加。
    - **模块级类 + 自带 RNG**：Windows 下 DataLoader 用 spawn，transform 要能 pickle。
      （worker 间相关性这一已知瑕疵与 `_QualityDegrade` 完全一致，此处**故意不修**，
        以保证两者的"增强轨迹随机性"可比。）

    ⚠️ **绝对不动 `RandomResizedCrop` 的 `scale`**：那个被 TTA 的 `rc0..rc15`
       视图当作"训练分布"引用，一改那 16 个视图全部变分布外、历史 TTA 结论作废。
       本类作用在 **RRC 之前**，改变的是"进入 RRC 的这张图有多大"，
       而 RRC 的 `scale=(0.7,1.0)` 语义（裁原图的 70%~100% 面积）**逐字保持**。
    """

    def __init__(self, p: float = 0.5, levels: tuple[int, ...] = (500, 640, 800),
                 seed: int = 0):
        self.p = float(p)
        self.levels = tuple(int(x) for x in levels)
        assert self.levels, "[size-rand] levels 不能为空"
        self.rng = __import__("random").Random(seed)
        self.n_applied = 0
        self.n_seen = 0
        # 每档实际命中计数（跑完打印，用于自证"三档真的都用上了"）
        self.hits = {int(x): 0 for x in self.levels}

    @classmethod
    def parse(cls, spec: str) -> "_SizeRandomize":
        """`spec` 形如 ``p=0.5,levels=500-640-800``（都可省略）。"""
        kw: dict[str, object] = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition("=")
            k, v = k.strip(), v.strip()
            if k == "levels":
                kw["levels"] = tuple(int(x) for x in v.split("-") if x.strip())
            elif k == "p":
                kw["p"] = float(v)
            elif k == "seed":
                kw["seed"] = int(v)
            else:
                raise ValueError(f"[size-rand] 无法识别参数 {k!r}（{spec!r}）")
        return cls(**kw)

    def __call__(self, im):
        from PIL import Image as _I

        self.n_seen += 1
        if self.rng.random() >= self.p:
            return im
        w, h = im.size
        L = self.rng.choice(self.levels)
        m = max(w, h)
        # ★ 只缩不放：本就比目标档位小的图，保持原样（避免插值糊图）
        if m > L:
            s = L / m
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                           _I.BICUBIC)
        self.hits[int(L)] = self.hits.get(int(L), 0) + 1
        self.n_applied += 1
        return im

    def stats(self) -> str:
        tot = sum(self.hits.values()) or 1
        hh = " ".join(f"{k}:{v}({v * 100.0 / tot:.1f}%)"
                      for k, v in sorted(self.hits.items()))
        return (f"[size-rand] 施加 {self.n_applied}/{self.n_seen}"
                f"（{self.n_applied * 100.0 / max(1, self.n_seen):.1f}%）"
                f" p={self.p} levels={self.levels} 命中 {hh}")


class LabeledImageDataset(Dataset):
    """带标签的图像数据集（端到端微调用，产出原图张量而非特征）。

    与 ``extract.ImageOnlyDataset`` 的区别：那个只需要图（提特征），
    这个还需要标签、并且要支持**训练期数据增强**。

    数据增强只做「随机水平翻转 + RandomResizedCrop」这类**保标签**的几何变换。
    刻意不做颜色抖动、Cutout 之类的强增强 —— 本任务已经带 33% 标签噪声，
    再叠一层强增强会让训练信号过于混乱，反而掉点。

    ★ 2026-09-23 补充：上面的判断**并未被实测验证过**，而且当时没考虑
    「训练集与官方测试集的成像差异」。现在 `--aug strong` 提供一组**只动成像、
    不动几何**的轻度增强（高斯模糊 + 轻颜色扰动 + 2% 灰度），
    依据是 `quality_drop.json` 实测的画质退化代价 −1.96 分。默认仍是 `plain`，
    即"上面那段原判断"，所以历史基线不受影响 —— 要验就得显式开。

    ``weights`` 给出时，``__getitem__`` 额外返回索引 ``i``，供训练循环套用
    CL 保真度降权 x 类平衡的**逐样本权重**。这一点很关键：端到端微调若不套
    权重，等于把已验证有效的抗噪机制（前一阶段实测 +2.24 分）丢掉。
    """

    def __init__(self, paths: list[str], labels: np.ndarray, train: bool = True,
                 weights: np.ndarray | None = None, degrade: str | None = None,
                 aug: str = "plain", size_rand: str | None = None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = train
        self.degrade_spec = degrade
        self.size_rand_spec = size_rand
        self.aug = aug
        self.tf = self._build_tf(degrade, aug, size_rand)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.n_failed = 0

    @staticmethod
    def _build_tf(degrade: str | None = None, aug: str = "plain",
                  size_rand: str | None = None):
        from torchvision import transforms as T
        from extract import CLIP_MEAN, CLIP_STD
        ops = []
        # ★ 画质退化放在**几何变换之前**：先退化成"测试集那种糊图"，
        #   再做裁剪/翻转，等价于"官方把图压好之后我们正常处理"。
        if degrade:
            ops.append(_QualityDegrade.parse(degrade))
        # ★★ 2026-10-04 新增 H2「随机尺寸归一化」。
        #   顺序刻意放在 **degrade 之后、RRC 之前**：
        #     degrade 管"画质维"（JPEG 重压缩），本层管"尺寸维"（长边档位），
        #     两者正交，谁先谁后不影响语义；放在 RRC 之前是**硬要求**，
        #     因为 RRC 的 `scale=(0.7,1.0)` 是对"刚传进来的图"取面积的。
        if size_rand:
            ops.append(_SizeRandomize.parse(size_rand))
        ops.append(T.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0),
                                       interpolation=T.InterpolationMode.BICUBIC))
        ops.append(T.RandomHorizontalFlip(p=0.5))
        # ★★ 2026-09-23 新增 `--aug strong`。
        #   动机（都有实测依据，不是"加增强总能涨"的经验主义）：
        #     · `probe_domain_gap.py`：测试图短边中位 375、**长边恒 500**、
        #       体积中位 38KB，而训练图 480/640/85KB ⇒ 测试图更小、**被重压缩过**；
        #     · `quality_drop.json`：把 val 压到 500+JPEG 就掉 **1.96 分**
        #       ⇒ 模型对"画质退化"没有免疫力。
        #   所以这里补两类**只针对成像退化、不动几何**的增强：
        #     ① 轻微高斯模糊 —— 让模型见过"多一次降采样"的输入；
        #     ② 轻量颜色扰动 —— 对应 JPEG 色度子采样带来的偏色。
        #   ⚠️ **几何一律不动**（仍是 scale=0.7~1.0 + hflip）。理由：
        #     TTA 的 `rc0..rc15` 视图被刻意定义成"与训练分布逐字一致"，
        #     一旦放宽 scale，那 16 个视图立刻变成"分布外视图"，
        #     前面所有 TTA 结论全部作废。改几何的代价远大于收益。
        if aug == "strong":
            ops.append(T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.30))
            ops.append(T.ColorJitter(brightness=0.3, contrast=0.3,
                                     saturation=0.2, hue=0.02))
            ops.append(T.RandomGrayscale(p=0.02))
        ops += [
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
        return T.Compose(ops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = load_image(self.paths[i])
        if im is None:
            self.n_failed += 1
            from PIL import Image
            im = Image.new("RGB", (C.IMG_SIZE, C.IMG_SIZE), (127, 127, 127))
        t = self.tf(im)
        if self.weights is not None:
            return t, int(self.labels[i]), i
        return t, int(self.labels[i])


@torch.no_grad()
def eval_transform(paths: list[str]):
    """验证/推理用的确定性预处理（短边 224 + 中心裁剪）。"""
    from torchvision import transforms as T
    from extract import CLIP_MEAN, CLIP_STD
    return T.Compose([
        T.Resize(C.IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(C.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ------------------------------------------------------------------- 模型
class LoRAClipClassifier(nn.Module):
    """CLIP 视觉骨干（LoRA 微调）+ 线性分类头。

    ``forward`` 返回分类 logits。头与 ``RobustHead`` 同构（含可学习 logit 温度），
    保证与「冻结骨干」路线出来的分数可比。
    """

    def __init__(self, clip_model, backbone_kind: str, num_classes: int,
                 feat_dim: int = C.FEAT_DIM, init_scale: float = 100.0,
                 hidden: int = 0):
        super().__init__()
        self.clip = clip_model
        self.backbone_kind = backbone_kind
        if hidden and hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
        else:
            self.head = nn.Linear(feat_dim, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(init_scale))))

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # ★ 位置编码插值：CLIP ViT-B/32 的位置编码是 7×7（对应 224/32），
        #   当 IMG_SIZE 升到 288/336 时 patch 网格变成 9×9/11×11，
        #   不插值 transformers 会直接抛 shape 不匹配。
        #   `interpolate_pos_encoding=True` 是官方支持的路径（双三次插值）。
        interp = C.IMG_SIZE != 224
        if self.backbone_kind == "transformers":
            f = take_embeds(self.clip.get_image_features(
                pixel_values=pixel_values, interpolate_pos_encoding=interp))
        else:
            f = self.clip.encode_image(pixel_values)
        return f

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feat(pixel_values)[0]

    def forward_with_feat(self, pixel_values: torch.Tensor):
        """返回 ``(logits, f_norm)``。

        ★ 为什么要有这个并行入口（2026-09-28）：``--drift-lambda``（表征漂移约束）
          需要**同一次前向**同时给出 logits 与 L2 归一化特征，否则就得在训练循环里
          把 forward 的计算复制一遍 —— 那种复制一旦与原 forward 漂移，就会变成
          "训练用的前向与评估用的前向不是同一个"的**静默错误**。
          本方法把两者收进单一真源，``forward`` 转调它，**对既有调用完全等价**。
        """
        f = self.encode(pixel_values)
        # ★ 特征必须 L2 归一化后再进头。
        # 理由：冻结路线里缓存的特征是归一化过的（extract 末尾做了 /norm），
        # 头在这套尺度上训练。端到端时若不归一化，输入尺度会随骨干权重漂移，
        # 与头预期的量级不一致 —— 这条对齐是两条路线可比的前提。
        f = F.normalize(f.float(), dim=-1)
        z = self.head(f)
        return self.logit_scale.clamp(max=math.log(100.0)).exp() * z, f


def _margin_logits(logits: torch.Tensor, y: torch.Tensor, m: float) -> torch.Tensor:
    """CosFace 的 **logit 形式**：只把**正确类**的 logit 压低 ``m``。

    ★ 为什么做这个（2026-09-28）
    ----------------------------
    本项目已用数据证明：错的主体是**细粒度类间重叠**、不是随机标签噪声 ——
    错分样本的真实类 ``rank(y)`` 中位数只有 **5.0**（若为随机噪声应 ≈376）。
    也就是说正确答案几乎总在 top-5 里，模型缺的是"把正确类与那几个竞争类
    拉开间隔"的能力。而冠军配方 SCE 里占 1/3 权重的 RCE（``−Σp·logp``，
    不含标签、只把预测推向均匀）方向恰好相反。
    ⇒ margin 直接拉这个间隔，落在官方"包括但不限于…鲁棒损失函数"的范畴。

    ★ 零回归风险
    ------------
    只作用于**训练损失的输入**；推理/评估路径（``forward``）一个字不改 ——
    推理时不需要 margin，拉开的间隔已经固化在权重里。
    本函数 **不原地修改** ``logits``（先 clone），所以 mixup 的两个方向
    可以各用各的 margin 视图。

    ★ 单位
    ------
    ``m`` 以**余弦空间**计（典型 0.1~0.5）。内部按 ``m × logit_scale`` 施加：
    logits 被 ``logit_scale``（初始 100）放大过，直接用 m 会被稀释到无效。
    """
    if m <= 0:
        return logits
    z = logits.clone()
    idx = torch.arange(z.size(0), device=z.device)
    z[idx, y] = z[idx, y] - m
    return z


# ---------------------------------------------------------------- 训练循环
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _ema_apply(pairs: list) -> list:
    """把 EMA 影子权重的当前值灌进模型，返回被覆盖权重的备份。

    ``pairs`` 是 ``[(param, shadow_fp32), ...]``。备份是**必须**的：验证只是
    临时借用 EMA 权重，验完要还原成训练中的真实权重，否则后半程的训练就从
    EMA 权重继续了 —— 那等于每验一次就把模型拉回平均值，训练直接被搅乱，
    而且不会报错（只会表现为"训不动"）。
    """
    backup = [(p, p.detach().clone()) for p, _ in pairs]
    with torch.no_grad():
        for p, t in pairs:
            p.copy_(t)
    return backup


def _ema_restore(backup: list) -> None:
    with torch.no_grad():
        for p, b in backup:
            p.copy_(b)


def train(args) -> int:
    _seed = C.SEED if getattr(args, "seed", None) is None else args.seed
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    device = args.device
    if device == "cuda":
        torch.cuda.manual_seed_all(_seed)

    # ---------------------------------------------------------- 数据清单
    tr = build_train_manifest()
    K = int(tr.attrs.get("num_classes", 0))
    paths = tr["path"].tolist()
    labels = tr["label"].to_numpy().astype(np.int64)

    # ------------------------------------------------ K 折 OOF 切分（可选）
    # ★ 用途：K 折留出（Out-Of-Fold）验证。当前折作为 holdout **排除出训练集**，
    #   并把它的路径+标签落盘（供 scripts/oof_fold_label.py 打标），剩余 K-1 折训练。
    #   目的：拿到「微调骨干对训练同域未见样本的预测」，测置换 skew 能否 >0.65
    #   （判断"改标签"是否系统性、可救）。详见 outputs/追90分_OOF方案.md。
    # ★ 折切分必须**按类别分层**（每类内按 seed 打乱后轮流分入 K 折），
    #   保证每折的类分布均衡（官方测试集类别均衡，holdout 也得均衡才有代表性）。
    # ★ 与 val 切分**独立**：val 仍从「剩余 K-1 折」里按 --val-per-class 再切，
    #   用于选 epoch；holdout 折绝不被训练触碰、也不参与 val。
    if getattr(args, "fold", -1) >= 0:
        _fold = int(args.fold)
        _Kfold = int(getattr(args, "k", 5))
        if not (0 <= _fold < _Kfold):
            raise SystemExit(f"[fold] 🛑 --fold={_fold} 须落在 [0,{_Kfold})")
        _frng = np.random.RandomState(int(getattr(args, "fold_seed", 20260926)))
        _fold_id = np.empty(len(labels), dtype=np.int64)
        for _kk in range(K):
            _idx = np.where(labels == _kk)[0]
            _order = _frng.permutation(len(_idx))
            _fold_id[_idx] = _order % _Kfold          # 每类样本轮流分入 K 折
        _ho_mask = _fold_id == _fold
        _ho_paths = [p for p, m in zip(paths, _ho_mask) if m]
        _ho_labels = labels[_ho_mask]
        # holdout 落盘（路径 + 标签，供打标脚本读）
        import json as _json
        _oof_dir = Path(args.ckpt).parent / "oof"
        _oof_dir.mkdir(parents=True, exist_ok=True)
        (_oof_dir / f"fold{_fold}_paths.json").write_text(
            _json.dumps(_ho_paths), encoding="utf-8")
        (_oof_dir / f"fold{_fold}_paths.json.labels").write_text(
            _json.dumps(_ho_labels.tolist()), encoding="utf-8")
        # 训练集 = 排除 holdout 折
        paths = [p for p, m in zip(paths, _ho_mask) if not m]
        labels = labels[~_ho_mask]
        print(f"[fold] ★ K 折 OOF：当前 fold={_fold}/{_Kfold}，"
              f"holdout {len(_ho_paths)} 张（落盘 {_oof_dir}/fold{_fold}_paths.json），"
              f"训练集剩余 {len(paths)} 张")

    if args.limit and args.limit < len(paths):
        # ★ 分层抽样而非截断前 N 条：训练集按类别文件夹排好序，
        # 直接切片只会拿到前几个类别，"小规模冒烟"就完全失去代表性。
        rng = np.random.RandomState(C.SEED)
        keep = []
        per = max(1, args.limit // max(K, 1))
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            sel = rng.choice(idx, size=min(per, len(idx)), replace=False)
            keep.append(sel)
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        paths = [paths[i] for i in keep]
        labels = labels[keep]
        print(f"[data] 分层抽样 {len(paths)} 张（覆盖 {len(np.unique(labels))}/{K} 类）")
    else:
        print(f"[data] 全量 {len(paths)} 张 / {K} 类")

    # 训练/验证切分（验证用于挑 epoch；含噪标签上的 val_acc 只作参考，
    # 真正的判据是排行榜 —— 见 docs/性能与自检记录.md 的说明）
    #
    # ★ 用「每类至少留 k 张」的分层切分，而不是全局比例。
    # 踩过的坑：冒烟时只有 216 张，val_ratio=0.05 只切出 11 张验证集 ——
    # 16 个类摊下来每类不到 1 张，val_acc 只能取到 0/1 两三个离散值，
    # 「按 val_acc 挑最优 epoch」等于在噪音上挑，会把训练带偏。
    rng = np.random.RandomState(C.SEED)
    val_mask = np.zeros(len(labels), dtype=bool)
    per_class_min = args.val_per_class
    _floor_cls = []          # 触发了"保底留 1 张训练图"的类（跑完报数自证）
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        want = min(len(idx), max(per_class_min,
                                 int(round(len(idx) * args.val_ratio))))
        # ★★ 2026-10-04 修：**保底每类至少留 1 张训练图**。
        #   原逻辑 `want = min(len(idx), ...)` 在 `len(idx) <= per_class_min` 时
        #   会把该类的**全部**样本划入 val ⇒ **训练集里这个类一张都不剩**。
        #   实测（本地 148,695 张 / 750 类）：**有 1 个类只有 5 张样本**，
        #   被整类划走 ⇒ 该类的 750 维 logit 在训练里**从未被正例激活过**。
        #   ⚠️ 影响量级：5 / 148695 = 0.0034% ⇒ 对榜分预期影响约 **0.003pp**，
        #      **不是** 6.44pp 缺口的原因；但这是个真 bug，顺手修掉。
        #   ⚠️ 只在"会出事"时才收敛 want，**其余类的切分逐字不变**
        #      ⇒ val 规模 7537 → 7536（只差那 1 张），历史读数基本可横比。
        if want >= len(idx) and len(idx) > 1:
            want = len(idx) - 1
            _floor_cls.append((k, len(idx)))
        sel = rng.choice(idx, size=want, replace=False)
        val_mask[sel] = True
    if _floor_cls:
        print(f"[val] ⚠️ 为 {len(_floor_cls)} 个类保底留了训练图（否则该类训练样本为 0）："
              f"{[f'类{k}(n={n})' for k, n in _floor_cls[:5]]}"
              f"{' …' if len(_floor_cls) > 5 else ''}")
    tr_paths = [p for p, m in zip(paths, val_mask) if not m]
    tr_labels = labels[~val_mask]
    va_paths = [p for p, m in zip(paths, val_mask) if m]
    va_labels = labels[val_mask]

    # ------------------------------------------------ 伪标签回收（标签替换）
    # 设计见 scripts/pseudo_label.py 头注释：第一信号（TTA 打标）+ 第二信号
    # （特征最近邻自洽）交叉后的回收清单，把训练子集内疑似噪声标签换成伪标签。
    # ★ 三道守卫（少一条都可能静默污染训练）：
    #   ① **按路径匹配** —— manifest 类内不排序，行号对齐会静默错位
    #   ② 命中数必须等于清单大小（漏命中 = 路径口径漂移，宁可中止）
    #   ③ 清单样本若落在 val ⇒ 中止（val 参与选优，绝不被伪标签触碰）
    # ★ 替换发生在 val 切分之后 ⇒ val 的位置/规模与历次实验完全一致，可横比。
    if getattr(args, "pseudo", ""):
        _pj = json.loads(Path(args.pseudo).read_text(encoding="utf-8"))
        _ppaths, _plabels = _pj["paths"], _pj["pseudo_labels"]
        if len(_plabels) != len(_ppaths):
            raise SystemExit(f"[pseudo] 🛑 paths({len(_ppaths)}) 与 pseudo_labels("
                             f"{len(_plabels)}) 不等长")
        # ★ 按**文件名**匹配（跨平台）：清单可能在 Windows 上生成（形如 <盘符>:\...\xxx.jpg），
        #   云端 manifest 是 /root/.../xxx.jpg —— 全路径必然全 miss。
        #   文件名是 32 位 hex hash，全局唯一；若 manifest 内出现重名 ⇒ 中止人工看
        #   （宁可停下也不能错配标签）。
        # ★★ 09-25 事故：Linux 上 `Path("X:\\a\\b.jpg").name` 返回**整条路径**
        #   —— PosixPath 不把 `\` 当分隔符 ⇒ 953 条清单全 miss、训练当场自杀。
        #   ⇒ 取 basename 前必须先把反斜杠归一成正斜杠，绝不能直接 Path(p).name。
        def _base(_p: str) -> str:
            return os.path.basename(str(_p).replace("\\", "/"))

        _pos = {}
        for i, p in enumerate(paths):
            _b = _base(p)
            if _b in _pos:
                raise SystemExit(f"[pseudo] 🛑 manifest 文件名重复：{_b}；"
                                 f"basename 匹配要求全局唯一，中止人工核查")
            _pos[_b] = i
        _miss = [p for p in _ppaths if _base(p) not in _pos]
        if _miss:
            raise SystemExit(f"[pseudo] 🛑 {len(_miss)} 条清单路径不在训练 manifest"
                             f"（口径漂移？）例如 {_miss[0]}")
        # ★ 09-25 修正：原来这里是「清单含 val 样本 ⇒ 一律中止」。但清单的生成机
        #   与训练机不是同一台（清单在别处生成、或在 841机 上生成后换到 105机），
        #   而 val 切分随 manifest 行序变化（目录遍历顺序因文件系统而异）⇒
        #   两台机器的 val 集合必然有少量差异，清单里就会混进训练机的 val 样本。
        #   实测：quick 953 条里 18 条、formal 1246 条里 23 条（≈1.8%）。
        #   ⇒ 正确做法不是中止，而是**把这少数几条跳过不回收**：
        #     既保住「val 绝不被伪标签触碰」这条硬约束，又不会因机器漂移卡死整轮。
        #   ⇒ 只在重叠比例异常大（>20%）时才中止 —— 那才是真口径漂移。
        _in_val = [p for p in _ppaths if val_mask[_pos[_base(p)]]]
        _frac = len(_in_val) / max(1, len(_ppaths))
        if _in_val:
            print(f"[pseudo] ⚠️ 清单 {len(_ppaths)} 条中有 {len(_in_val)} 条落在本机 val"
                  f"（{_frac:.1%}）—— 生成机与训练机的 val 切分随 manifest 行序漂移，"
                  f"属预期；这 {len(_in_val)} 条**跳过不回收**（val 绝不被触碰）。"
                  f"例：{[_base(x) for x in _in_val[:3]]}")
            if _frac > 0.2:
                raise SystemExit(f"[pseudo] 🛑 val 重叠 {_frac:.1%} > 20%，"
                                 f"疑似清单与训练集口径漂移，中止人工核查")
        _pairs = [(p, pl) for p, pl in zip(_ppaths, _plabels)
                  if not val_mask[_pos[_base(p)]]]
        _changed = 0
        for p, pl in _pairs:
            i = _pos[_base(p)]
            if labels[i] != int(pl):
                labels[i] = int(pl)
                _changed += 1
        tr_labels = labels[~val_mask]                   # 布尔索引是副本，必须重算
        print(f"[pseudo] ★ 回收 {len(_pairs)} 条（清单 {len(_ppaths)} 条，"
              f"跳过 val {len(_in_val)} 条；实际改标签 {_changed} 条，"
              f"其余伪标签与原标签一致）；val 切分未受影响")

    # ------------------------------------------------ 预处理缓存（可选）
    # ★ 只在**训练**里用，推理/出包一律走原图：缓存改的是训练分布，
    #   它的唯一用途是把「每张 23 ms 的 CPU 解码+变换」压到 7 ms 左右，
    #   好把扫参的轮速提上来。绝对分数不可与旧结果横比，见 build_preproc_cache.py。
    if args.preproc:
        from data import find_train_root
        p_root = C.DATA_DIR / "_preproc" / "train"
        tr_root = find_train_root()
        missing = 0

        def _mp(ps):
            nonlocal missing
            out = []
            for p in ps:
                try:
                    q = p_root / Path(p).relative_to(tr_root)
                except ValueError:
                    q = None
                if q is not None and q.exists():
                    out.append(str(q))
                else:
                    out.append(p)
                    missing += 1
            return out

        tr_paths, va_paths = _mp(tr_paths), _mp(va_paths)
        print(f"[preproc] ★ 启用预处理缓存 {p_root}"
              f"（缺失回落 {missing} 张）—— 绝对分数不可与未启用时横比")
        if missing:
            print(f"[preproc] ⚠️ 缺失比例 {missing/(len(tr_paths)+len(va_paths)):.1%}，"
                  f"回落部分会明显拖慢轮速")

        # ---------------------------------------------------------------
        # ★★★ 缓存指纹自证（2026-10-04 新增）
        #
        # 为什么必须加这一段（两个真实教训）
        # ---------------------------------------------------------------
        # ① `scripts/cloud_run384.sh:5-12` 记录的坑：v1 把 `data/train` 软链切到
        #    `_preproc384`，但 `_mp()` 把缓存根**写死**成 `C.DATA_DIR/_preproc/train`
        #    ⇒ 软链**改不到它**，训练继续读 320 缓存，而 transform 按 384 走
        #    ⇒ **等价于"纯上采样"，不报任何错**。当时的判据只有"配置路径写对了"，
        #      而配置路径**恰恰是骗人的那一项**。
        # ② 外部复核指出：只看目录配置不够，要核对**实际读到的像素**。
        #
        # 判据设计（**真去读 PIL 像素**，不看配置）：
        #   · 抽 12 张**映射后**的训练图，打印各自短边
        #   · 统计 `短边落点`：若绝大多数 == 320 而配置声称是 384 ⇒ **磁盘上其实是 320 缓存**
        #   · 同时报"回落原图"的抽样短边（原图短边中位 480，与缓存显著不同）
        #   · 与 `p_root` 名字里的数字**交叉验证**（名字说 384、像素说 320 ⇒ 立即报警）
        # ★ 全部只抽样 12 张、只读 header（PIL 的 `.size` 不解码像素），开销可忽略。
        # ---------------------------------------------------------------
        try:
            import random as _rnd

            from PIL import Image as _I

            _rng = _rnd.Random(7)
            _idx = list(range(len(tr_paths)))
            _rng.shuffle(_idx)
            _picks, _shorts = [], []
            for _i in _idx[:240]:                    # 最多探 240 张，凑够 12 张有效即止
                _p = tr_paths[_i]
                try:
                    with _I.open(_p) as _im:
                        _w, _h = _im.size
                    _picks.append((_p, min(_w, _h), max(_w, _h)))
                    _shorts.append(min(_w, _h))
                except Exception:
                    continue
                if len(_picks) >= 12:
                    break

            if _shorts:
                import statistics as _st

                _mn, _md, _mx = min(_shorts), int(_st.median(_shorts)), max(_shorts)
                # 从配置路径里抠出声称的分辨率（`_preproc384` → 384；`_preproc` → 320）
                _claimed = 320
                _nm = p_root.name
                for _c in ("128", "160", "192", "224", "256", "288", "320", "384",
                           "448", "512", "640"):
                    if _c in _nm:
                        _claimed = int(_c)
                        break

                print(f"[preproc] 🔎 指纹自证：抽 {len(_picks)} 张**映射后**的图为 "
                      f"短边 min={_mn} med={_md} max={_mx}（配置路径声称 {_claimed}）")
                print(f"[preproc] 🔎 样例："
                      + " | ".join(f"{Path(p).name[:12]}…{min(w, h)}" for p, w, h in _picks[:4]))
                # 断言：若"声称 >320"但"实测中位 ==320" ⇒ 磁盘上其实是 320 缓存
                if _claimed > 320 and _md == 320 and _mx <= 320:
                    raise SystemExit(
                        f"[preproc] 🛑 **指纹自证失败**：配置声称 {_claimed}，但实测短边 "
                        f"med={_md} max={_mx} ⇒ 磁盘上实际是 **320 缓存**（软链/路径写死的经典坑）"
                        f"。若这是无意的，训练会静默降级为『纯上采样』；"
                        f"若无意的就请检查 AIC_DATA_DIR 与 data384/ 工作区。")
                if _mx > _claimed:
                    print(f"[preproc] ⚠️ 实测 max={_mx} > 声称 {_claimed}"
                          f"（预处理是『只缩不放』，出现这种情况通常意味着缓存里混入了未缩放的图）")
                print(f"[preproc] ✅ 指纹自证通过：实测短边与路径声称一致（{_claimed}）")
        except SystemExit:
            raise
        except Exception as _e:                     # 自证本身不能拖垮训练
            print(f"[preproc] ⚠️ 指纹自证跳过（{type(_e).__name__}: {_e}）")

    print(f"[data] 训练 {len(tr_paths)} / 验证 {len(va_paths)}"
          f"（验证每类至少 {per_class_min} 张）")
    # ★ 护栏：训练集为空时，往下走会得到一句完全看不出原因的报错
    #   （`ValueError: num_samples should be a positive integer value, but got num_samples=0`
    #    来自 DataLoader 的 RandomSampler），极难从报错反推是"冒烟抽样太小"。
    #   实测踩到：`--limit 800`（= 每类 1 张）配默认 `--val-per-class 8`
    #   ⇒ 每类那 1 张**全被划给验证集** ⇒ 训练集 0 张。
    #   ⇒ 在这里直接拦下并给出可执行的修法。
    if len(tr_paths) == 0:
        per = max(1, (args.limit or len(paths)) // max(K, 1))
        raise SystemExit(
            f"[data] ✗ 训练集为空。原因：冒烟抽样每类只有 {per} 张，"
            f"而 `--val-per-class {per_class_min}` 要求每类至少留 {per_class_min} 张验证，"
            f"于是每类那几张全被划走。\n"
            f"        修法（任选其一）：调大 `--limit`（需 > "
            f"{per_class_min * K} 才有余量），或调小 `--val-per-class`"
            f"（例如 `--limit 6000 --val-per-class 2`）。")
    if len(va_paths) < K:
        print(f"[data] ⚠️ 验证集（{len(va_paths)}）少于类别数（{K}），"
              f"val_acc 参考价值有限，请以排行榜为准")

    # ------------------------------------------------------------ 模型
    kind, clip_model, _proc = load_clip(device)
    clip_model = clip_model.float()   # 训练用 fp32 权重 + AMP 混合精度

    if args.lora_scale > 0:
        stats = LO.inject_lora(clip_model, r=args.r, alpha=args.alpha,
                               dropout=args.dropout,
                               targets=tuple(args.targets.split(",")),
                               last_n_layers=int(getattr(args, "lora_last_n", 0) or 0))
    else:
        for p in clip_model.parameters():
            p.requires_grad = False
        stats = None
        print("[lora] lora-scale=0，退化为纯冻结骨干基线")

    # ------------------------------------------------------- ★ VPT 注入
    # ★ 必须在 inject_lora **之后**：inject_lora 会把全模型参数置为
    #   requires_grad=False，若 VPT 在那之前注入就会被**静默冻住**
    #   （参数在、梯度为零、训练毫无变化，loss 照降）。inject_vpt 内部还会
    #   再显式置 True 一次，双保险。
    # ★ VPT 与 LoRA 正交：改的是**输入序列**而非权重，可单独开、可叠加。
    #   --lora-scale 0 + --vpt N ⇒ VPT-only 臂（纯冻结骨干 + prompt）。
    vpt_info = {"n_prompt": 0, "params": 0, "injected": False}
    if int(args.vpt) > 0:
        if args.lora_scale <= 0:
            print("[vpt] 本次为 VPT-only 臂（lora-scale=0，骨干全冻结）")
        vpt_info = VPT.inject_vpt(clip_model, int(args.vpt),
                                  init_std=float(args.vpt_std))
        # 点火前自检
        _vrep = VPT.report(clip_model, C.IMG_SIZE, batch=2, device=device)
        if not _vrep.get("ok"):
            raise SystemExit(
                "[vpt] 🛑 注入自检未通过，拒绝开跑（详见上方 vpt-report）。"
                "VPT 的失败态几乎都是静默的，跑下去只会白烧机时。")
    elif int(args.vpt) == 0:
        print("[vpt] 未启用（--vpt 0）")

    # ★★ 位置很重要（2026-10-07 修正）：**必须在 `unfreeze_last` 处理之后**。
    #   unfreeze 要访问 `vision_model.encoder.layers`，而挂适配器会把
    #   `vision_model` 换成包装类 `_RegionVisionModel`（它没有 `encoder` 属性）
    #   ⇒ 挂早了会让 unfreeze 报「找不到 encoder.layers」并 SystemExit。
    #   ⇒ 那会导致本臂**丢掉「解冻最后 4 层」这个 U 冠军关键变量** ⇒ 不是干净对照。
    # ★★ 区域自注意力适配器（2026-10-07 引入，实测先验 +0.30pp）
    #   动机：这是**与 LoRA 正交**的一个维度——LoRA 改"单点权重"，
    #   它改"patch token 的区域级信息聚合"。我们已投的所有轴都没碰它。
    #   ★ 必须在 inject_lora **之后**：它要接管 clip_model.vision_model.forward，
    #   而 inject_lora 会替换视觉塔里的 Linear 子模块。
    region_adapter = None
    if int(getattr(args, "region_attn", 0)) > 0:
        from region_attention import attach_region_adapter
        region_adapter = attach_region_adapter(
            clip_model, heads=args.region_heads, pool=args.region_pool)
        n_reg = sum(p.numel() for p in region_adapter.parameters())
        print(f"[region] 已挂区域自注意力适配器 heads={args.region_heads} "
              f"pool={args.region_pool} 参数量={n_reg / 1e6:.2f}M "
              f"gate_init={args.region_gate}")

    # ★ 区域适配器参数必须显式可训练（它在 clip_model 内部，
    #   而 clip_model 的其余部分已被 inject_lora 全冻结）
    if region_adapter is not None:
        for p in region_adapter.parameters():
            p.requires_grad = True
        print(f"[region] 适配器参数已置为可训练（{sum(1 for _ in region_adapter.parameters())} 个张量）")
    model = LoRAClipClassifier(clip_model, kind, K, hidden=args.hidden).to(device)
    # 头与 logit_scale 始终可训练
    for p in model.head.parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad = True

    # ---------------------------------------------------- ★ 骨干解冻
    # ★ 放在 inject_lora **之后**：LoRALinear 已把原 Linear 换成 `base` 子模块，
    #   这里解冻的是 `L.parameters()` 的**全部**参数（含 base / LayerNorm /
    #   两个 LoRA 旁路）。旁路参数随后会在分组时被 `_dedup` 剔掉
    #   （`lora_ps` 先入 `seen`），所以两组的参数不会重叠 —— 一旦重叠
    #   AdamW 会直接抛 "some parameters appear in more than one parameter group"。
    bb_ps: list = []
    # ★ 2026-10-09 LLRD 备用：按层保留参数列表（顺序＝picked 即"早层→晚层"）。
    #   仅当 --llrd-decay > 0 时用于建"逐层 lr"分组；默认（0）路径完全不受影响。
    _bb_layer_ps: list = []
    if args.unfreeze_all or args.unfreeze_last > 0:
        vt = getattr(clip_model, "vision_model", None) or getattr(clip_model, "visual", None)
        if vt is None:
            raise SystemExit("[unfreeze] 🛑 找不到视觉塔（vision_model / visual）")
        layers = getattr(getattr(vt, "encoder", None), "layers", None)
        if layers is None:
            raise SystemExit("[unfreeze] 🛑 视觉塔里找不到 encoder.layers")
        n_l = len(layers)
        n_un = n_l if args.unfreeze_all else min(args.unfreeze_last, n_l)
        picked = list(layers)[n_l - n_un:]
        for L in picked:
            for p in L.parameters():
                p.requires_grad = True
        # 最后一层后面紧跟的 post_layernorm 也一起解冻，否则解冻层的输出
        # 会被一个冻结的归一化层压回去（pre_layernorm 同理，在塔入口）。
        for nm in ("post_layernorm", "pre_layernorm"):
            m = getattr(vt, nm, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
        bb_ps = [p for L in picked for p in L.parameters()]
        for nm in ("post_layernorm", "pre_layernorm"):
            m = getattr(vt, nm, None)
            if m is not None:
                bb_ps += list(m.parameters())
        _bb_layer_ps = [[p for p in L.parameters()] for L in picked]   # LLRD 用（早→晚）
        print(f"[unfreeze] ★ 解冻视觉塔最后 {n_un}/{n_l} 层"
              f"（骨干 lr = {args.lr}×{args.backbone_lr_mult}"
              f" = {args.lr * args.backbone_lr_mult:.2e}）")
        print(f"[unfreeze]   ⚠️ 这是本项目第一次让骨干参数参与训练；"
              f"若 val 曲线先涨后掉即为过拟合，选优请用 --select-by lowsim_a")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] 可训练 {n_train:,} / {n_total:,}（{n_train/n_total*100:.3f}%）")

    # ------------------------------------------------- 噪声转移矩阵 T
    T_t = None
    if args.use_cl or args.kind.startswith("forward"):
        tp = C.CACHE_DIR / "T.npy"
        if tp.exists():
            T = NZ.temper_transition_matrix(np.load(tp), args.t_strength)
            T_t = torch.as_tensor(T, dtype=torch.float32, device=device)
            print(f"[noise] 载入 T，收缩强度 {args.t_strength}")
        else:
            print(f"[noise] ⚠️ 未找到 {tp.name}，CL 降权与前向校正将跳过"
                  f"（只影响权重，不影响能否训练）")

    # 样本权重：CL 保真度降权 x 类平衡
    w_np = NZ.combine_weights(labels, None if T_t is None else T_t.cpu().numpy(), K,
                              use_cl=args.use_cl, cl_lo=C.CLIP_LO, cl_hi=C.CLIP_HI,
                              class_balance=C.USE_CLASS_BALANCE, cb_beta=C.CB_BETA)
    print(f"[noise] 样本权重：min={w_np.min():.3f} max={w_np.max():.3f} "
          f"mean={w_np.mean():.3f}")
    # 切出训练子集那一份权重（验证集不套权重，评估要的是真实分布）
    w_tr = w_np[~val_mask].astype(np.float32)
    # 训练循环里按索引取权重，放 GPU 上省一次 H2D 调用
    w_all_gpu = torch.as_tensor(w_tr, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------------
    # ★★ 2026-10-06 软目标 + 冲突组降权（**默认关闭 ⇒ 上面各行逐字不变**）
    #   规格（用户 2026-10-06 定）：只依据**文件 md5** 处理冲突组；
    #   **每个冲突组总权重设为 1**（组内每张 1/m）；软目标在组内**不同标签间均分**。
    #   ★ 缺省 `args.soft_conflicts is None` ⇒ `conf_w` 为全 1、`soft_lut` 为 None，
    #     训练循环仍走原来的 `NZ.compute_loss` ⇒ **75.0641 的复现路径不受影响**。
    conf_w = None
    soft_lut = None          # (n_train,) 的 object 数组：每项 (classes, probs) 或 None
    if getattr(args, "soft_conflicts", None):
        import json as _json
        _base = lambda p: str(p).replace("\\", "/").rsplit("/", 1)[-1]
        _rec = _json.load(open(args.soft_conflicts, encoding="utf-8"))
        # 训练子集索引 → 软目标查表（val 不参与，故按 ~val_mask 取）
        _tr_paths = [p for p, m in zip(paths, val_mask) if not m]
        _m = {}
        for _g in _rec["groups"]:
            for _mem in _g["members"]:
                _m[_base(_mem["path"])] = _g
        conf_w = np.ones(len(_tr_paths), dtype=np.float32)
        soft_lut = np.empty(len(_tr_paths), dtype=object)
        _hit = 0
        for _i, _p in enumerate(_tr_paths):
            _g = _m.get(_base(_p))
            if _g is None:
                soft_lut[_i] = None
                continue
            _hit += 1
            conf_w[_i] = float(_g["weight_each"])
            _cls = np.asarray(_g["labels"], dtype=np.int64)
            _pr = np.asarray([_g["soft_target"][str(int(c))] for c in _cls],
                             dtype=np.float32)
            soft_lut[_i] = (_cls, _pr)
        # ★★ 2026-10-07 修正：原文字**硬编码"组总权重=1"**，但清单可以是变体
        #   （如 w05 清单的 weight_each=0.25、组总权重实为 0.5）⇒ **日志误导审计**。
        #   改成从清单**实算**，避免日志与事实不一致。
        _tots = [_g["weight_each"] * len(_g["members"]) for _g in _rec["groups"]]
        _gt_lo, _gt_hi = min(_tots), max(_tots)
        _gt = f"{_gt_lo:.4f}" if abs(_gt_hi - _gt_lo) < 1e-3 else f"{_gt_lo:.4f}~{_gt_hi:.4f}"
        print(f"[soft] 冲突组={_rec['n_groups']} 训练集命中={_hit} "
              f"★ 组总权重={_gt}（组内每张 {_rec['groups'][0]['weight_each']:.4f}…）")
        print(f"[soft] 软目标：组内不同标签均分；未命中样本仍为 one-hot 硬标签")
        w_tr = (w_tr * conf_w).astype(np.float32)
        w_all_gpu = torch.as_tensor(w_tr, dtype=torch.float32, device=device)
        print(f"[soft] 权重叠加后：min={w_tr.min():.4f} max={w_tr.max():.4f} "
              f"mean={w_tr.mean():.4f} sum={w_tr.sum():.1f}/{len(w_tr)}")

    # ------------------------------------------------------------------------
    # ★★★ 2026-10-10 教师蒸馏（KD）软目标（**默认关闭 ⇒ 上面各行逐字不变**）
    #   教师 = 本队自训的最强权重（如 d_sam_swa_f）；先对**训练集**打软标签落盘，
    #   本处按 **basename** 对齐（与 --pseudo / --soft-conflicts 同规则），
    #   混成 target = (1-λ)·one_hot(y) + λ·p_teacher，取前 topk 稀疏写进同一套 soft_lut
    #   ⇒ **零循环手术**（训练循环与损失函数一行不动）。
    #   ★ 冲突组优先：命中 md5 冲突组的样本保留冲突组软目标（不动冠军机制）。
    #   ★ 合规：教师是本队自训权重；软标签只来自**训练集**；不碰测试集。
    if getattr(args, "kd_probs", None):
        import json as _kjson, os as _kos
        _kdir = args.kd_probs
        _knpy = _kdir if str(_kdir).endswith(".npy") else _kos.path.join(_kdir, "probs_train.npy")
        # 索引文件：优先 paths_train.json；否则回退 pseudo_label.py 的 pseudo_index.json（取 "paths" 键）
        _kjson_p = (_kos.path.dirname(_knpy) or ".") + "/paths_train.json"
        if not _kos.path.exists(_kjson_p):
            _kjson_p = (_kos.path.dirname(_knpy) or ".") + "/pseudo_index.json"
        _lam = float(getattr(args, "kd_lambda", 0.3))
        _topk = int(getattr(args, "kd_topk", 32))
        if not (0.0 < _lam < 1.0):
            raise SystemExit(f"[kd] 🛑 --kd-lambda 必须落在开区间 (0,1)，收到 {_lam}")
        _P = np.load(_knpy)
        _paths_kd = _kjson.load(open(_kjson_p, encoding="utf-8"))
        if isinstance(_paths_kd, dict):            # pseudo_index.json ⇒ 取 "paths"
            _paths_kd = _paths_kd["paths"]
        if _P.shape[0] != len(_paths_kd):
            raise SystemExit(f"[kd] 🛑 probs 行数 {_P.shape[0]} ≠ paths 数 {len(_paths_kd)}")
        _basef = lambda p: str(p).replace("\\", "/").rsplit("/", 1)[-1]
        _row_of = {_basef(p): i for i, p in enumerate(_paths_kd)}
        _tr_paths_kd = [p for p, m in zip(paths, val_mask) if not m]
        if soft_lut is None:
            soft_lut = np.empty(len(_tr_paths_kd), dtype=object)
        _khit = _kconf = _kmiss = 0
        for _i, _p in enumerate(_tr_paths_kd):
            _r = _row_of.get(_basef(_p))
            if _r is None:
                _kmiss += 1
                continue
            _khit += 1
            if soft_lut[_i] is not None:          # ★ 冲突组优先
                _kconf += 1
                continue
            _mix = _P[_r].astype(np.float32) * _lam
            _mix[int(tr_labels[_i])] += (1.0 - _lam)
            _sel = np.argpartition(-_mix, _topk)[:_topk]
            _pr = _mix[_sel]
            _s = float(_pr.sum())
            if _s > 0:
                _pr = (_pr / _s).astype(np.float32)
            soft_lut[_i] = (_sel.astype(np.int64), _pr)
        print(f"[kd] ★ 教师蒸馏已启用：λ={_lam} topk={_topk}｜训练集命中={_khit}"
              f"（冲突组优先保留 {_kconf}）｜未命中 {_kmiss}｜probs={_knpy}")
        print(f"[kd]   ★ 合规：教师＝本队自训权重；软标签只来自训练集（不碰测试集）")

    # ------------------------------------------------------------------------
    # ★★★ 2026-10-10 sim_max 带重加权（**默认关闭 ⇒ 上面各行逐字不变**）
    #   依据：落差 3.15pp 的 **89% 由 sim_max 解释**；测试集在训练池 sim_max 的
    #         **p5~p30 带过度代表 2.0~2.3×** ⇒ 训练强调与该带不匹配。
    #   做法：用**冻结 CLIP 特征**（`cache/feat_train.npy`，与 manifest 行序对齐、
    #         即 --track-lowsim 的同一份）算每个训练样本的 **sim_max ＝ 到训练子池的
    #         最大余弦（排除自身）** ⇒ 越"不典型"权重越高：
    #             w_i = clip( ((1-s_i)/mean(1-s))^alpha , lo, hi )
    #   ★ 合规：权重只由**训练集自身**统计量导出（label-free），**不使用任何测试集信息**
    #     （测试侧分布仅作立项动机，不作训练输入）。
    #   ★ 自证行 `[smrw] ★ sim_max 重加权已启用` 供链条闸门 grep；缺行即中止。
    if getattr(args, "simmax_rew", None):
        _spec = str(args.simmax_rew)
        _kw = {}
        for _seg in _spec.split(","):
            if "=" in _seg:
                _k, _v = _seg.split("=", 1)
                _kw[_k.strip()] = float(_v)
        _alpha = float(_kw.get("alpha", 1.0))
        _lo = float(_kw.get("lo", 0.5))
        _hi = float(_kw.get("hi", 2.0))
        if not (_lo > 0 and _hi >= _lo):
            raise SystemExit(f"[smrw] 🛑 权重上下界不合法：lo={_lo} hi={_hi}")
        _Xp = C.CACHE_DIR / "feat_train.npy"
        if not _Xp.exists():
            raise SystemExit(f"[smrw] 🛑 找不到特征矩阵 {_Xp}（无法算 sim_max）")
        _X = np.load(_Xp)
        _tr_idx = np.where(~val_mask)[0]                 # 训练子集在 manifest 里的行号
        _Xt = _X[_tr_idx].astype(np.float32)
        _Xt /= (np.linalg.norm(_Xt, axis=1, keepdims=True) + 1e-8)   # 行归一 ⇒ 余弦
        _n = _Xt.shape[0]
        _smax = np.empty(_n, dtype=np.float32)
        _CH = 4096
        for _s in range(0, _n, _CH):
            _e = min(_s + _CH, _n)
            _blk = _Xt[_s:_e] @ _Xt.T                    # (chunk, n) 余弦
            _rows = np.arange(_e - _s)
            _blk[_rows, np.arange(_s, _e)] = -2.0        # ★ 排除自身
            _smax[_s:_e] = _blk.max(axis=1)
        _atyp = np.clip(1.0 - _smax, 1e-6, None)         # 越不典型越大
        _rw = np.power(_atyp / _atyp.mean(), _alpha)
        _rw = np.clip(_rw, _lo, _hi).astype(np.float32)
        _q = lambda a: " / ".join(f"{v:.3f}" for v in np.quantile(a, [0.25, 0.5, 0.75]))
        print(f"[smrw] ★ sim_max 重加权已启用：spec={_spec}｜训练样本 {_n:,}"
              f"｜sim_max 分位(25/50/75)={_q(_smax)}｜权重分位={_q(_rw)}"
              f"｜升权比例 {( _rw > 1.0).mean():.1%}｜加权前后 mean={_rw.mean():.3f}")
        print(f"[smrw]   ★ 合规：权重仅由训练集自身特征导出（label-free，不用任何测试信息）")
        w_tr = (w_tr * _rw).astype(np.float32)
        w_all_gpu = torch.as_tensor(w_tr, dtype=torch.float32, device=device)
        print(f"[smrw] 权重叠加后：min={w_tr.min():.4f} max={w_tr.max():.4f} "
              f"mean={w_tr.mean():.4f} sum={w_tr.sum():.1f}/{len(w_tr)}")

    # ------------------------------------------------ 噪声筛选状态（课程式）
    # 索引天然对齐：`w_tr` 就是 `w_np[~val_mask]`，而 Dataset 的 `__getitem__`
    # 返回的正是**这个训练子集内**的下标 ⇒ 不需要任何重映射表。
    # （这也是这里**不去重建 DataLoader** 的原因：一旦按 keep 子集重建，下标语义
    #   就会变，得额外维护映射，且被剔除的样本再也拿不到 loss、无法回头。）
    sel_on = args.sel_frac > 0
    sel_sum = sel_cnt = sel_flag = None
    if sel_on:
        if not (0.0 < args.sel_frac < 1.0):
            raise SystemExit(f"[sel] 🛑 --sel-frac 必须落在开区间 (0,1)，"
                             f"收到 {args.sel_frac}")
        if args.sel_start > args.epochs:
            raise SystemExit(f"[sel] 🛑 --sel-start={args.sel_start} 大于 "
                             f"--epochs={args.epochs} ⇒ 永远筛不到，等于白开")
        _n_tr = int(len(w_tr))
        sel_sum = np.zeros(_n_tr, dtype=np.float64)    # 累积逐样本损失（**未加权**）
        sel_cnt = np.zeros(_n_tr, dtype=np.int64)
        sel_flag = np.zeros(_n_tr, dtype=bool)         # 已被判为疑似噪声
        _nc = np.bincount(tr_labels, minlength=K)
        print(f"[sel] ★ 噪声样本筛选已启用：sel_frac={args.sel_frac} "
              f"sel_start={args.sel_start} min_keep={args.sel_min_keep}")
        print(f"[sel]   类内样本数 min={int(_nc.min())} 中位={int(np.median(_nc))} "
              f"max={int(_nc.max())}；按**类内分位**筛，不跨类比较 loss")
        _floor = int((_nc <= args.sel_min_keep).sum())
        if _floor:
            print(f"[sel]   ⚠️ {_floor} 个类的样本数 ≤ {args.sel_min_keep}，"
                  f"这些类**一张都不会被筛**（硬地板生效）")

    # ---------------------------------------------------------- 加载器
    ds_tr = LabeledImageDataset(tr_paths, tr_labels, train=True, weights=w_tr,
                                degrade=args.degrade, aug=args.aug,
                                size_rand=args.size_rand)
    if args.aug == "strong":
        print(f"[train] ★ 强增强已启用（--aug strong）：高斯模糊 p=0.30 + "
              f"ColorJitter(0.3,0.3,0.2,0.02) + 灰度 p=0.02。"
              f"**几何未动**（scale 仍 0.7~1.0），TTA 的 rc 视图语义不受影响。")
    if args.degrade:
        spec = ds_tr.tf.transforms[0]
        print(f"[train] ★ 画质退化增强已启用：{args.degrade}"
              f"（p={spec.p}, max_side={spec.max_side}, "
              f"q∈[{spec.q_lo},{spec.q_hi}]）—— 用于对齐官方测试集的"
              f"「压到长边 500 + 重压缩」画质；实测代价 −0.0196")
    # ★ H2 自证：按**类型**取，不写死下标（`--degrade` 开关会移动下标）
    if args.size_rand:
        from torchvision import transforms as _T

        _chain = list(ds_tr.tf.transforms)
        _sr = [t for t in _chain if isinstance(t, _SizeRandomize)]
        assert len(_sr) == 1, f"[size-rand] 期望恰好 1 个 _SizeRandomize，实得 {len(_sr)}"
        _sr = _sr[0]
        _i_sr = next(i for i, t in enumerate(_chain) if isinstance(t, _SizeRandomize))
        _i_rrc = next(i for i, t in enumerate(_chain)
                      if isinstance(t, _T.RandomResizedCrop))
        assert _i_sr < _i_rrc, (
            f'[size-rand] 顺序错误：size_rand@{_i_sr} 必须在 RRC@{_i_rrc} 之前'
            f'（RRC 的 scale 是对「刚传进来的图」取面积的，顺序错了语义就变了）')
        print(f"[train] ★★ 随机尺寸归一化已启用（H2）：{args.size_rand}"
              f"（p={_sr.p}, levels={list(_sr.levels)}, 只缩不放）——"
              f"对齐官方测试集的「长边恒 500」"
              f"（实测：测试集长边==500 占 99.4%、>500 占 0.0%）")
        print(f"[train]   偏序自证：{' -> '.join(type(t).__name__ for t in _chain)}")
        print(f"[train]   ⇒ size_rand@{_i_sr} < RRC@{_i_rrc} ✅；"
              f"RRC.scale={_chain[_i_rrc].scale}（**必须仍是 (0.7, 1.0)**）")
    ds_va = LabeledImageDataset(va_paths, va_labels, train=False)
    # 验证集要用确定性预处理，直接换掉 transform
    ds_va.tf = eval_transform(va_paths)

    nw = args.workers
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=nw, drop_last=True,
                       pin_memory=False,
                       prefetch_factor=4 if nw > 0 else None)
    dl_va = DataLoader(ds_va, batch_size=args.eval_batch, shuffle=False,
                       num_workers=nw, pin_memory=False,
                       prefetch_factor=2 if nw > 0 else None)

    # ---------------------------------------------------------- 优化器
    # ★ 参数分组时**必须去重**。
    # 踩过的坑：先拼一个 params 总表、再按模块分别建组，同一参数会同时出现在
    # 总表与某个分组里，AdamW 直接抛
    #   ValueError: some parameters appear in more than one parameter group
    # 正确做法：每类参数只归属一个组，且组间用 id() 判重。
    def _dedup(ps, seen: set) -> list:
        out = []
        for p in ps:
            if id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
        return out

    seen: set[int] = set()
    lora_ps = _dedup(LO.lora_parameters(model), seen)
    # ★★★ 2026-10-07：**区域适配器必须在 bb_ps 之前去重**（顺序很关键！）
    #   适配器的参数**也是 `clip_model.parameters()` 的一部分**，
    #   若 bb_ps 先跑 `_dedup`，它会把适配器的 4.80M 全部"吃掉" ⇒ `reg_ps` 变空
    #   ⇒ 适配器跟着骨干 lr（`lr×0.1` = 2e-05，比 LoRA 小 10 倍）走，
    #     而且日志显示 `region=0`，看不出异常。
    #   实测（本地复刻）：bb_ps 先跑时 `region=0`，总数照样对得上 ⇒ **闸门拦不住**。
    reg_ps = _dedup(list(region_adapter.parameters())
                    if region_adapter is not None else [], seen)
    # ★ 骨干参数**必须在 lora_ps 与 reg_ps 之后**去重：解冻层的 `L.parameters()`
    #   里也包含 LoRALinear 的 lora_A / lora_B，先入 seen 就能把它们剔出去，
    #   保证每个参数只出现在一个组里。
    bb_ps = _dedup([p for p in bb_ps if p.requires_grad], seen)
    # ★ VPT 参数也走 _dedup：虽然按名（vpt_prompt / lora_）判定不会重叠，
    #   但一旦将来有人改了命名规则，只有这一道去重能拦住 AdamW 直接抛
    #   "some parameters appear in more than one parameter group"。
    vpt_ps = _dedup(VPT.vpt_parameters(model), seen)
    head_ps = _dedup(model.head.parameters(), seen)
    scale_ps = _dedup([model.logit_scale], seen)
    params = lora_ps + vpt_ps + reg_ps + bb_ps + head_ps + scale_ps

    nan_lora = sum(p.numel() for p in lora_ps)
    nan_vpt = sum(p.numel() for p in vpt_ps)
    nan_bb = sum(p.numel() for p in bb_ps)
    nan_reg = sum(p.numel() for p in reg_ps)
    nan_head = sum(p.numel() for p in head_ps)
    print(f"[opt] 分组参数：lora={nan_lora:,} vpt={nan_vpt:,} backbone={nan_bb:,} "
          f"region={nan_reg:,} head={nan_head:,} "
          f"logit_scale={sum(p.numel() for p in scale_ps)}")
    # ★ 账目核对（闸门）：分组总数必须等于"可训练参数总数"，
    #   不等就说明有参数既没进组（永远不更新）或被重复计入。
    _n_grouped = sum(p.numel() for p in params)
    if _n_grouped != n_train:
        raise SystemExit(
            f"[opt] 🛑 分组参数总数 {_n_grouped:,} ≠ 可训练参数总数 {n_train:,}"
            f"（差 {n_train - _n_grouped:,}）。拒绝在错配状态下训练 —— "
            f"这类错配不报错、只是有参数静默不更新或用了错的学习率。")

    # ------------------------------------------------------- EMA 影子权重
    # ★ 只跟踪**可训练参数**（LoRA + 头 + logit_scale ≈ 2.65M ⇒ fp32 约 10 MB），
    #   不复制整个模型：冻结的骨干在训练中一字不变，复制它们纯属浪费显存。
    # ★ 用 `named_parameters()` 收集、再按 **id** 与优化器参数列表对齐，
    #   而不是靠顺序或名字匹配：`params` 是**去重后**的拼接列表
    #   （lora_ps + bb_ps + head_ps + scale_ps），它和 named_parameters 的
    #   顺序没有任何保证。错配的后果是 EMA 把 A 的权重灌进 B，**不报错**、
    #   只是分数莫名变差 —— 属于本工程最讨厌的那类静默错误。
    _train_ids = {id(p) for p in params}
    ema_pairs: list = []
    if args.ema_decay > 0:
        if not (0.0 < args.ema_decay < 1.0):
            raise SystemExit(f"[ema] 🛑 --ema-decay 必须落在开区间 (0,1)，"
                             f"收到 {args.ema_decay}（1.0 会让影子变成永不更新的"
                             f"初始权重）")
        for _n, _p in model.named_parameters():
            if id(_p) in _train_ids:
                ema_pairs.append((_p, _p.detach().clone().float()))
        if len(ema_pairs) != len(_train_ids):
            raise SystemExit(
                f"[ema] 🛑 EMA 跟踪到 {len(ema_pairs)} 个张量，而优化器在更新 "
                f"{len(_train_ids)} 个 —— 两者必须一致。拒绝在错配状态下训练。")
        _mb = sum(t.numel() for _, t in ema_pairs) * 4 / 2 ** 20
        print(f"[ema] ★ 启用 decay={args.ema_decay}，从第 {args.ema_start} 轮起累积，"
              f"跟踪 {len(ema_pairs)} 个张量（{_mb:.0f} MB fp32）")

    # 分组学习率：LoRA 用小 lr（预训练权重不宜大改），头用大 lr（从零开始学）
    #
    # ⚠️ 这个倍数不能想当然。实测教训：合成探针里 512 张图 / 16 类，
    # args.lr=3e-4 + 倍数 10 ⇒ 头 lr = 3e-3，AdamW 在几十步内就把头推到
    # 「所有样本预测同一两个类」的退化解（提交文件里只剩 2 个唯一标签），
    # 准确率掉到 8%（随机水平）。当时差点把这当成"LoRA 不行"的证据 ——
    # 实际是学习率把模型练塌了，跟容量假设毫无关系。
    # 现在倍数是显式参数，且默认值从 10 降到 5（仍能让头快速起步）。
    head_mult = args.head_lr_mult
    groups = []
    if scale_ps:
        groups.append({"params": scale_ps, "lr": args.lr * head_mult,
                       "name": "logit_scale"})
    if head_ps:
        groups.append({"params": head_ps, "lr": args.lr * head_mult, "name": "head"})
    if lora_ps:
        groups.append({"params": lora_ps, "lr": args.lr, "name": "lora"})
    if vpt_ps:
        groups.append({"params": vpt_ps, "lr": args.lr * args.vpt_lr_mult,
                       "name": "vpt"})
        print(f"[opt] VPT prompt 学习率 = {args.lr}×{args.vpt_lr_mult}"
              f" = {args.lr * args.vpt_lr_mult:.2e}（从零训练，非预训练权重）")
    if bb_ps:
        # ★★★ 2026-10-09 LLRD（逐层学习率衰减）—— 可选，默认关闭（零影响）
        #   配方（BEiT/MAE 系标准）：lr_k = base × decay^(n-1-k)，k=0 为最早解冻层
        #   ⇒ 顶层（离头最近）保持 base，越往底层越衰减。
        #   动机：直接服务官方五.3「最大程度保留和利用 CLIP 预训练先验知识」；
        #   与"平铺单档"的**唯一变量 = lr 在各层的分布**（顶层不变、底层降）。
        #   ⚠️ 已知口径：平均 lr 会低于平铺臂（顶层不变、其余降）——
        #      若日后要"同预算只改分布"，用 --backbone-lr-mult 抬整体再衰减。
        _llrd = float(getattr(args, "llrd_decay", 0.0) or 0.0)
        if _llrd > 0.0 and _bb_layer_ps:
            _keep = {id(p) for p in bb_ps}          # 已去重后的骨干参数集
            _n = len(_bb_layer_ps)
            _in_layer: set = set()
            _sizes = []
            for k, _lps in enumerate(_bb_layer_ps):
                _ps = [p for p in _lps if id(p) in _keep]
                _in_layer |= {id(p) for p in _ps}
                if not _ps:
                    continue
                _lr = args.lr * args.backbone_lr_mult * (_llrd ** (_n - 1 - k))
                groups.append({"params": _ps, "lr": _lr, "name": f"bbL{k}"})
                _sizes.append(len(_ps))
            # pre/post_layernorm 不属任何"层"⇒ 归入顶层档（decay^0），与顶层同 lr
            _rest = [p for p in bb_ps if id(p) not in _in_layer]
            if _rest:
                groups.append({"params": _rest,
                               "lr": args.lr * args.backbone_lr_mult,
                               "name": "bbLN"})
            print(f"[opt] ★ LLRD 启用：逐层 lr = {args.lr}×{args.backbone_lr_mult}"
                  f"×{_llrd}^k（k=0 顶层 → {_n - 1} 底层）"
                  f"｜顶层 {args.lr * args.backbone_lr_mult:.2e} → "
                  f"底层 {args.lr * args.backbone_lr_mult * (_llrd ** (_n - 1)):.2e}"
                  f"｜每层参数数 {_sizes}")
        else:
            groups.append({"params": bb_ps, "lr": args.lr * args.backbone_lr_mult,
                           "name": "backbone"})
    # ★★★ 2026-10-07：区域适配器单独一组。
    #   理由：它是**从零初始化**的模块（不像 LoRA 旁路起点为零但仍挂在预训练主干上），
    #   且相关对照是在 **同一 lr** 下做的（该路线未用 LoRA，全塔 lr 一致）。
    #   ⇒ 保守取 **= args.lr × 1.0**（与 LoRA 同档），不放大也不缩小，
    #     这样"唯一变量 = 有没有这个模块"最干净；若将来要调，
    #     用 `--region-lr-mult` 显式给倍数，别改这里。
    if reg_ps:
        groups.append({"params": reg_ps, "lr": args.lr * args.region_lr_mult,
                       "name": "region"})
        print(f"[opt] 区域适配器学习率 = {args.lr}×{args.region_lr_mult}"
              f" = {args.lr * args.region_lr_mult:.2e}（从零初始化）")
    if not groups:
        raise SystemExit("[opt] 没有任何可训练参数，无法训练")
    # ★★ SAM 开关必须**先于使用处**定义（2026-10-07 修：曾把它写在 scaler 那里，
    #   而 SAM 实例化在更前面 ⇒ UnboundLocalError。本地验证拦住了这次）
    _sam_rho = float(getattr(args, "sam_rho", 0.0) or 0.0)

    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)

    # ★★★ 2026-10-08 Lookahead 包装（默认关闭；k/alpha 文献标准值 5/0.5）
    _la_on = bool(getattr(args, "lookahead", False))
    if _la_on:
        from swa_lookahead import Lookahead
        opt = Lookahead(opt, k=args.la_steps, alpha=args.la_alpha)
        print(f"[lookahead] 已启用：k={args.la_steps}, α={args.la_alpha}"
              f"（每 {args.la_steps} 步 slow←slow+α(fast−slow)，fast←slow）")

    # ★★★ 2026-10-07 SAM 包装（默认 rho=0 ⇒ `_sam is None` ⇒ **完全走原路径**）
    # ★★★ 2026-10-09 增变体开关：--sam-variant sam|asam（默认 sam＝原路径逐字不变）
    _sam = None
    if _sam_rho > 0:
        _variant = str(getattr(args, "sam_variant", "sam") or "sam").lower()
        if _variant == "asam":
            from asam import ASAM as _SamCls
            print(f"[sam] ★ 变体 = ASAM（尺度自适应：ε=ρ·w²⊙g/‖|w|⊙g‖）"
                  f"｜靶心 = 参数组尺度异质（LoRA/头/logit_scale）")
        else:
            from sam import SAM as _SamCls
            print(f"[sam] 变体 = 原版 SAM（ε=ρ·g/‖g‖，所有参数一视同仁）")
        _sam = _SamCls(params, opt, rho=_sam_rho)
        print(f"[sam] 已启用：ρ={_sam_rho}（每步两次前向；AMP 已强制关闭）")
        print(f"[sam] 扰动参数张量数 = {len(_sam.params)}"
              f"｜★ 解析：min_w max_{{‖ε‖≤ρ}} L(w+ε) ⇒ 求平坦极小 ⇒ 抗输入扰动")

    # ★★★ 2026-10-08 SWA 初始化（默认关闭；与优化器正交，训练末期对 raw 权重平均）
    _swa = None
    if getattr(args, "swa", False):
        from swa_lookahead import SWA
        _swa = SWA(model, start_epoch=args.swa_start)
        print(f"[swa] 已启用：从 epoch {args.swa_start} 起对权重做滑动平均"
              f"（训练结束后用平均权重当提交权重）")

    steps_per_epoch = max(1, len(dl_tr))
    # ★ 梯度累积：调度器与「跳步比例」都该按**优化步**算，不是 batch 数。
    accum = max(1, args.grad_accum)
    opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / accum))
    total_steps = opt_steps_per_epoch * args.epochs
    # ★ OneCycleLR 的 warmup 阶段需要足够多的步数才有意义。
    # 冒烟场景下 total_steps 可能只有个位数，warmup 会退化成
    # 「第 0 步就冲顶然后立刻退火」，训练等于没退火就结束了。
    # 步数太少时改用余弦退火，曲线更稳且无 warmup 依赖。
    #
    # ⚠️ 这段注释原先还写着「那条 scheduler.step() before optimizer.step()
    # 警告其实是构造期的初始探测」—— **那是猜错的**。实测：单独构造
    # OneCycleLR / CosineAnnealingLR 都不产生该警告；真正的来源是 AMP 下
    # scaler 跳过 opt.step() 却照常推进 sched（见下方训练循环里的说明，已修）。
    # 教训：注释里的"原因"若只是推测，写上就是给未来的自己挖坑。
    if total_steps < 100:
        print(f"[sched] total_steps={total_steps} 偏少，改用 CosineAnnealingLR"
              f"（避开 OneCycleLR 的 warmup 退化）")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps), eta_min=args.lr * 0.05)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups], total_steps=total_steps,
            pct_start=0.1)

    # ★★ SAM 开启时**强制关闭 AMP**：AMP 的 scaled-grad 与 SAM 的扰动步长会互相打架，
    #   且 `second_step` 里的 `base.step()` 无法走 scaler 的跳步逻辑。
    #   我们是 LoRA（可训练参数仅 ~37M）⇒ fp32 显存完全够，用简单换可靠。
    _amp_on = (device == "cuda" and not args.no_amp and _sam_rho <= 0)
    if _sam_rho > 0 and args.no_amp is False and device == "cuda":
        print(f"[sam] ρ={_sam_rho} ⇒ ★ 强制关闭 AMP（走 fp32）以保证 SAM 的扰动/恢复正确")
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_on)

    # 断点续训
    start_ep = 1
    # ★ `sel` = 用于**选优**的那个数（默认等于 acc）。单独一个键，
    #   是为了让"选优口径"可以换（全体 / 低相似 A 半）而不动历史语义。
    best = {"acc": -1.0, "sel": -1.0, "state": None, "ep": -1, "is_ema": False}
    # 每轮 val_acc 轨迹，供训练结束后的「随机水平检测」判断是真没训够还是被压平。
    _acc_hist: list[float] = []
    # 每轮 (低相似全体, A 半, B 半)；仅在 --track-lowsim 时有值
    _low_hist: list[tuple[float, float, float]] = []
    ckpt_p = Path(args.ckpt)
    if args.resume and ckpt_p.exists():
        ck = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        best = ck.get("best", best)
        # 旧版 ckpt 里的 best 没有 sel 键 —— 补上，否则第一轮就 KeyError
        best.setdefault("sel", best.get("acc", -1.0))
        print(f"[resume] 从 {ckpt_p} 恢复，epoch {ck['epoch']} -> {start_ep}")

    print(f"[train] {args.epochs} 轮 x {steps_per_epoch} 步"
          f"（优化步 {opt_steps_per_epoch}/轮，grad_accum={accum}），"
          f"batch={args.batch}，lr(lora)={args.lr} lr(head)={args.lr*head_mult}"
          f"（{head_mult:g}倍）")
    print(f"[train] kind={args.kind} use_cl={args.use_cl} "
          f"class_balance={C.USE_CLASS_BALANCE} amp={scaler.is_enabled()}")

    # ==============================================================
    # ★★★ 「低相似」双列追踪 —— 修一个真实的选优缺陷
    #
    # 缺陷是什么：上面这段验证只报**全体 val 准确率**，而本地留出集里
    # **18.97% 是近重复图**（>0.95 相似度，见 `probe_leak_full.py`）。
    # 近重复图靠**记忆**就能做对 ⇒ 「全体 val」会在模型已经开始背噪声之后
    # **继续上涨**。于是"按全体 val 选最优 epoch"可能选到的是**记忆效应**
    # 最强的那个检查点，而不是迁移最好的那个。
    # 实测痕迹：基线 6 轮的 val 增量是 0.0535→0.0285→0.0173→**0.0038→0.0034**，
    # 而同期的 train loss 从 0.0833 掉到 0.0662（**还降 20%**）——
    # 形状上就是"val 走平、loss 还在掉"，而这正是开始拟合噪声的签名。
    # 结论：**低相似那一列从来没被逐轮记录过，所以无从知道它什么时候见顶。**
    #
    # 修法：一次前向拿到逐样本预测，再用**三套掩码**分别算三列 ——
    # 「全体」/「低相似 A 半」/「低相似 B 半」。成本可忽略（同一批预测）。
    #
    # ★ 为什么要把低相似拆成 A/B 两半：如果**用低相似列选 epoch、又用低相似列
    #   报数**，那就是"在同一份数据上既选方案又评方案"，选出的数字必然虚高
    #   （本项目已量化过这个偏差：约 +0.005，与标准误同量级 ⇒ 足以吃掉结论）。
    #   所以选优只看 A，报数只看 B。
    low_masks = None
    if args.track_lowsim:
        # ★ 必须先拦掉 --limit：低相似掩码要用**全量标签序**去索引冻结特征矩阵，
        #   而 `--limit` 之后 `labels` 是子集 ⇒ `va_idx` 会指向**错误的行**，
        #   `Xf` 照样能索引（形状对得上）⇒ **不报错，只是悄悄算错**。
        #   这正是本项目反复吃亏的那一类「静默错误」，所以直接拒绝这个组合。
        if args.limit:
            raise SystemExit(
                "[track] 🛑 --track-lowsim 不能与 --limit 同时用："
                "低相似掩码按**全量标签序**索引冻结特征矩阵，"
                "而 --limit 之后 labels 已是子集，va_idx 会指向错误的行"
                "（形状对得上、不报错，结果默默算错）。\n"
                "        冒烟请改用 `--epochs 1` 跑全量（约 13 min）。")
        # `finetune_lora.py` 在 `src/` 下，而 `sys.path` 里只有 src（PYTHONPATH=src）
        # ⇒ 直接 import 会 `ModuleNotFoundError`（实测踩到，白跑了一次 setup）。
        # 这里显式把工程根的 `scripts/` 加进来。
        import sys as _sys
        _sd = Path(__file__).resolve().parents[1] / "scripts"
        if str(_sd) not in _sys.path:
            _sys.path.insert(0, str(_sd))
        import compare_lora_vs_frozen as CLF
        Xf = np.load(C.CACHE_DIR / "feat_train.npy")
        va_idx = np.where(val_mask)[0]
        low, ms = CLF.low_sim_mask(Xf, labels, va_idx,
                                   np.arange(len(labels))[~val_mask])
        # ★ 规模断言：口径一旦漂移，"低相似"就不是同一批样本了。
        #   （本项目在"标尺用错特征"上踩过一次：子集从 3089 胀到 6725。）
        if len(low) != len(va_labels):
            raise SystemExit(f"[track] 🛑 低相似掩码 {len(low)} ≠ 验证集 {len(va_labels)}"
                             f" ⇒ 与验证集顺序对不上，中止")
        if int(low.sum()) != 3089:
            print(f"[track] ⚠️ 低相似子集 {int(low.sum())} 张 ≠ 基准 3089 张"
                  f" ⇒ 口径与其它脚本不同，本轮的「低相似」列**不可横向比较**")
        lowA = low.copy()
        lowA[np.where(low)[0][1::2]] = False     # 偶数位 → A
        lowB = low & ~lowA                       # 奇数位 → B
        low_masks = (low, lowA, lowB)
        print(f"[track] ★ 逐轮追踪已开启：低相似 {int(low.sum())} 张"
              f"（A 选优 {int(lowA.sum())} / B 报数 {int(lowB.sum())}）；"
              f"选优口径 = {args.select_by}")
        if args.select_by == "lowsim_a":
            print(f"[track] ⚠️ 选优改用低相似 A 半 ⇒ **报数请看 B 半**，"
                  f"不要报 A 半（那是「选出来的」数字）")
    elif args.select_by != "full":
        raise SystemExit("[track] 🛑 --select-by 非 full 时必须同时给 --track-lowsim")

    # ==============================================================
    # ★★★ 表征漂移约束的教师模型（2026-09-28 新增，官方点名最多的方向）
    #
    # 赛题原文反复点名这一方向（五-2「表征约束」、五-3「最大程度保留和利用
    # CLIP 预训练先验知识」、十一-一-2「特征约束、灾难性遗忘抑制」、
    # 十-二「约束表征漂移」、十二-6「表征一致性约束及灾难性遗忘抑制」）——
    # 而且题目本身就是"网络含噪数据微调基础模型"，核心风险正是
    # **在噪声梯度下把预训练知识毁掉**（赛题背景里直接写了"灾难性遗忘"）。
    #
    # 实现：deepcopy 一份**当前（= 未微调）状态**的 model 当教师，全程 eval +
    #   requires_grad=False，每步用**同一个增强视图** x_in 前向取特征。
    #
    # ★ 为什么必须在**在线**前向、不能用预计算的 feat_train.npy：
    #   训练变换含 RandomResizedCrop(scale 0.7~1.0) + RandomHorizontalFlip
    #   ⇒ 每个 step 看到的视图都不同，而预计算特征是"中心裁剪"那一版
    #   ⇒ 两者**不是同一个输入**，拿它当教师等于让学生去拟合错位的目标。
    # ★ 代价：每步多一次无梯度前向，训练时间约 +40~50%。
    # ==============================================================
    teacher = None
    drift_acc, drift_n = 0.0, 0
    if float(args.drift_lambda) > 0:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for _p in teacher.parameters():
            _p.requires_grad = False
        print(f"[drift] ★ 表征漂移约束开启 λ={args.drift_lambda:g}；"
              f"教师 = 未微调初始模型副本（冻结 "
              f"{sum(p.numel() for p in teacher.parameters()):,} 参数）")
        print(f"[drift]   ⚠️ 每步多一次无梯度前向 ⇒ 训练时间约 +40~50%")
    else:
        print("[drift] 未启用（--drift-lambda 0）")

    t0 = time.time()
    skipped = 0       # AMP 因 inf/nan 梯度跳过的优化步数（累计）
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        tot, nb = 0.0, 0
        for bi, batch in enumerate(dl_tr, 1):
            idx = None      # ★ 显式初值：两元组 batch（无 weights）时不定义 idx，
                            #   而下面筛选累积要判断它 ⇒ 不给初值会 NameError
            # 有 weights 时 collate 出来是 (x, y, idx) 三元组
            if len(batch) == 3:
                x, y, idx = batch
                idx = idx.to(device, non_blocking=False)
                sw = w_all_gpu[idx]
            else:
                x, y = batch
                sw = None
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            # ★ 梯度累积（--grad-accum>1）：每 accum 个 batch 才走一次 opt.step()。
            #   窗口的**第一个** batch 清零梯度；窗口的**最后一个**（或本轮
            #   最后一个 batch，兜住不能被 accum 整除的余数）才裁剪+更新。
            #   loss 要除以 accum 才是"这个窗口的平均梯度"，否则等效于把 lr 放大 accum 倍。
            win_start = ((bi - 1) % accum == 0)
            is_step = (bi % accum == 0) or (bi == steps_per_epoch)
            if win_start:
                opt.zero_grad(set_to_none=True)
            # ------------------------------------------------ mixup（可选）
            # x' = λ·x + (1-λ)·x[perm]，对应损失 λ·L(x',y) + (1-λ)·L(x',y[perm])。
            # ★ 两个方向**各自的样本权重也要一起置换**（sw[perm]）：CL 降权系数
            #   是逐样本的，不置换就等于把 A 样本的噪声系数安到 B 样本头上。
            mix_lam = None
            x_in, y_b, sw_b = x, None, None
            idx_b = None
            if args.mixup > 0 and x.size(0) > 1:
                mix_lam = float(np.random.beta(args.mixup, args.mixup))
                _perm = torch.randperm(x.size(0), device=x.device)
                x_in = mix_lam * x + (1.0 - mix_lam) * x[_perm]
                y_b = y[_perm]
                sw_b = None if sw is None else sw[_perm]
                # ★ 2026-10-06：软目标也要跟着置换（否则 B 方向会查回 A 样本的软标签）
                idx_b = None if idx is None else idx[_perm]
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                # ★ 表征漂移约束开启时走 forward_with_feat（同一前向多返回特征），
                #   否则走原 forward —— 两条路等价，不开时**零额外开销**。
                if teacher is not None:
                    logits, f_stu = model.forward_with_feat(x_in)
                else:
                    logits, f_stu = model(x_in), None
                # ★ 走与冻结路线**完全相同**的损失函数，这样两条路线的分数才可比。
                # 早先这里误写成写死的 kind="ce" 且 sample_w=None，等于把
                # CL 降权与 SCE 全部丢弃 —— 那就是在拿「没有抗噪的 LoRA」
                # 去和「有抗噪的线性头」比，结论必然误导。
                _lkw = dict(T=T_t, kind=args.kind, alpha=args.sce_alpha,
                            beta=args.sce_beta, q=C.GCE_Q,
                            label_smooth=C.LABEL_SMOOTH)
                # ★ margin：只压正确类的 logit，单位换算成余弦空间的 m×scale。
                #   用 clone 实现（不原地改 logits）⇒ mixup 两个方向各取所需。
                _m = 0.0
                if args.margin > 0:
                    _m = float(args.margin) * float(
                        model.logit_scale.clamp(max=math.log(100.0)).exp().detach())
                # ★★ 2026-10-06 软目标分支（`soft_lut is None` ⇒ 下面这行与原来逐字相同）
                if soft_lut is not None:
                    loss_raw = NZ.compute_loss_soft(
                        _margin_logits(logits, y, _m),
                        _soft_targets(soft_lut, idx, y, K, device),
                        y, sample_w=sw, **_lkw)
                    if mix_lam is not None:
                        loss_raw = (mix_lam * loss_raw
                                    + (1.0 - mix_lam) * NZ.compute_loss_soft(
                                        _margin_logits(logits, y_b, _m),
                                        _soft_targets(soft_lut, idx_b, y_b, K, device),
                                        y_b, sample_w=sw_b, **_lkw))
                else:
                    loss_raw = NZ.compute_loss(_margin_logits(logits, y, _m),
                                               y, sample_w=sw, **_lkw)
                    if mix_lam is not None:
                        loss_raw = (mix_lam * loss_raw
                                    + (1.0 - mix_lam) * NZ.compute_loss(
                                        _margin_logits(logits, y_b, _m),
                                        y_b, sample_w=sw_b, **_lkw))
                loss = loss_raw / accum
                # ★ 表征漂移约束：1 − cos(student, teacher)，teacher = **未微调的
                #   初始模型副本**（下方建好、全程冻结）。
                #   用余弦距离而非 MSE：CLIP 特征的方向才是语义所在，模长无意义。
                #   除以 accum 与主损失同口径（梯度累积时 λ 保持同一尺度）。
                if teacher is not None:
                    with torch.no_grad():
                        f_tea = teacher.encode(x_in)
                        f_tea = F.normalize(f_tea.float(), dim=-1)
                    drift = (1.0 - (f_stu * f_tea).sum(dim=-1)).mean()
                    loss = loss + float(args.drift_lambda) * (drift / accum)
                    drift_acc += float(drift.detach())
                    drift_n += 1
                # ★ 筛选判据：额外算一次**不带 sample_w** 的逐样本损失。
                #   刻意不改动上面那条既有 loss 路径一个字 —— 这样"开筛选"与
                #   "关筛选"的梯度完全一致，最终分数差异只可能来自**数据选择本身**，
                #   而不是被顺手改掉的损失函数（本项目在这一点上踩过坑：
                #   早先训练循环里误写成写死的 kind="ce"，等于把抗噪机制全丢了）。
                #   代价约 2%（相对 forward+backward），可忽略。
                if sel_on:
                    per_sel = NZ.compute_loss(logits, y, return_per=True, **_lkw)
            if sel_on and idx is not None:
                _ii = idx.detach().cpu().numpy()
                sel_sum[_ii] += per_sel.detach().float().cpu().numpy()
                sel_cnt[_ii] += 1
            stepped = False
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if is_step:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in params], args.clip)
                    # ★ 判断这一步优化器**是否真的更新了**。
                    # AMP 下 `scaler.step()` 遇到 inf/nan 梯度会**跳过** opt.step()，
                    # 若不加判断、照样调 sched.step()，学习率曲线就比实际更新
                    # **跑在前面**，PyTorch 也会打
                    # `lr_scheduler.step() before optimizer.step()` 警告。
                    # 实测确认这个警告不是误报（构造调度器、正常顺序都不触发，
                    # 只有 AMP 跳步时才触发）。
                    #
                    # 判据用「update() 后的 scale 是否变小」——跳步时 scaler 会把
                    # scale 减半，正常步保持不变。实测：正常 65536→65536，
                    # 跳步 65536→32768。
                    # ⚠️ 必须在 `update()` **之后**读；在 step/update 之间读到的
                    # 还是旧值，两种情形都是 65536，判断不出来。
                    scale_before = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    stepped = scaler.get_scale() >= scale_before
                    if not stepped:
                        skipped += 1
            else:
                loss.backward()
                if is_step:
                    if _sam is not None:
                        # ★★★ SAM 的两步（2026-10-07）
                        #   ① 用【累积梯度】扰动权重：w += ρ·g/‖g‖
                        #   ② 用【当前 batch】在扰动点上重算 forward/backward 得 g'
                        #   ③ 恢复权重 w −= ρ·g/‖g‖，再用 g' 做真正更新
                        #   ⚠️ 顺序不能反（先恢复、再 step），否则不是 SAM。
                        _sam.first_step(scale=1.0)
                        opt.zero_grad(set_to_none=True)
                        # ★ 第二次前向：口径必须与上方**逐字相同**
                        #   （软目标 + mixup + margin 全带上，只把 logits 换成扰动点的）
                        # ⚠️ 返回类型不同：`forward_with_feat` 返回 (logits, feat)，
                        #   而 `model(x_in)` **直接返回 logits**（不是 tuple）
                        #   ⇒ 2026-10-07 修：曾统一写成 `[0]` ⇒ IndexError（本地验证拦住）
                        _logits_s = (model.forward_with_feat(x_in)[0]
                                     if teacher is not None else model(x_in))
                        if soft_lut is not None:
                            _loss_s = NZ.compute_loss_soft(
                                _margin_logits(_logits_s, y, _m),
                                _soft_targets(soft_lut, idx, y, K, device),
                                y, sample_w=sw, **_lkw)
                            if mix_lam is not None:
                                _loss_s = (mix_lam * _loss_s
                                           + (1.0 - mix_lam) * NZ.compute_loss_soft(
                                               _margin_logits(_logits_s, y_b, _m),
                                               _soft_targets(soft_lut, idx_b, y_b, K, device),
                                               y_b, sample_w=sw_b, **_lkw))
                        else:
                            _loss_s = NZ.compute_loss(_margin_logits(_logits_s, y, _m),
                                                      y, sample_w=sw, **_lkw)
                            if mix_lam is not None:
                                _loss_s = (mix_lam * _loss_s
                                           + (1.0 - mix_lam) * NZ.compute_loss(
                                               _margin_logits(_logits_s, y_b, _m),
                                               y_b, sample_w=sw_b, **_lkw))
                        (_loss_s / accum).backward()
                        torch.nn.utils.clip_grad_norm_([p for p in params], args.clip)
                        _sam.second_step(scale=1.0)
                        stepped = True
                    else:
                        torch.nn.utils.clip_grad_norm_([p for p in params], args.clip)
                        opt.step()
                        stepped = True
            if stepped:
                try:
                    sched.step()
                except Exception:  # noqa: BLE001
                    pass
                # EMA 累积：必须放在 `stepped` 分支里。AMP 因 inf/nan 跳步时
                # 权重根本没变，此时累积等于把"新鲜度"算错（影子被多衰减一次）。
                if ema_pairs and ep >= args.ema_start:
                    _d = args.ema_decay
                    with torch.no_grad():
                        for _p, _t in ema_pairs:
                            _t.mul_(_d).add_(_p.detach().float(), alpha=1.0 - _d)
            tot += loss_raw.item()
            nb += 1
            if bi % args.log_every == 0:
                el = time.time() - ep_t0
                eta = (steps_per_epoch - bi) * (el / max(bi, 1))
                print(f"[ep{ep:2d}] {bi}/{steps_per_epoch} "
                      f"loss={tot/max(nb,1):.4f} 已用 {el/60:.1f}min "
                      f"预计还需 {eta/60:.1f}min", flush=True)

        # ---------------------------------------- 噪声筛选：决定下一轮用哪些样本
        if sel_on and ep >= args.sel_start:
            _prog = (ep - args.sel_start + 1) / max(1, args.epochs - args.sel_start + 1)
            _frac = args.sel_frac * min(1.0, _prog)      # 课程式：筛的比例渐进
            _flag = NZ.classwise_small_loss_flag(sel_sum, sel_cnt, tr_labels,
                                                 _frac, args.sel_min_keep)
            _new = int((_flag & ~sel_flag).sum())
            sel_flag |= _flag                            # 单向：只减不召回
            w_tr[sel_flag] = 0.0
            # ★ 重新归一化是**必须**的：被剔除样本越多，`per.mean()` 的分母越大、
            #   有效梯度越小 —— 那等于在筛样本的同时**悄悄调低了学习率**，
            #   分数一变就说不清是"筛得对"还是"学得慢"。
            if (w_tr > 0).any():
                w_tr = w_tr / w_tr.mean()
            w_all_gpu = torch.as_tensor(w_tr, dtype=torch.float32, device=device)
            print(f"[sel] ep{ep} 新增剔除 {_new} 张，累计剔除 {int(sel_flag.sum())} 张"
                  f"（{sel_flag.mean():.1%}），本轮 frac={_frac:.3f}，"
                  f"剩余 {int((w_tr > 0).sum())} 张参与训练", flush=True)

        # -------------------------------------------------- 每轮验证
        # ★ EMA 开启时**两条曲线都量**：raw = 训练中的真实权重，EMA = 滑动平均。
        #   两个都打出来，"EMA 到底有没有用"才可归因；只报一条就只剩猜。
        #   代价是每轮多一次前向验证（约 30~60 s），相对一轮训练可忽略。
        y_va, pred_raw = _evaluate_preds(model, dl_va, device)
        pred_ema = None
        if ema_pairs and ep >= args.ema_start:
            _bak = _ema_apply(ema_pairs)
            try:
                _, pred_ema = _evaluate_preds(model, dl_va, device)
            finally:
                _ema_restore(_bak)
        # 选优/出包用哪一份：EMA 开启时默认用 EMA（--ema-off 可退回 raw）
        use_ema = pred_ema is not None and not args.ema_off
        pred_va = pred_ema if use_ema else pred_raw
        acc = float((pred_va == y_va).mean())
        _acc_hist.append(float(acc))
        extra = ""
        if low_masks is not None:
            m_all, mA, mB = low_masks
            accL = float((pred_va[m_all] == y_va[m_all]).mean())
            accLA = float((pred_va[mA] == y_va[mA]).mean())
            accLB = float((pred_va[mB] == y_va[mB]).mean())
            _low_hist.append((accL, accLA, accLB))
            extra = (f" 低相似={accL:.4f} [A {accLA:.4f} / B {accLB:.4f}]")
        if pred_ema is not None:
            extra += (f"  [raw {float((pred_raw == y_va).mean()):.4f} "
                      f"EMA {float((pred_ema == y_va).mean()):.4f}]")
        # ★ 表征漂移量：本轮 1−cos(student, teacher) 的均值。判读该臂的主要中间量 ——
        #   若它一直不掉，说明约束没起作用（λ 太小 / 梯度被主损失淹没）；
        #   若掉得太快，说明 λ 过大、模型被拉回原点、失去微调效果。
        if drift_n > 0:
            extra += f"  [drift {drift_acc / drift_n:.5f}]"
            drift_acc, drift_n = 0.0, 0
        print(f"[ep{ep:2d}] ★ 完成 loss={tot/max(nb,1):.4f} val_acc={acc:.4f}{extra} "
              f"用时 {(time.time()-ep_t0)/60:.1f}min", flush=True)

        # ★ 选优口径：默认仍是全体 val（与历史基线可比）；
        #   开了 --track-lowsim + --select-by lowsim_a 才改用低相似 A 半。
        #   两者都记录下来，方便事后对比"两种选优会挑到不同的 epoch 吗"。
        sel = acc if (low_masks is None or args.select_by == "full") \
            else _low_hist[-1][1]
        improved = sel > best["sel"]
        if improved:
            # ★ 选优用的指标来自哪份权重，就必须存哪份权重。
            #   若指标是 EMA 的、存下来的是 raw 的，那么"出包权重"和
            #   "选它的那个指标"不是同一个东西 —— 榜单会直接给出反例。
            _st = None
            if use_ema:
                _bak2 = _ema_apply(ema_pairs)
                try:
                    _st = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
                finally:
                    _ema_restore(_bak2)
            else:
                _st = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
            best = {"acc": acc, "sel": sel, "state": _st, "ep": ep,
                    "is_ema": bool(use_ema)}
            if low_masks is not None:
                best["low"] = _low_hist[-1]
            print(f"[ep{ep:2d}] ↑ 新最优（按 {args.select_by}）")

        # ★ 2026-10-08 SWA：每轮结束（epoch >= start）把当前 raw 权重并入滑动平均
        if _swa is not None:
            _swa.update(ep)

        # 每轮都存一份 ckpt（含优化器状态，可续训），便于中断恢复
        # ★ `img_size` 必须写进检查点（2026-09-23 新增）。
        #   它是一个**不写在参数列表里**的全局量（由环境变量 AIC_IMG_SIZE 驱动），
        #   若不落盘，加载方就会**静默按默认 224 去跑 288 的权重** ——
        #   位置编码插值关掉、TTA 视图几何也全按 224 算，包会整体作废且**看不出原因**。
        #   与前面 `targets` 的静默退回是同一类坑，处理方式也一样：记下来 + 加载时校验。
        # ★★ 区域适配器的权重**单独存**：它挂在 `clip_model.vision_model.region_adapter`
        #   下，而 `model` 是 `LoRAClipClassifier` —— 键前缀不同，
        #   **不会**出现在 `model.state_dict()` 的常规结构里。
        #   ⇒ 不单独存 ⇒ 推理侧重建了适配器却是随机权重 ⇒ **静默失分**。
        _reg_sd = (region_adapter.state_dict()
                   if region_adapter is not None else None)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "num_classes": K, "r": args.r, "kind": kind,
                    "img_size": C.IMG_SIZE,
                    "region_attn": int(getattr(args, "region_attn", 0) or 0),
                    "region_adapter": _reg_sd,
                    "args": vars(args)}, ckpt_p)
        print(f"[ep{ep:2d}] ckpt -> {ckpt_p}")
        # ★ 另存一份**只含权重**的逐轮快照。
        #   原因：上面那份 ckpt_p 每轮**覆盖同一个文件**，训完只剩末轮；
        #   而本项目正好需要"回头比较不同 epoch 的迁移表现"——
        #   没有逐轮快照就只能重训一遍（72 min）才能回答。
        #   只存权重（不含优化器状态）是为了省磁盘：约 16MB/份。
        if args.save_every_epoch:
            snap = ckpt_p.with_name(f"{ckpt_p.stem}_ep{ep:02d}.pt")
            torch.save({"model": {k: v for k, v in model.state_dict().items()},
                        "num_classes": K, "r": args.r, "backbone_kind": kind,
                        "img_size": C.IMG_SIZE,
                        "region_attn": int(getattr(args, "region_attn", 0) or 0),
                        "region_adapter": _reg_sd,
                        "ep": ep, "val_acc": acc,
                        "low": None if low_masks is None else _low_hist[-1],
                        "args": vars(args)}, snap)
            print(f"[ep{ep:2d}] 快照 -> {snap.name}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # ★ 2026-10-08 SWA：若有，用平均权重**替换**最终提交权重（标准 SWA 用法）
    if _swa is not None:
        _swa.apply_to(model)
        print(f"[swa] ★ 已加载 SWA 平均权重（共 {_swa.count} 个 epoch 平均，"
              f"start={args.swa_start}）⇒ 作为提交权重")
        # ★  同尺读数：SWA 权重自己的 val_acc。没有它，"平均把模型
        #   平均塌了"这类灾难只能靠烧一次榜分额度才能发现；有了它，
        #   训练结束当场就能和选优值（同一把 val 尺）对照。
        _yv, _pred_swa = _evaluate_preds(model, dl_va, device)
        _acc_swa = float((_pred_swa == _yv).mean())
        print(f"[swa] SWA 权重 val_acc={_acc_swa:.4f}"
              f"（对照：选优 {best['acc']:.4f} @ep{best['ep']}，同一把 val 尺）")

    # 报告 AMP 跳步次数。少量跳步是正常的（梯度偶发溢出）；
    # 但**持续跳步**说明训练不稳定，此时 loss 曲线与 lr 曲线都会失真，
    # 不能拿这一轮的结论当数。
    if skipped:
        total_steps_done = opt_steps_per_epoch * args.epochs
        frac = skipped / max(total_steps_done, 1)
        print(f"[amp] 因 inf/nan 梯度跳过 {skipped} 步"
              f"（占 {frac:.1%}）；这些步的 lr 未推进（已与调度器对齐）")
        if frac > 0.10:
            print(f"[amp] ⚠️ 跳步比例偏高，训练可能不稳定："
                  f"建议降 --lr 或加大 --clip")
    # 单独存一份「最优权重」，推理时用这个
    best_p = ckpt_p.with_name(ckpt_p.stem + "_best.pt")
    torch.save({"model": {k: v for k, v in model.state_dict().items()},
                "num_classes": K, "r": args.r, "backbone_kind": kind,
                "img_size": C.IMG_SIZE,
                "best_ep": best["ep"], "best_val_acc": best["acc"],
                "best_is_ema": bool(best.get("is_ema", False)),
                "args": vars(args)}, best_p)
    print(f"\n[train] 完成，总用时 {(time.time()-t0)/60:.1f} min")
    print(f"[train] 最优 epoch={best['ep']} val_acc={best['acc']:.4f}"
          f"（权重来源：{'EMA' if best.get('is_ema') else 'raw'}）")
    print(f"[train] 最优权重 -> {best_p}")

    # ==================================================================
    # ★★ 随机水平检测 —— 拦住「损失函数把模型压成均匀分布」这类塌缩
    # ==================================================================
    #
    # 这一条是拿真实代价换来的。实测：K=20 的子集上，LoRA 用默认 `--kind sce`
    # 训 6 轮，loss 从 1.6 稳降到 1.08（**看着在正常收敛**），
    # 但 val_acc 始终卡在 0.0375~0.05，而随机水平是 1/20 = 0.05。
    # 换成朴素 CE、别的参数一字不动，val_acc 立刻从 0.05 升到 0.53。
    #
    # 机理：SCE = alpha*CE + beta*RCE，默认 alpha=0.1、beta=1.0 ⇒
    # **RCE 的权重是 CE 的 10 倍**。而 RCE = -Σ p log p 完全不含标签，
    # 它唯一的作用就是把预测分布推向均匀。K 类时 RCE 的极小值就是均匀分布，
    # 于是准确率被死死摁在 1/K 上 —— 且 loss 还在降，极具欺骗性。
    #
    # 为什么冻结路线没暴露：那时只训线性头，输入是**已含类别判别信息的冻结特征**，
    # 头只要放大特征里的方向就能压过 RCE 的均摊压力。
    # 端到端微调时骨干附近的梯度更弱，RCE 就赢了。
    # ⇒ **这个失败模式是端到端路线特有的。**
    # 记录每轮 val_acc 轨迹，供下面的随机水平检测判断「是否一直在原地踏步」。
    # 只要 acc 轨迹全程贴地，就说明不是"还没训够"，而是模型被压平了。
    _rnd = 1.0 / max(K, 1)
    _traj = " ".join(f"{a:.3f}" for a in _acc_hist)
    if _acc_hist:
        print(f"[train] val_acc 轨迹：{_traj}")
    _flat = bool(_acc_hist) and max(_acc_hist) <= _rnd * 1.35
    if best["acc"] <= _rnd * 1.35:
        print()
        print("=" * 68)
        print(f"[guard] 🛑 结果停在本任务的**随机水平**附近，几乎可以断定训练没学进去：")
        print(f"[guard]    最优 val_acc = {best['acc']:.4f}，而随机猜 = 1/{K} = {_rnd:.4f}")
        if _flat:
            print(f"[guard]    而且 {len(_acc_hist)} 轮全程贴地（{_traj}）—— 不是「没训够」，")
            print(f"[guard]    是**模型被压平了**。")
        else:
            print(f"[guard]    损失看着在降，但准确率不动 —— 这是「被压成均匀分布」的典型特征。")
        print(f"[guard]")
        print(f"[guard]    最可能的原因（已实测确认）：--kind {args.kind} 的 SCE 权重配比")
        print(f"[guard]    alpha={C.SCE_ALPHA} / beta={C.SCE_BETA} 里 beta 太大。")
        print(f"[guard]    SCE = alpha*CE + beta*RCE，而 RCE=-Σp·logp **完全不含标签**，")
        print(f"[guard]    它唯一的作用就是把预测推向均匀 ⇒ K 类时极小值恰是均匀分布")
        print(f"[guard]    ⇒ 准确率被摁在 1/K。beta 是 alpha 的 {C.SCE_BETA/max(C.SCE_ALPHA,1e-12):.0f} 倍，")
        print(f"[guard]    RCE 会盖过分类信号。")
        print(f"[guard]")
        print(f"[guard]    先试这个（别的参数一律不动）：")
        print(f"[guard]        --kind ce --no-cl          # 朴素 CE，最快判断装置是否正常")
        print(f"[guard]    实测同一子集：默认 SCE → 0.05；换 --kind ce → 0.53。")
        print(f"[guard]    若想保留抗噪性，再试 --kind gce 或调小 beta。")
        print(f"[guard]")
        print(f"[guard]    ⚠️ **不要**把本结果当成「LoRA 没用 / 容量假设不成立」的证据。")
        print(f"[guard]       它只说明这套损失配比在端到端下不work。")
        print("=" * 68)
        print()

    # ★ 退化检测（collapse guard）。
    # 训练塌掉时 val_acc 会是个**很低但看不出异常**的数（比如 0.125），
    # 而真正致命的信号是「预测标签几乎全落在少数几类上」。
    # 实测踩过：512 张图的探针里，两个配置都只输出 2 个唯一标签，
    # 准确率 8.3%（≈1/12），当时的对照表看起来"LoRA 比冻结差 4 倍"——
    # 完全是假信号。没有这道检查，这种结果会被当成实验结论一路带下去。
    #
    # ⚠️ 这里读的 `pred_results.csv` 是**最近一次**训练/predict 遗留的文件 ——
    # 本函数（训练入口）自己**不写**提交文件，写提交的是 predict 分支。
    # 所以 `n_pred_cls` 反映的是「上一次推理覆盖了多少类」，**不是本次训练的效果**。
    # 想让这个信号指向本次结果，必须在训练后单独跑一次 predict。
    # 之所以仍然留着它：它至少能拦住「上一次就塌了、这次还照抄结论」这种连续误判，
    # 并在报告里留下痕迹。真正的本次效果判据只有一个 —— 排行榜分数。
    n_pred_cls = 0
    try:
        import pandas as _pd
        _sub = C.SUB_DIR / "pred_results.csv"
        if _sub.exists():
            _a = _pd.read_csv(_sub, header=None)
            n_pred_cls = int(_a[1].nunique())
    except Exception:  # noqa: BLE001
        pass
    if 0 < n_pred_cls < max(3, K // 4):
        print(f"[guard] ⚠️ 退化警告：K={K} 类，但预测只落在 {n_pred_cls} 个类上。"
              f"\n[guard]    这通常意味着训练塌了（学习率过大 / 数据太少 / 权重失衡），"
              f"\n[guard]    不要把它当成「本方法不行」的证据。建议："
              f"\n[guard]      · 降 --lr（或 --head-lr-mult）"
              f"\n[guard]      · 增加每类样本数")

    (C.CACHE_DIR / f"lora_train_report_{ckpt_p.stem}.json").write_text(
        json.dumps({"best_ep": best["ep"], "best_val_acc": best["acc"],
                    "args": vars(args), "trainable": n_train, "total": n_total,
                    "n_pred_classes": n_pred_cls},
                   ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")
    return 0


@torch.no_grad()
def _evaluate(model, dl, device, K) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in dl:
        x, y = batch[0], batch[1]     # 容忍 (x,y) 与 (x,y,idx) 两种形态
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
        logits = logits.float()
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _evaluate_preds(model, dl, device) -> tuple[np.ndarray, np.ndarray]:
    """跑一遍验证集，返回 ``(y_true, pred)``，顺序与 ``dl`` 一致。

    与 ``_evaluate`` 的区别只是**把逐样本预测留下来**而不是只留一个准确率 ——
    这样「全体 / 低相似 A 半 / 低相似 B 半」三列可以**一次前向**同时算出来
    （它们本来就是同一批预测、只是掩码不同）。

    ★ 为什么要这样：验证集 7537 张、评估一轮只要十几秒，看起来分开算也无所谓；
    但**分开算的代价不在时间，而在口径** —— 一旦某次只改了其中一个掩码的取法，
    三列之间就悄悄不可比了。同一次前向 + 不同掩码，从结构上消掉这种可能。
    """
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for batch in dl:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=False)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
        ps.append(logits.float().argmax(1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


# ------------------------------------------------------------------ 推理
@torch.no_grad()
def predict(args) -> int:
    """载入 LoRA 权重，对测试集推理并写提交文件。"""
    from data import ClassMap, write_submission

    device = args.device
    meta_p = C.CACHE_DIR / "train_meta.json"
    if not meta_p.exists():
        raise SystemExit("[predict] 缺少 train_meta.json，请先跑 pipeline.py scan")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    K = meta["num_classes"]
    names = meta["class_names"]

    ck = torch.load(args.predict_only, map_location=device)
    kind = ck.get("backbone_kind", "transformers")
    r = ck.get("r", args.r)
    tr_args = ck.get("args", {})
    hidden = int(tr_args.get("hidden", 0))
    targets = tuple(str(tr_args.get("targets", "q,k,v,out")).split(","))
    # ★ 分辨率守卫：与 tta_lora_eval.load_lora_model 同一条，理由见训练侧保存处。
    _ck_img = ck.get("img_size")
    if _ck_img is not None and int(_ck_img) != int(C.IMG_SIZE):
        raise SystemExit(
            f"[predict] 🛑 分辨率不匹配：检查点 IMG_SIZE={_ck_img}，"
            f"当前 C.IMG_SIZE={C.IMG_SIZE}。请用 `AIC_IMG_SIZE={_ck_img} …` 重跑。")

    kind_loaded, clip_model, _proc = load_clip(device)
    clip_model = clip_model.float()
    if r > 0:
        LO.inject_lora(clip_model, r=r, alpha=tr_args.get("alpha"),
                       dropout=0.0, targets=targets,
                       last_n_layers=int(tr_args.get("lora_last_n", 0) or 0),
                       verbose=False)
    # ★ VPT 必须按**训练时的配置**重放注入，否则 state_dict 里的
    #   prompt 权重没有落点 —— 表现为 missing key 警告 + 静默输出错误预测。
    #   这里从 ckpt 自带的 args 快照读，不依赖任何外部文档。
    _vpt_n = int(tr_args.get("vpt", 0) or 0)
    if _vpt_n > 0:
        VPT.inject_vpt(clip_model, _vpt_n,
                       init_std=float(tr_args.get("vpt_std", 0.02) or 0.02),
                       verbose=False)
        print(f"[predict] ★ 已按训练配置重放 VPT 注入（n_prompt={_vpt_n}）")
    model = LoRAClipClassifier(clip_model, kind, K, hidden=hidden).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if _vpt_n > 0:
        _bad = [k for k in list(missing) + list(unexpected) if VPT.VPT_TAG in k]
        if _bad:
            raise SystemExit(
                f"[predict] 🛑 VPT 权重未能对齐（{_bad[:3]} …）——"
                f"说明注入配置与训练时不一致，预测结果不可信，拒绝出包。")
    print(f"[predict] 载入 {args.predict_only}（best_ep={ck.get('best_ep')}, "
          f"val_acc={ck.get('best_val_acc')}）")
    if unexpected:
        print(f"[predict] ⚠️ 未使用到的键 {len(unexpected)} 个：{unexpected[:3]}")
    if missing:
        print(f"[predict] ⚠️ 缺失的键 {len(missing)} 个：{missing[:3]}")
    model.eval()

    te = build_test_manifest()
    ds = LabeledImageDataset(te["path"].tolist(),
                            np.zeros(len(te), dtype=np.int64), train=False)
    ds.tf = eval_transform(te["path"].tolist())
    dl = DataLoader(ds, batch_size=args.eval_batch, shuffle=False,
                    num_workers=args.workers, pin_memory=False,
                    prefetch_factor=2 if args.workers > 0 else None)

    probs_all = []
    for bi, (x, _y) in enumerate(dl, 1):
        x = x.to(device, non_blocking=False)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
        probs_all.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        if bi % 20 == 0:
            print(f"[predict] {bi}/{len(dl)}", flush=True)
    probs = np.concatenate(probs_all, 0)
    print(f"[predict] 概率矩阵 {probs.shape}")

    if args.prior != "none":
        # ⚠️⚠️⚠️ **2026-10-07 加护栏**：本分支**与2026-10-06实测到的重大事故同构**。
        #   做法是 probs = probs / q_tr再归一化 ⇒ 等价于 logit 加 **−log(训练集先验)**。
        #   而训练集先验 max/min = **49.6×**，测试集却是**均衡**（每类约 50 张）。
        #   ⇒ 方向恰好相反，会系统性偏向冷门类。
        #   实测（今天）：在37,444 张上施加同款偏置（系数 1.0）会改写
        #     **4.25%（1,592 张）**的预测（系数 1.5 ⇒ 6.52% / 2,440 张）。
        #   ⇒ 若对方队（他们用 −1.5·log(prior)）因此丢了约 9pp（76.09→ 67），
        #   **本开关同样危险**。
        #⇒ 正确做法是**均衡先验**（`--target uniform`，见 calib_sinkhorn_apply.py），
        #   使用训练集先验。
        #⇒ 默认 none；本方案未启用先验修正。
        q_tr = np.load(C.CACHE_DIR / "train_prior.npy")
        q = q_tr
        ratio = float(q.max() / max(q.min(), 1e-12))
        print("=" * 70)
        print(f"[predict] ⚠️⚠️⚠️ 已做先验修正（{args.prior}）"
              f"—— 训练集先验 max/min = {ratio:.1f}×")
        print(f"[predict] ★★ 但测试集是**均衡**的（每类约 50 张）"
              f"⇒ 该修正方向可能**相反**")
        print(f"[predict] ★★ 正确做法：`--target uniform`（见 calib_sinkhorn_apply.py）")
        print(f"[predict] ★★ 若你是为了复现历史包才开它，请忽略此警告；"
              f"否则请改用均衡先验")
        print("=" * 70)
        probs = probs / np.maximum(q, 1e-12)
        probs = probs / probs.sum(1, keepdims=True)

    class_map = ClassMap({k: int(v) for k, v in meta["class_map"].items()})
    class_map.confidence = meta.get("class_map_confidence", "unknown")
    class_map.source = meta.get("class_map_source", "?")
    class_ids = np.array([class_map[n] for n in names],
                         dtype=np.int64)[probs.argmax(1)]
    write_submission(te["img_name"].tolist(), class_ids,
                     out_dir=args.out_dir, class_map=class_map, class_names=names,
                     allow_unverified=args.allow_unverified_map)
    return 0



def _soft_targets(soft_lut, idx, y, K: int, device):
    """★ 2026-10-06：按 batch 索引组装软目标 (B,K)。

    规则（用户 2026-10-06 规格）：
      · 命中冲突组的样本 ⇒ 该组**不同标签均分**（`u[c] = 1/|Y|`）
      · 未命中的样本   ⇒ 保持 one-hot 硬标签（`u = onehot(y)`）
    ⇒ 因此 `compute_loss_soft` 对普通样本**逐位等价于**原 `compute_loss`
      （已由 `noise.soft_selfcheck()` 断言）。

    ★ 用稠密 (B,K) 而非稀疏：batch=96、K=750 ⇒ 288KB/批，可忽略；
      换来的是实现简单、不易错（稀疏索引更容易写错，这是本项目的教训）。
    """
    import torch as _t
    B = int(y.shape[0])
    u = _t.zeros(B, K, dtype=_t.float32, device=device)
    u[range(B), y] = 1.0
    if idx is None:
        return u
    _idx = idx.detach().cpu().numpy()
    for _i in range(B):
        _e = soft_lut[_idx[_i]]
        if _e is None:
            continue
        _cls, _pr = _e
        u[_i] = 0.0
        u[_i, _t.as_tensor(_cls, dtype=_t.long, device=device)] = \
            _t.as_tensor(_pr, dtype=_t.float32, device=device)
    return u

def main() -> int:
    ap = argparse.ArgumentParser(description="CLIP + LoRA 端到端微调")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA 学习率；头为 head_lr_mult 倍")
    ap.add_argument("--head-lr-mult", type=float, default=5.0,
                    help="分类头/logit_scale 相对 LoRA 的学习率倍数。"
                         "调大会让头快速起步，但过大（配合小数据集）会退化成"
                         "「全预测同一类」——实测 10 倍 + 512 张已足以练塌。")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪范数")
    ap.add_argument("--r", type=int, default=16, help="LoRA 秩")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    # ------------------------------------------------------- ★ VPT（2026-09-27 新增）
    # Visual Prompt Tuning：往视觉塔输入序列里插 N 个**从零训练**的 prompt token。
    # 与 LoRA/解冻层的本质区别 —— 前者改"输入条件"，后者改"权重"，二者正交。
    # 动机： 已确认旋钮级改动（解冻层数/秩/目标/分辨率）的 flip 精度
    # 全在 46~52%，等于抛硬币 ⇒ 追分必须走**机制级**改动。
    ap.add_argument("--region-attn", type=int, default=0,
                    help="★ 区域自注意力适配器（0=关闭；先验 +0.30pp）")
    ap.add_argument("--region-heads", type=int, default=12,
                    help="区域注意力头数（ViT-B/32 宽度 768 ⇒ 须能被heads 整除）")
    ap.add_argument("--region-pool", type=int, default=4,
                    help="区域池化核：把 patch 网格压成 1/pool 边长")
    # ★★★ 2026-10-07 SAM（Sharpness-Aware Minimization）—— 本项目首次引入
    #   `min_w max_{‖ε‖≤ρ} L(w+ε)`：求"平坦极小值" ⇒ 抗输入扰动 ⇒ **跨域泛化**。
    #   动机：我们的差距主体是"训练-测试落差"（冠军 2.15pp、region_attn 5.27pp），
    #   而 SAM **不加结构性参数**（与 −3.04pp 的 region_attn 不同类）⇒ 风险低。
    #   ★ 默认 0 = **完全关闭**（不进任何新分支）⇒ 现有复现路径零影响。
    #   ★ rho 标准取 0.05；开启后**自动关 AMP**（避免 AMP×SAM 的已知坑）。
    ap.add_argument("--sam-rho", type=float, default=0.0,
                    help="SAM 的邻域半径 ρ（0=关闭，标准值 0.05）。"
                         "开启后每步两次前向 ⇒ 训练时间约 ×1.7")
    # ★★★ 2026-10-09 SAM 变体：asam = 尺度自适应（Kwon et al. ICML 2021）。
    #   动机：我们参数组尺度差异极大（LoRA A~0.02 / LoRA B 零初始化 / 头 / logit_scale~4.6），
    #   原版 SAM 一视同仁的 ε 对小尺度参数相对过强、对 logit_scale 相对过弱。
    #   默认 sam ⇒ 与已上榜的 76.8107 逐字一致（零影响）。
    ap.add_argument("--sam-variant", default="sam", choices=["sam", "asam"],
                    help="SAM 变体：sam=原版（ε=ρ·g/‖g‖）；asam=尺度自适应（ε=ρ·w²⊙g/‖|w|⊙g‖）")
    # ★★★ 2026-10-08 SWA / Lookahead（与 SAM 同族，优化器层扩展）
    #   默认全关 ⇒ 冠军复现路径零影响。两者都只改优化过程，**合规**。
    #   - SWA：训练末期对 raw 权重滑动平均，平均权重当提交权重（跨域更稳）。
    #   - Lookahead：快权重探索、慢权重收敛（k=5, α=0.5 文献标准值）；
    #     包在 AdamW 外，且能与 SAM 兼容（SAM 的 second_step 调 base.step()）。
    ap.add_argument("--swa", action="store_true",
                    help="启用 SWA（从 --swa-start 起平均权重）。提交权重改用 SWA 平均。")
    ap.add_argument("--swa-start", type=int, default=5,
                    help="SWA 开始平均的 epoch（默认 5；后半程才平均更合理）。")
    ap.add_argument("--lookahead", action="store_true",
                    help="启用 Lookahead 优化器（包在 AdamW 外；与 SAM 兼容）。")
    ap.add_argument("--la-steps", type=int, default=5,
                    help="Lookahead 的 k：每 k 个 inner step 做一次 slow↔fast 同步。")
    ap.add_argument("--la-alpha", type=float, default=0.5,
                    help="Lookahead 的 α：slow 向 fast 插值系数（0.5=各半）。")
    ap.add_argument("--region-lr-mult", type=float, default=1.0,
                    help="★ 区域适配器学习率倍数（默认 1.0 = 与 LoRA 同档）。"
                         "保持 1.0 才能让「唯一变量 = 有没有这个模块」最干净；"
                         "要调先想清它会引入第二个变量")
    ap.add_argument("--region-gate", type=float, default=-2.2,
                    help="门控初值（sigmoid(-2.2)≈0.10，**不可设 0**，否则注意力无梯度）")
    ap.add_argument("--vpt", type=int, default=0,
                    help="VPT prompt token 个数（0=关闭）。插在 embeddings 输出的 "
                         "CLS 之后、patch 之前，序列变为 1+N+P。"
                         "代码名是 prompt tuning 而非 prompt engineering —— "
                         "它是可训练参数，不是文本模板。")
    ap.add_argument("--vpt-std", type=float, default=0.02,
                    help="prompt 初始化标准差。⚠️ prompt 会过 pre_layrnorm，"
                         "模长被归一化、只有方向有意义 ⇒ 调小 init %%不会%%"
                         "让起点趋近原模型（VPT 没有 LoRA 那种零初始化等价性）。")
    ap.add_argument("--vpt-lr-mult", type=float, default=1.0,
                    help="prompt 相对 --lr 的学习率倍数。prompt 从零训练，"
                         "常给与 LoRA 同档或更高的 lr；骨干则用小倍数。")
    # --------------------------------------------------------- 正则配方（2026-09-23 新增）
    # ★ 为什么是"配方"而不是一个个单变量：
    #   本项目此前把 288 / 长训 / 强增强 / dropout 全部**单独**试过，
    #   每一项的结论都是"收益≈0 或更差"，于是逐项关闭。
    #   但 `ep4 见顶回落` 这个现象本身就说明**是过拟合**（train 继续降、
    #   val 掉头），而"训更久"在欠正则的条件下必然如此 ——
    #   逐项单独测，就把「需要配套才能兑现的项」系统性判死了。
    #   ⇒ mixup / EMA / dropout 属于同一族（抑制对噪声样本的记忆），
    #     必须**打包**评测，不能拆开。
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="mixup 的 Beta(a,a) 参数；0=关闭。建议 0.2")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="权重 EMA 衰减；0=关闭。建议 0.999。"
                         "只跟踪可训练参数（约 10MB fp32），不复制整个模型")
    ap.add_argument("--ema-start", type=int, default=1,
                    help="从第几轮起累积 EMA（早期权重噪声大，可设 2）")
    ap.add_argument("--ema-off", action="store_true",
                    help="开了 EMA 但只用它做诊断对比、不用它选优/出包")
    # ------------------------------- 噪声样本筛选（课程式，2026-09-23 新增）
    # ★ 这是**唯一还没做过的一族**。官方细则（十、解题思路）点名方向为
    #   「数据过滤、样本选择、置信度建模、课程学习或重标注机制」，并在
    #   「五、算法设计要求 6」写明「如需引入数据清洗环节，**建议采用自动清洗
    #   或筛选策略**」。而本项目至今：损失函数一族（SCE/GCE/forward/CL 降权）
    #   已扫透，"数据选择"一族几乎为零。
    # ★ 之所以现在才做：`probe_dup_clusters.py` 全量实测证明近重复簇只覆盖 5.2%
    #   样本（且 94% 是 size=2 的平票簇，无法多数票纠标）⇒ **33.8% 的噪声绝大
    #   部分是"图像唯一、单纯标错"**，没有视觉重复可依据，只剩"置信度/小损失"
    #   这一条路可走。这不是换着花样试，而是把唯一剩下的方向补上。
    ap.add_argument("--soft-conflicts", type=str, default=None,
                    help="2026-10-06：冲突组软目标清单（build_soft_conflicts.py 产出）。"
                         "%%默认 None ⇒ 训练路径逐字不变%%（保 75.0641 复现）。"
                         "启用后：冲突组总权重=1（组内每张 1/m）+ 组内不同标签均分为软目标。")
    ap.add_argument("--sel-frac", type=float, default=0.0,
                    help="每轮筛掉的『类内损失最高』比例（0=关闭）。"
                         "建议 0.10~0.20；过大易触发确认偏差")
    ap.add_argument("--sel-start", type=int, default=2,
                    help="从第几轮起开始筛。前几轮模型还没学到东西，"
                         "此时的 loss 排序几乎只是『难易』而非『噪声』")
    ap.add_argument("--sel-min-keep", type=int, default=10,
                    help="★ 每类硬地板：类内样本数 ≤ 此值就一张都不筛。"
                         "防小类被清空（官方测试集是均衡的，删类直接扣分）")
    ap.add_argument("--targets", default="q,k,v,out")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="0 = 退化为纯冻结骨干基线（用于对照）")
    # ------------------------------------------------------------------
    # ★★ 骨干解冻（2026-09-23 新增）—— 本项目最大的一个**未探容量轴**。
    #   在此之前所有 LoRA 实验都只训 1.755% 参数（2.65M / 151M），
    #   骨干永远冻结。而本地留出集 → 榜分之间横着一个 **11.5 个点的 gap**
    #   （本地 mlp 全体 0.7308 vs 榜分 61.58），且已实测排除两个候选解释：
    #     · 画质/压缩：`quality_drop.json` 只值 1.96 分，占 gap 的 17%
    #     · TTA 视图数：V23 已比 V15 差（视图扩容到顶）
    #     · 模型侧超参：lr 两向都负、加长训练在诚实指标上负
    #   ⇒ 剩下的主要解释是**容量/拟合能力不足**，而这一轴从未被触碰。
    # ------------------------------------------------------------------
    ap.add_argument("--unfreeze-last", type=int, default=0,
                    help="解冻视觉塔%%最后 N 个 encoder 层%%的全部参数"
                         "（含 attn/MLP/LayerNorm，与 LoRA 叠加）。默认 0 = 维持历史行为。"
                         "⚠️ 解冻后 lr 必须小：见 --backbone-lr-mult。")
    ap.add_argument("--unfreeze-all", action="store_true",
                    help="解冻整个视觉塔（等价 --unfreeze-last 12）。最激进的一档。")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="骨干学习率 = --lr × 本倍数（默认 0.1）。"
                         "★ 预训练权重不宜大改：LoRA 的 lr 是给「从零初始化的小矩阵」用的，"
                         "直接套到 86M 预训练参数上必然把 CLIP 表征打散。")
    # ★★★ 2026-10-09 LLRD（逐层学习率衰减，ViT 微调标准配方；BEiT/MAE 用 0.65~0.75）
    #   lr_k = lr×backbone_lr_mult × decay^(n-1-k)，k=0 为最早解冻层 ⇒ 顶层保持、底层衰减。
    #   动机：直接从"精调 CLIP 先验"的官方要求出发；与平铺单档的唯一变量＝lr 分布。
    #   默认 0.0 = 关闭（零影响）。
    ap.add_argument("--llrd-decay", type=float, default=0.0,
                    help="逐层学习率衰减系数（0=关闭=平铺单档）。建议 0.65~0.75；"
                         "顶层（离头最近）保持 backbone lr，每往下一层乘一次 decay。")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="梯度累积步数（默认 1=关闭）。解冻骨干后显存吃紧时，"
                         "用 --batch 32 --grad-accum 3 换等效 batch 96 的梯度质量。")
    # ★★★ 2026-10-10 教师蒸馏（KD）软目标：官方点名"结合教师模型、自训练"方向。
    #   接口：--kd-probs <目录>，目录内含
    #     · probs_train.npy : float16 (N,750)，教师对**训练集**的软预测（可多视图平均）
    #     · paths_train.json: 与行对齐的 basename 列表（**按 basename 对齐**，与 --pseudo/--soft-conflicts 同规则）
    #   混成 target = (1-λ)·one_hot(y) + λ·p_teacher，取前 topk 稀疏存进 soft_lut；
    #   ★ 冲突组优先（命中冲突组的样本保留其软目标，不动冠军机制）。
    #   默认 None ⇒ 上面各行逐字不变。
    ap.add_argument("--kd-probs", default=None,
                    help="教师蒸馏软目标目录（含 probs_train.npy + paths_train.json）；默认关闭")
    ap.add_argument("--kd-lambda", type=float, default=0.3,
                    help="教师软目标混合系数 λ（target=(1-λ)y+λ·p_teacher），默认 0.3")
    ap.add_argument("--kd-topk", type=int, default=32,
                    help="每个样本保留的软目标候选类数（稀疏化省内存），默认 32")
    ap.add_argument("--lora-last-n", type=int, default=0,
                    help="LoRA 只注入**最后 N 个 encoder 层**（0=全层，与历次臂一致）。"
                         "★ 动机：注入范围（全 12 层）从未变过；官方合规核对文档点名"
                         "'同样容量放到不同层'＝魔改第一枪。出包侧会按 ckpt 的 args 自动重放。")
    ap.add_argument("--simmax-rew", default=None,
                    help="sim_max 带重加权规格，如 'alpha=1.0,lo=0.5,hi=2.0'（默认关闭）。"
                         "用冻结 CLIP 特征算每样本 sim_max（到训练池最大余弦、排除自身），"
                         "越不典型权重越高；权重只由训练集自身导出（不碰测试集）。")
    ap.add_argument("--degrade", default=None,
                    help="训练期画质退化增强的规格，默认 None（关闭）。"
                         "形如 'p=0.5,max_side=500,q=40-90'（各段可省略）。"
                         "★ 动机有实测依据：官方测试集被压到长边 500px 并重压缩"
                         "（体积中位 38KB vs 训练 85KB），而训练侧原先无画质增强；"
                         "实测 LoRA 在『压到 500 + JPEG q70』上掉 1.96 分，"
                         "且『长边统一到 500』本身不值钱（CLIP 裁剪框不变），"
                         "代价全在%%重压缩%%上。")
    ap.add_argument("--aug", default="plain", choices=["plain", "strong"],
                    help="训练期增强配方。plain=历史行为（仅 RandomResizedCrop"
                         "scale=0.7~1.0 + hflip）；strong=在上面的基础上再叠"
                         "%%只动成像不动几何%%的轻度增强（高斯模糊 p=0.30、"
                         "ColorJitter(0.3,0.3,0.2,0.02)、灰度 p=0.02）。"
                         "★ 依据：官方测试图被压到长边 500 且重压缩（中位 38KB "
                         "vs 训练 85KB），而 quality_drop.json 实测把 val 压到"
                         "同档会掉 1.96 分 ⇒ 模型对成像退化没有免疫力。"
                         "⚠️ 几何一律不动，否则 TTA 的 rc0..rc15「训练分布内视图」"
                         "语义作废。")
    # ------------------------------------------------- H2 随机尺寸归一化（2026-10-04）
    ap.add_argument("--size-rand", default=None, dest="size_rand",
                    help="训练期%%随机尺寸归一化%%的规格，默认 None（关闭）。"
                         "形如 'p=0.5,levels=500-640-800'（各段可省略）。"
                         "★ 动机（全量实测 2026-10-04）：官方对测试集做了统一的长边"
                         "归一化 —— 测试集长边==500 占 %%99.4%%%%、>500 占 %%0.0%%%%、"
                         "max 恰为 500；而训练集长边中位 640、>500 占 %%66.59%%%%、"
                         "max 8688 ⇒ %%训练侧从未做过这件事%%。"
                         "★★ 但%%不能「统一缩到 500」%%：那是 q75pack 的同构做法，"
                         "已被榜分否证（q75pack 全部固定 q75 %%−1.88pp%%，"
                         "而 deg 随机 q70~80 %%+0.26pp%%）"
                         "⇒ 规律是「域对齐应%%扩宽覆盖%%，而非收窄到一个点」。"
                         "故本选项在%%多个档位间随机%%（默认 500/640/800），"
                         "把测试集的 500 变成区间内的一个点，而不是要外推的边界值。"
                         "⚠️ %%只缩不放%%：本就小于目标档位的图保持原样，"
                         "避免插值糊图制造坏样本。"
                         "⚠️⚠️ %%绝对不动 RandomResizedCrop 的 scale%%：那被 TTA 的"
                         "rc0..rc15 视图当作「训练分布」引用，一改则 16 个视图全变"
                         "分布外、历史 TTA 结论全部作废。本层作用在 RRC %%之前%%，"
                         "改变的是「进入 RRC 的图有多大」，RRC 语义逐字保持。")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--val-per-class", type=int, default=8,
                    help="验证集每类至少留几张（避免小规模冒烟时验证集过小）")
    ap.add_argument("--track-lowsim", action="store_true",
                    help="★ 每轮额外记录「低相似 ≤0.90」子集的两列（A 选优 / B 报数）。"
                         "默认关（保持与历史基线一致）。用途：检查"
                         "「按全体 val 选优」是不是在选到%%记忆效应%%——"
                         "本地留出集有 18.97%% 近重复图，靠记忆也能涨。")
    ap.add_argument("--select-by", default="full", choices=["full", "lowsim_a"],
                    help="选优口径：full=全体 val（历史行为）；lowsim_a=低相似 A 半"
                         "（需同时 --track-lowsim）。⚠️ 用 lowsim_a 时"
                         "%%报数必须看 B 半%%，否则就是「在同一份数据上既选又评」。")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="每轮另存一份只含权重的快照（`%%_epNN.pt`）。"
                         "默认每轮覆盖同一个 ckpt ⇒ 训完只剩末轮，"
                         "想回头比较不同 epoch 就只能重训。约 16MB/份。")
    # ⚠️ 2026-10-06 修：`--help` 在这一项上崩（TypeError: * wants int）。
    #   原因：argparse 会对 help 字符串做 `%` 展开，而 help 里的字面 `*` 被当成格式符。
    #   修法：字面 `*` 写成 `%%`（argparse 随后还原成 `*`）。
    #   ★ 这也是"跑新分支前先 --help"能抓到的一类缺陷：
    #     它不报错地训练，但**评委想看参数说明时会直接崩**。

    ap.add_argument("--seed", type=int, default=None,
                    help="覆写全局随机种子（默认 C.SEED=历史行为）。"
                         "设不同值得到真正不同初始化/增强采样的种子重跑。")
    ap.add_argument("--pseudo", default="",
                    help="伪标签回收清单 json（键：paths/pseudo_labels，按路径匹配）。"
                         "训练时把清单内样本的标签替换为伪标签；"
                         "三道守卫见 train() 内注释；val 内样本直接中止。")
    ap.add_argument("--limit", type=int, default=0, help="分层抽样上限（冒烟用）")
    ap.add_argument("--fold", type=int, default=-1,
                    help="K 折 OOF：当前折下标（0..k-1），-1=关闭。设 0..k-1 时"
                         "把该折作为 holdout 排除出训练集，并落盘 holdout 清单供"
                         " scripts/oof_fold_label.py 打标。见 train() 内注释。")
    ap.add_argument("--k", type=int, default=5, help="K 折 OOF 的折数（配合 --fold）")
    ap.add_argument("--fold-seed", type=int, default=20260926,
                    help="K 折切分的随机种子（保证各折可复现、互不重叠）")
    ap.add_argument("--preproc", action="store_true",
                    help="★ 训练读「短边 320 预处理缓存」而非原图，轮速约 3 倍。"
                         "代价是改了训练分布（缓存图更锐）⇒ %%绝对分数不可与"
                         "未启用时横比%%，只用于扫参排名；扫完须在真管线上复核。"
                         "缓存由 scripts/build_preproc_cache.py 生成。")
    # ★★ 这段 help 在 2026-09-22 被实测推翻并改写，原文是错的。
    #  原文据 `bench_io_real.py`（自造 4096 张图、只跑 16 步、图全在 OS cache 里）
    #  写下「workers=0 最快 484 img/s、单轮 5.1 min」。实测不可能：单张
    #  解码 11 ms + 变换 10 ms = 23 ms，96 张就要 2.2 s/步，而它报 0.198 s/步。
    #  `calibrate_pipeline.py` 用**真图真 Dataset** 重测（batch 96 / 45 步）：
    #      workers=0 → 2472~2664 ms/步（单轮 64~69 min）
    #      workers=4 →  614 ms/步（15.9 min）
    #      workers=6 →  417 ms/步（10.8 min）
    #      workers=8 →  400 ms/步（10.3 min）   ← 默认
    #      workers=8 + OMP_NUM_THREADS=1 → 364 ms/步（9.4 min）
    #      workers=16 → 860 ms/步（22.2 min，过度订阅反而更慢）
    #  ⇒ 瓶颈是 **CPU 解码+几何变换**，不是磁盘（暖读 11.2 ms vs 冷读 13.1 ms）。
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader worker 数，默认 8。"
                         "★ 见上方注释：worker=0 会慢 6 倍（单轮 68 min vs 10 min），"
                         "worker=16 过度订阅也慢一倍。配 OMP_NUM_THREADS=1 再快 9%%。")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt", default=str(C.CACHE_DIR / "lora_ckpt.pt"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--kind", default="sce",
                    help="损失类型：ce|gce|forward_ce|forward_gce"
                         "，以及 sce|forward_sce（= %%标准 SCE%%，Wang et al. 2019，"
                         "RCE = -Σ p·log y，%%新默认%%）与 "
                         "sce_legacy|forward_sce_legacy（= %%历史实现%%，"
                         "其 RCE 被写成预测熵 -Σ p·log p %%不含标签%%，不是 SCE）。"
                         "★ 要复现 2026-10-05 之前已交并得分的包"
                         "（如榜分 74.3 的 d_r448mlp1024）%%必须用 --kind sce_legacy%%。"
                         "见 outputs/SCE实现错误_复核与后果_20261005.md")
    # ★ SCE 的 alpha/beta 提成命令行参数。
    # 理由：这两个值原先只能从 config.py 读，而"端到端该用多大 beta"与
    # "只训线性头该用多大 beta"**并不是一回事**（见下方 guard 说明），
    # 做对照时必须能在不改源码、不串味的前提下扫参。
    ap.add_argument("--sce-alpha", type=float, default=C.SCE_ALPHA,
                    help=f"SCE 里 CE 的权重（默认 {C.SCE_ALPHA}）")
    ap.add_argument("--sce-beta", type=float, default=C.SCE_BETA,
                    help=f"SCE 里 RCE 的权重（默认 {C.SCE_BETA}）。"
                         f"⚠️ RCE 不含标签、只把预测推向均匀，端到端微调时"
                         f"过大（尤其 beta>>alpha）会锁死准确率在 1/K")
    ap.add_argument("--use-cl", action="store_true", default=True)
    ap.add_argument("--no-cl", dest="use_cl", action="store_false")
    # ★★ 2026-09-28 新增：margin 类损失（CosFace 的 logit 形式）
    ap.add_argument("--margin", type=float, default=0.0,
                    help="对%%正确类%%的 logit 减 m×logit_scale（余弦空间单位，"
                         "典型 0.1~0.5）。0=关闭（默认，行为与历史逐字一致）。"
                         "只作用于训练损失，推理路径不变。依据：本项目已证明错因是"
                         "细粒度类间重叠（错分样本 rank(y) 中位 5.0），margin 直击该间隔。")
    # ★★ 2026-09-28 新增：表征漂移约束（官方点名 5 处的方向）
    ap.add_argument("--drift-lambda", type=float, default=0.0,
                    help="表征漂移约束权重：把当前特征拉向「未微调的初始模型」特征。"
                         "0=关闭（默认）。>0 时 deepcopy 一份冻结的初始模型做教师，"
                         "每步多一次无梯度前向（训练时间约 +40~50%%）。"
                         "依据：赛题五-2/五-3/十一-一-2/十-二/十二-6 反复点名"
                         "「表征约束」「保留CLIP预训练先验」「灾难性遗忘抑制」。")
    ap.add_argument("--t-strength", type=float, default=C.T_STRENGTH)
    ap.add_argument("--prior", default="none", choices=["none", "train_prior"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--predict-only", default=None,
                    help="只做推理：传入 ckpt 路径")
    ap.add_argument("--allow-unverified-map", action="store_true")
    a = ap.parse_args()

    print(f"[cfg] device={a.device} cuda={torch.cuda.is_available()} "
          f"root={C.ROOT}")
    if a.predict_only:
        return predict(a)
    return train(a)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
