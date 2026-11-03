"""LoRA 路线的**图像级多视图 TTA** 离线评估。

为什么不能复用 `tta_eval.py`
----------------------------
`tta_eval.py` 是**冻结头**的 TTA：它翻转的是**已经抽好的特征**
（`feat_train.npy` 那套），线性头在特征上平均。这条路径对 LoRA **不成立** ——
LoRA 改了骨干，特征得重新前向；而且「翻转特征」不是「翻转图像的等价物」，
两者只有在骨干严格等变时才相同，LoRA 微调后并不等变。
⇒ 所以 LoRA 的 TTA 必须在**图像层面**做（同一张图多个预处理视图，前向多次，
在**概率**上平均）。

为什么视图参数化在「短边缩放」上（★ 本脚本最有价值的一点）
----------------------------------------------------------
训练用 `RandomResizedCrop(224, scale=(0.7,1.0))`；评估用
`Resize(224) + CenterCrop(224)`（短边缩到 224，再中心裁 224）。
对一张 4:3 的图，后者等于**只看中心约 75% 的面积**。

关键推论：**喂给 224 模型的裁剪块越小 ⇒ 每个 32×32 patch 覆盖的原图区域越小
⇒ 等效分辨率越高。** 而细粒度识别恰恰吃分辨率。
⇒ 「裁得更小」是一个**纯预处理**的提分辨率手段：**不改模型、不改任何权重**
（不需要插值位置编码，不触碰"只用官方权重"这条规则），
只是在推理时换裁剪范围。这正是本脚本要量的东西。

⚠️ 但它有边界：裁剪范围离训练分布 `area∈[0.7,1.0]` 越远，越像在考没见过的题。
   所以视图从 224（=训练分布内）到 288、到更小，**要逐个量，不能想当然**。

评估口径（务必连这段一起读）
----------------------------
* 用 **LoRA 自己那份按类分层的留出集**（`lora_val_mask`，7537 张），
  与 `compare_lora_vs_frozen.py` 同源 ⇒ **本脚本的 no-TTA 那一行必须与
  该脚本报出的 LoRA 行对上**。对不上就说明切分复刻错了，别往下读。
* 取舍看「**低相似 ≤0.90**」列（理由见 `probe_stratified.py` 与技能坑 21）：
  本地「全体」指标被近似重复图撑着，官方测试集是另来源、无近似重复。
* 这是**配对比较**（同一批样本、同一模型、只有预处理不同）⇒
  配对差值比绝对值可信，但**绝不能当榜分预测**。

用法::

    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/tta_lora_eval.py \
        --lora outputs/cache/lora_full750_best.pt

    # 只跑部分视图（省时间）
    ... --views base,flip,zoom288
"""
from __future__ import annotations

import os
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C            # noqa: E402
import numpy as np            # noqa: E402
import torch                  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# ★ 复刻切分的唯一真源：直接复用对照脚本里的函数，而不是在这里重写一遍。
#   重写的风险是"看起来一样但 rng 调用顺序差一步" ⇒ 切分不同 ⇒ 全表作废。
import compare_lora_vs_frozen as CLF  # noqa: E402
from data import build_train_manifest  # noqa: E402


# --------------------------------------------------------------- 视图定义
class _ViewSpec:
    """一个 TTA 视图。

    两种模式：
      · ``rc=False``：**确定性**视图。``shortest`` 非空表示"短边缩到 shortest
        再中心裁 224"；``full_square=True`` 表示"整图压成 224×224"。
      · ``rc=True``：**训练分布内的随机视图** —— 用与训练**逐字相同**的
        `RandomResizedCrop(224, scale=(0.7,1.0)) + RandomHorizontalFlip(0.5)`。
        ``seed_base`` 决定这一视图的随机序列。

    ★ 为什么要有 ``rc`` 这一路
    --------------------------
    `zoom288/336` 的等效面积占比只有 **0.45 / 0.33**，而训练分布是
    `RandomResizedCrop(scale=(0.7,1.0))` ⇒ **它们在训练分布之外**，
    等于"考没教过的裁剪范围"，收益不确定。
    而 ``rc`` 视图的裁剪分布与训练**完全一致**，只是换了几次随机抽样
    ⇒ 这是"只降方差、不引入分布偏移"的那一类 TTA，安全得多。
    两条路都留着，让数据决定。
    """

    def __init__(self, name: str, shortest: int | None = None,
                 full_square: bool = False, flip: bool = False,
                 rc: bool = False, seed_base: int = 0):
        self.name = name
        self.shortest = shortest
        self.full_square = full_square
        self.flip = flip
        self.rc = rc
        self.seed_base = seed_base

    @property
    def area(self) -> str:
        if self.full_square:
            return "1.00"
        if self.rc:
            return "0.70~1.00（=训练分布）"
        return f"{(C.IMG_SIZE / int(self.shortest)) ** 2:.2f}"

    def __repr__(self) -> str:
        return f"_ViewSpec({self.name})"


class _HFlip:
    """确定性水平翻转。

    ⚠️ 必须写成**模块级类**、不能写成 lambda：
    Windows 下 DataLoader 用 spawn，`T.Compose([...])` 要能 pickle，
    lambda 直接报 `AttributeError: Can't pickle local object`。
    （本工程已在 `_QualityDegrade` 上踩过一次同类问题。）
    """

    def __call__(self, im):
        from torchvision.transforms import functional as TF
        return TF.hflip(im)


def build_view_tf(v: _ViewSpec):
    from torchvision import transforms as T
    from extract import CLIP_MEAN, CLIP_STD

    ops = []
    if v.rc:
        # ★ 与训练**逐字相同**的几何增强（见 finetune_lora._build_tf）。
        #   scale=(0.7,1.0)、flip p=0.5 —— 改一个数就不是"训练分布内的 TTA"了。
        ops.append(T.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0),
                                       interpolation=T.InterpolationMode.BICUBIC))
        ops.append(T.RandomHorizontalFlip(p=0.5))
    elif v.full_square:
        # 整图压成 224×224：内容最全，但长宽比被扭曲（CLIP 没见过这种输入）
        ops.append(T.Resize((C.IMG_SIZE, C.IMG_SIZE),
                            interpolation=T.InterpolationMode.BICUBIC))
    else:
        ops.append(T.Resize(int(v.shortest),
                            interpolation=T.InterpolationMode.BICUBIC))
        ops.append(T.CenterCrop(C.IMG_SIZE))
        if v.flip:
            ops.append(_HFlip())
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    return T.Compose(ops)


class _ViewDataset(Dataset):
    """按给定视图预处理图片。标签不参与，只借 `data.load_image` 的读图逻辑。

    ★ ``seed_base`` 非空时，**逐样本**重置全局 torch 随机种子
    （`seed = seed_base + i`）。这样随机裁剪/翻转完全由 (seed_base, 样本下标)
    决定 ⇒ **与 DataLoader worker 数无关**。
    踩过的需求来源：本项目已实测 `--workers 0` 与 `--workers 6` 的增强轨迹不同
    （随机数在 worker 进程里生成），导致两次训练的对照混入了"随机轨迹"这个变量。
    逐样本播种可以从根上消掉这个混杂。
    """

    def __init__(self, paths: list[str], tf, seed_base: int | None = None):
        # ★ `load_image` 在 data.py，不在 finetune_lora —— 别照抄错了源头，
        #   否则读出两套不同的解码行为（EXIF 旋转、灰度转 RGB 等），
        #   与生产推理就不是同一个预处理了。
        from data import load_image
        self.paths = paths
        self.tf = tf
        self.seed_base = seed_base
        self._load = load_image
        self.n_failed = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = self._load(self.paths[i])
        if im is None:
            self.n_failed += 1
            from PIL import Image
            im = Image.new("RGB", (C.IMG_SIZE, C.IMG_SIZE), (127, 127, 127))
        if self.seed_base is not None:
            torch.manual_seed(self.seed_base + i)
        return self.tf(im), 0


# --------------------------------------------------------------- 模型
def load_lora_model(ckpt_path: Path, K: int, device: str):
    """载入 LoRA 权重 —— 逐字对齐 `finetune_lora.predict` 的载入方式。"""
    import lora as LO
    import vpt as VPT
    from extract import load_clip
    from finetune_lora import LoRAClipClassifier

    ck = torch.load(ckpt_path, map_location=device)
    kind = ck.get("backbone_kind", "transformers")
    r = int(ck.get("r", 16))
    tr_args = ck.get("args", {})
    hidden = int(tr_args.get("hidden", 0))
    targets = tuple(str(tr_args.get("targets", "q,k,v,out")).split(","))
    print(f"[tta] {ckpt_path.name} best_ep={ck.get('best_ep')} "
          # ★ 把 targets 一起打出来。它是从 ckpt 的 `args` 里读的；
          #   万一哪天读不到，会**静默退回默认的 q,k,v,out** —— 那样注入的层
          #   与训练时不一致，权重加载要么报错、要么（更糟）只加载上一部分。
          #   打出来才能一眼核对。
          f"best_val_acc={ck.get('best_val_acc')} r={r} hidden={hidden} "
          f"targets={','.join(targets)}")

    # ★★ 分辨率守卫（2026-09-23）。
    #   `IMG_SIZE` 由环境变量 `AIC_IMG_SIZE` 驱动，**不在 ckpt 的 args 里** ——
    #   如果不校验，拿一个 288 训练的权重在默认 224 下评估/出包，
    #   会静默关掉位置编码插值、把 TTA 视图几何全按 224 算，
    #   结果整体作废而且**看不出原因**（本项目对 `targets` 静默退回踩过同类坑）。
    ck_img = ck.get("img_size")
    if ck_img is None:
        print("[tta] ⚠️ 该检查点未记录 img_size（老版本产物）⇒ 无法校验分辨率，"
              "按当前 C.IMG_SIZE 继续")
    elif int(ck_img) != int(C.IMG_SIZE):
        raise SystemExit(
            f"[tta] 🛑 分辨率不匹配：检查点训练于 IMG_SIZE={ck_img}，"
            f"当前 C.IMG_SIZE={C.IMG_SIZE}。\n"
            f"        修法：加环境变量重跑，即 `AIC_IMG_SIZE={ck_img} …`。\n"
            f"        不修的话位置编码插值与 TTA 视图几何都会按错的分辨率计算。")

    _, clip_model, _ = load_clip(device)
    clip_model = clip_model.float()
    if r > 0:
        LO.inject_lora(clip_model, r=r, alpha=tr_args.get("alpha"),
                       dropout=0.0, targets=targets,
                       last_n_layers=int(tr_args.get("lora_last_n", 0) or 0),
                       verbose=False)
    # ★★ VPT 重放（2026-09-27）。与分辨率的坑**完全同源**，而且更隐蔽：
    #   不重放注入 ⇒ ckpt 里的 prompt 权重没有落点 ⇒ load_state_dict(strict=False)
    #   只打一行 missing 警告，**预测照出** —— 但用的是随机初始化的 prompt，
    #   出包整体作废且看不出原因。⇒ 必须"按 ckpt 自带 args 重放 + VPT 键硬断言"。
    _vpt_n = int(tr_args.get("vpt", 0) or 0)
    if _vpt_n > 0:
        VPT.inject_vpt(clip_model, _vpt_n,
                       init_std=float(tr_args.get("vpt_std", 0.02) or 0.02),
                       verbose=False)
        print(f"[tta] ★ 已按训练配置重放 VPT 注入（n_prompt={_vpt_n}）")
    # ★★ 区域自注意力适配器（2026-10-07 引入）
    #   ★ 必须在 `load_clip` 之后、模型封装之前 —— 之前我插在 ck 解析处，
    #     那时 `clip_model` 还没赋值 ⇒ UnboundLocalError（10-07 实测）。
    #   作用：训练侧若开了，推理侧**必须重建并载入权重**，
    #     否则适配器是随机权重 ⇒ **静默失分**。
    # ★★★ 2026-10-07 修正（重要）：区域适配器的权重**就在 `ck["model"]` 里**，
    #   键名是 `clip.vision_model.region_adapter.*`（因为适配器是 `vision_model` 的子模块）。
    #   ⇒ **不需要单独的顶层字段**；只要**把适配器挂上**，
    #     后面那句 `model.load_state_dict(ck["model"], strict=False)` 会**自动载入**。
    #   ⚠️ 踩过的坑：我最初只查顶层 `ck["region_attn"]` 字段 ⇒ 自证失败、
    #     误判成"本臂等同 U 冠军"，**白跑了一次 1.9h 训练**（实际权重好端端在 ckpt 里）。
    #   ⇒ 现在改成**从 `model` 的键自动探测**，向后兼容两种 ckpt。
    _ck_model = ck.get("model", {}) or {}
    _reg_keys = [k for k in _ck_model if "region_adapter" in k]
    ck_region = int(ck.get("region_attn", 0) or 0) or (1 if _reg_keys else 0)
    if ck_region > 0:
        from region_attention import attach_region_adapter
        _heads = int(ck.get("region_heads", 12) or 12)
        _pool = int(ck.get("region_pool", 4) or 4)
        _gate = float(ck.get("region_gate", -2.2) or -2.2)
        attach_region_adapter(clip_model, heads=_heads, pool=_pool, gate_init=_gate)
        print(f"[tta] [region] 已挂适配器 heads={_heads} pool={_pool} gate={_gate}"
              f"（权重将由下面的 load_state_dict 自动载入，共 {len(_reg_keys)} 个键）")
        if not _reg_keys:
            print("[tta] ⚠️⚠️ 探测到要挂适配器，但 ckpt 里**没有** region_adapter 权重"
                  " ⇒ 适配器将是随机权重 ⇒ 结果无效！")

    model = LoRAClipClassifier(clip_model, kind, K, hidden=hidden).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # ★ 适配器权重是否真的载入（missing 里不应出现 region_adapter）
    if ck_region > 0:
        _reg_missing = [k for k in missing if "region_adapter" in k]
        if _reg_missing:
            print(f"[tta] ⚠️⚠️ region_adapter 有 {len(_reg_missing)} 个键未载入 "
                  f"⇒ 随机权重 ⇒ 结果无效！例：{_reg_missing[:3]}")
        else:
            print(f"[tta] ✅ region_adapter 权重已全部载入（{len(_reg_keys)} 个键）")
    if _vpt_n > 0:
        _bad = [k for k in list(missing) + list(unexpected) if VPT.VPT_TAG in k]
        if _bad:
            raise SystemExit(
                f"[tta] 🛑 VPT 权重未对齐（{_bad[:3]} …）—— 注入配置与训练时不一致，"
                f"评估/出包结果不可信，拒绝继续。")
    if missing or unexpected:
        print(f"[tta] ⚠️ 载入有差异：missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def infer(model, paths: list[str], tf, device: str, batch: int,
          workers: int, tag: str, seed_base: int | None = None,
          head_ckpt: Path | None = None) -> np.ndarray:
    """按 `tf` 跑一遍，返回概率矩阵。

    ``head_ckpt`` 给出时，改用**外部的头**（`train_head` 存的 `RobustHead`）
    接在 LoRA 骨干后面 —— 这样 TTA 也能叠在「LoRA 特征 + 充分收敛的头」
    那条路线（`eval_lora_features_head.py`）上。
    """
    ds = _ViewDataset(paths, tf, seed_base=seed_base)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=False, prefetch_factor=2 if workers > 0 else None)

    if head_ckpt is not None:
        # 走外部头：先把每个视图的特征收齐，再交给 train_head 的推理路径
        import train_head as TH
        feats = []
        with torch.no_grad():
            for bi, (x, _y) in enumerate(dl, 1):
                x = x.to(device, non_blocking=False)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    f = model.encode(x)
                feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
                if bi % 10 == 0:
                    print(f"[tta/{tag}] {bi}/{len(dl)}", flush=True)
        X = torch.cat(feats, 0)
        hm = TH.load_final(head_ckpt, device=device)
        p = TH.accuracy(hm, X.to(device), None, prior_q=None).cpu().numpy()
        del hm, X, feats
        if device == "cuda":
            torch.cuda.empty_cache()
        if ds.n_failed:
            print(f"[tta/{tag}] ⚠️ {ds.n_failed} 张读图失败，已用灰图顶替")
        return p

    out = []
    # ★ 2026-10-06：AMP 推理开关。
    #   实测：autocast(fp16) 下 **85.3% 的样本 max 概率 ≥ 0.9999**，
    #   且**每一个**非零元素都是 float16 可表示的数（比例 1.000000）
    #   ⇒ 量化发生在 **GPU 前向那一刻**，不是保存那一刻。
    #   ⇒ 后果：τ 阈值筛选**形同虚设**（τ=0.9 仍留下 97.65%），
    #     且margin/置信度分层的读数全部失真。
    #   ⇒ 需要真实置信度时用 --no-amp（会慢一些，但概率是全精度的）。
    _amp = (device == "cuda") and not _NO_AMP
    with torch.no_grad():
        for bi, (x, _y) in enumerate(dl, 1):
            x = x.to(device, non_blocking=False)
            with torch.amp.autocast("cuda", enabled=_amp):
                logits = model(x)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            if bi % 10 == 0:
                print(f"[tta/{tag}] {bi}/{len(dl)}", flush=True)
    p = np.concatenate(out, 0)
    if ds.n_failed:
        print(f"[tta/{tag}] ⚠️ {ds.n_failed} 张读图失败，已用灰图顶替")
    return p


def mcnemar(ok_a: np.ndarray, ok_b: np.ndarray) -> tuple[int, int, float]:
    """配对二项检验。返回 (a错b对, a对b错, 精确 p)。"""
    fixed = int((~ok_a & ok_b).sum())
    broke = int((ok_a & ~ok_b).sum())
    m = fixed + broke
    if m == 0:
        return 0, 0, 1.0
    p = sum(comb(m, k) for k in range(min(fixed, broke) + 1)) / 2 ** m * 2
    return fixed, broke, min(1.0, p)


# --------------------------------------------------------------- 主流程
#
# ★★ 2026-09-23：视图几何改为**由 C.IMG_SIZE 同比缩放**。
#
#   此前这里的 shortest 是写死的绝对像素（224/240/256/288/320/336），只在
#   IMG_SIZE=224 下自洽。一旦把 IMG_SIZE 提到 288，"shortest=224 再 CenterCrop
#   288"就变成不可能（小图裁大图，torchvision 会补灰边），整套视图语义全坏。
#
#   现在以 224 为基准换算：`_P(224)=IMG_SIZE`、`_P(240)=IMG_SIZE×240/224` …
#   ⇒ **IMG_SIZE=224 时数值与本改动前逐位相同**（向后兼容，已提交的三个包仍能复现），
#      IMG_SIZE=288 时整套视图等比放大，"等效面积占比"的含义保持不变。
_VIEW_BASE = 224


def _P(px: int) -> int:
    """把「按 224 设计的视图像素」换算到当前 IMG_SIZE。"""
    return int(round(px * C.IMG_SIZE / _VIEW_BASE))


# ★ 2026-10-06：全局 AMP 开关（默认 False ⇒ **保持原行为不变**）
#   用环境变量 AIC_NO_AMP=1 或 main() 的 --no-amp 打开。
_NO_AMP = os.environ.get("AIC_NO_AMP", "") not in ("", "0", "false", "False")

ALL_VIEWS = [
    # ---- 确定性视图（可复现，跨机器一致）
    _ViewSpec("base",   _P(224), False, False),   # = 当前生产评估预处理
    _ViewSpec("flip",   _P(224), False, True),
    # ★ 缩放到"短边 = n"的档位。等效面积占比 = (IMG_SIZE/n)^2。
    #   训练分布是 RandomResizedCrop(scale=(0.7,1.0))，面积**均匀**落在 [0.7,1.0]
    #   ⇒ 训练时见到的面积中位约 0.85，而 base 只有 **0.75**（4:3 图缩到 224×298
    #   再中心裁 224）⇒ base 本身就**偏小**。
    #   所以 240(0.87) / 256(0.77) **落在训练分布之内**，而 288(0.61)/320(0.49)/336(0.44)
    #   已经**在分布之外**。这两类要分开读，不能混成一句"提高分辨率"。
    _ViewSpec("zoom240", _P(240), False, False),  # 等效面积 0.87 ★ 训练分布内
    _ViewSpec("zoom256", _P(256), False, False),  # 等效面积 0.77 ★ 训练分布内
    _ViewSpec("zoom288", _P(288), False, False),  # 等效面积 0.61（分布外，更"放大"）
    _ViewSpec("zoom288f", _P(288), False, True),
    _ViewSpec("zoom320", _P(320), False, False),  # 等效面积 0.49（分布外更远）
    _ViewSpec("zoom336", _P(336), False, False),  # 等效面积 0.44（离训练分布最远）
    _ViewSpec("square", None, True, False),   # 整图压方形（内容最全，长宽比扭曲）
    # ---- 训练分布内的随机视图（含 hflip），逐样本播种 ⇒ 与 worker 数无关
    #   加更多抽样个数 = 把"随机裁剪的期望"估得更准，只降方差、不引分布偏移。
    _ViewSpec("rc0", rc=True, seed_base=110_000),
    _ViewSpec("rc1", rc=True, seed_base=120_000),
    _ViewSpec("rc2", rc=True, seed_base=130_000),
    _ViewSpec("rc3", rc=True, seed_base=140_000),
    _ViewSpec("rc4", rc=True, seed_base=150_000),
    _ViewSpec("rc5", rc=True, seed_base=160_000),
    _ViewSpec("rc6", rc=True, seed_base=170_000),
    _ViewSpec("rc7", rc=True, seed_base=180_000),
    # ★ rc8..rc15：把"训练同款随机裁剪"的期望估得更准。
    # ⚠️⚠️ **2026-10-07 更正**：原注释写「k=8 时实测仍未饱和」，
    #   **今天的实测不支持这句话**（U 冠军 15 视图，7,536 张 val）：
    #     rc 数 2→4 +0.0398(p=.75) / 4→6 +0.1194(p=.15) / 6→8 +0.0663(**p=.42**)
    #   ⇒ **边际收益已在噪声里**；按递减一半外推 8→16 约 **+0.03pp**，
    #     远低于 0.3pp 发车门槛 ⇒ **不值得为它花 16 分钟 GPU**。
    #   ⇒ 保留这些定义仅为复现旧包；**别照旧注释去扩视图**。
    #   这一路是**唯一**与训练分布逐字一致的随机视图 ⇒ 只降方差、不引分布偏移，
    #   属于"加多少都不会变坏"的那一类（除非降到过平滑）。
    #   ⚠️ 但边际递减很明显：k 4→8 涨 +0.0107（低相似列），再翻倍能涨多少要实测，
    #      而且**必须做配对检验**（Δ 很可能落在标准误 ~0.009 之内）。
    _ViewSpec("rc8",  rc=True, seed_base=190_000),
    _ViewSpec("rc9",  rc=True, seed_base=200_000),
    _ViewSpec("rc10", rc=True, seed_base=210_000),
    _ViewSpec("rc11", rc=True, seed_base=220_000),
    _ViewSpec("rc12", rc=True, seed_base=230_000),
    _ViewSpec("rc13", rc=True, seed_base=240_000),
    _ViewSpec("rc14", rc=True, seed_base=250_000),
    _ViewSpec("rc15", rc=True, seed_base=260_000),
]

# ★ 实测最优（2026-09-22，15 视图）
#   配对检验（相对 8 视图版）：Δ全体 +0.0044 / Δ低相似 +0.0049，
#   错→对 83 / 对→错 50，**McNemar p=0.0053 ⇒ 显著**。
#   且**低相似列涨得比全体更多** ⇒ 符合"真泛化"判据。
#   `rc` 个数递增到 k=8 仍单调上升（未饱和），后续可再扩。
DEFAULT_VIEWS = ("base,flip,rc0,rc1,rc2,rc3,rc4,rc5,rc6,rc7,"
                 "zoom240,zoom256,zoom288,zoom320,square")

# 上一轮的 8 视图版（保留下来用于**复现已提交的包**，别删）
V8_VIEWS = "base,flip,rc0,rc1,rc2,rc3,zoom288,square"

# 把 rc 扩到 16 个的候选组合（= 15 视图 + rc8..rc15，共 23 视图）。
# ★ 用途是**验证 rc 这一维到底饱和没有**，而不是直接拿去做提交：
#   如果它对 15 视图的增益配不上显著性，就不该为了 16 分钟的额外推理去换。
#   验证时**只需跑新增的 8 个视图**（`--views rc8,...,rc15 --save-probs ...`），
#   再和已有的 15 视图 npz 离线合并 —— 见 `analyze_tta_views.py` 的多 npz 支持。
V23_VIEWS = DEFAULT_VIEWS + "," + ",".join(f"rc{i}" for i in range(8, 16))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="LoRA 图像级多视图 TTA 评估")
    ap.add_argument("--lora", default=str(C.CACHE_DIR / "lora_full750_best.pt"))
    ap.add_argument("--views", default=DEFAULT_VIEWS,
                    help="逗号分隔的视图名；可用："
                         + ",".join(v.name for v in ALL_VIEWS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6,
                    help="★ 默认 6 —— 推理瓶颈在 CPU 解码，与训练相反，"
                         "这里开 worker 是有收益的（实测 4 偏保守；"
                         "★ 改这个数%%%%不影响结果%%%%：视图用逐样本播种，"
                         "见 _ViewDataset 的 seed_base 说明）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只评前 N 张（快速试跑用；>0 时结论仅供参考）")
    ap.add_argument("--head", default="",
                    help="改用%%%%外部的头%%%%（outputs/cache 下的 train_head 产物）"
                         "接在 LoRA 骨干后面，用于给「LoRA 特征 + 充分收敛的头」"
                         "那条路线做 TTA。留空则用检查点自带的头。")
    ap.add_argument("--save-probs", default="",
                    help="把%%%%逐视图%%%%概率存成 npz（float16）。"
                         "★ 动机：本轮实测「8 视图 vs 15 视图」的差值本身%%%%就在标准误内%%%%"
                         "（+0.0044 全体 / +0.0049 低相似，而 SE 是 0.0055 / ~0.009），"
                         "%%%%只有配对检验才判得动%%%%；而配对必须用同一批样本的逐样本预测。"
                         "存下来之后，任意子集组合、任意两两配对、留一法都可以离线重算，"
                         "不必为每个组合重跑 GPU。")
    ap.add_argument("--no-amp", action="store_true",
                    help="★关闭 GPU 自动混合精度推理（默认保持开启）"
                         "autocast(fp16) 会让 85%% 的样本概率饱和到 ≥0.9999，"
                         "使置信度/margin 类分析全部失真")
    a = ap.parse_args()
    global _NO_AMP
    if a.no_amp:
        _NO_AMP = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tta] device={device}")

    tr = build_train_manifest()
    K = int(tr.attrs.get("num_classes", 0))
    labels = tr["label"].to_numpy().astype(np.int64)
    paths = tr["path"].tolist()

    m = CLF.lora_val_mask(labels, K)
    # ★ 09-26 新增：**外部 val 名单覆盖**（默认关闭，只在 `AIC_VAL_NAMES=<文件>` 时生效）
    #   动机：本机算出的留出集与**云端训练时用的留出集不是同一批**（实测重叠仅 63.8%）。
    #   根因：`data.build_train_manifest()` 用 `d.iterdir()` 建行序（**不排序、不缓存**），
    #   而 `finetune_lora` 的切分是 `rng.choice(idx, size=want)` —— 抽的是**位置**，
    #   所以行序一变、成员就变（大小恒为 7537 不变，所以只比大小看不出来）。
    #   ⇒ 用云端的 val 名单在本机重评，就能一次分开两种解释：
    #      「是新代模型在另一个 val 上本来就不行」 vs 「是云端那次测量有问题」。
    #   名单格式：每行一个 basename（`#` 开头为注释行），由 `scripts/dump_val_names.py` 生成。
    _vn = os.environ.get("AIC_VAL_NAMES", "")
    if _vn:
        _want = [l.strip() for l in Path(_vn).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        _wset = set(_want)
        _bs = [str(p).replace("\\", "/").rstrip("/").split("/")[-1] for p in paths]
        m = np.array([b in _wset for b in _bs], dtype=bool)
        _hit = int(m.sum())
        print(f"[tta] ★ val 名单覆盖：{_vn} ⇒ 名单 {len(_wset)} 个 basename，"
              f"在本机 manifest 命中 {_hit} 张（未命中 {len(_wset) - _hit}）")
        if _hit != len(_wset):
            print(f"[tta] ⚠️ 有 {len(_wset) - _hit} 个名单项在本机 manifest 里找不到 "
                  f"⇒ 两边数据集合可能不一致，先查这个再解读下面的读数")
    # AIC_AUDIT_EXPECT_VAL：U 冠军（10-06）训练日志是 7536 张
    #（10-04 补了保底修）；旧臂（09-26）走修前逻辑是 7537。
    _EXPECT_VAL = int(os.environ.get("AIC_AUDIT_EXPECT_VAL", "7537"))
    if int(m.sum()) != _EXPECT_VAL:
        print(f"[tta] X 复刻出的留出集是 {int(m.sum())} 张，"
              f"期望 {_EXPECT_VAL} 张（设AIC_AUDIT_EXPECT_VAL 可改）"
              f" => 切分复刻错了，结论不可用。")
        return 1
    va_idx = np.where(m)[0]
    va_paths = [p for p, keep in zip(paths, m) if keep]
    yva = labels[m]
    print(f"[tta] 留出集 {len(yva)} 张 / 覆盖 {len(np.unique(yva))}/{K} 类")

    # 低相似掩码（标尺统一用冻结 CLIP 特征）
    Xtr = np.load(C.CACHE_DIR / "feat_train.npy")
    low, ms = CLF.low_sim_mask(Xtr, labels, va_idx, np.arange(len(labels))[~m])
    print(f"[tta] 低相似（同类最近邻 ≤0.90）：{int(low.sum())} 张"
          f"（中位余弦 {float(np.median(ms)):.4f}）")

    # ★ 09-26 新增：**软域尺子**（默认关闭，只在 AIC_VAL_SOFT=1 时生效）
    #   动机：训练期 `--preproc` 会把 **train 与 val 的路径都改指向 320 短边缓存**
    #   （`finetune_lora.py` 的 `_mp(tr_paths), _mp(va_paths)`）⇒ 训练日志里的
    #   `val_acc` 量的是**软图**；而本脚本原先读的是 `data/train/train` 的**原图**。
    #   实测尺寸分布：训练原图短边中位 **480**（45% ≥512）、320 缓存 **320**、
    #   官方测试图 **375** ⇒ 本机评估条件的锐度**高于**榜分条件 ⇒ 会把
    #   "只在锐图上占优"的改动高估（疑似 ul4/v19 那 7pp 的来源）。
    #   打开本开关即可用**与训练期 val 同一套图**再量一次，检验"7pp 是不是域造成的"。
    if os.environ.get("AIC_VAL_SOFT") == "1":
        from data import find_train_root
        _p_root = Path(os.environ.get("AIC_VAL_SOFT_ROOT", "")) \
            if os.environ.get("AIC_VAL_SOFT_ROOT") else (C.DATA_DIR / "_preproc" / "train")
        _tr_root = find_train_root()
        _miss, _out = 0, []
        for _p in va_paths:
            try:
                _q = _p_root / Path(_p).relative_to(_tr_root)
            except ValueError:
                _q = None
            if _q is not None and _q.exists():
                _out.append(str(_q))
            else:
                _out.append(_p)
                _miss += 1
        va_paths = _out
        print(f"[tta] ★ 软域模式：留出集改读 {_p_root}"
              f"（缺失回落 {_miss} 张，占 {_miss/max(1,len(va_paths)):.1%}）")

    if a.limit:
        sel = np.arange(min(a.limit, len(va_paths)))
        va_paths = [va_paths[i] for i in sel]
        yva = yva[sel]
        low = low[sel]
        print(f"[tta] ⚠️ --limit {a.limit} ⇒ 只用 {len(va_paths)} 张，结论仅供参考")

    ckpt = Path(a.lora)
    if not ckpt.exists():
        print(f"[tta] 🛑 找不到 {ckpt}")
        return 1
    model = load_lora_model(ckpt, K, device)
    head_p = (C.CACHE_DIR / a.head) if a.head else None
    if head_p is not None:
        if not head_p.exists():
            print(f"[tta] 🛑 找不到外部头 {head_p}")
            return 1
        print(f"[tta] ★ 改用外部头 {head_p.name}（LoRA 骨干 + train_head 的头）")

    want = [s.strip() for s in a.views.split(",") if s.strip()]
    by_name = {v.name: v for v in ALL_VIEWS}
    unknown = [w for w in want if w not in by_name]
    if unknown:
        print(f"[tta] 🛑 未知视图 {unknown}")
        return 1

    probs: dict[str, np.ndarray] = {}
    for w in want:
        v = by_name[w]
        kind = ("训练分布内随机裁剪" if v.rc
                else ("整图压方形" if v.full_square else f"短边={v.shortest}"))
        print(f"\n[tta] === 视图 {v.name}（{kind} flip={v.flip} "
              f"等效面积占比≈{v.area}）===")
        probs[w] = infer(model, va_paths, build_view_tf(v), device,
                         a.batch, a.workers, w,
                         seed_base=(v.seed_base if v.rc else None),
                         head_ckpt=head_p)
        if v.rc:
            # ★ 自检：随机视图若与 base 完全相同 ⇒ transform 没生效，
            #   后面的"集成增益"全是自欺（本工程反复踩到的静默空操作模式）。
            if "base" in probs:
                d = float(np.abs(probs[w] - probs["base"]).max())
                print(f"[tta/{w}] 与 base 的概率最大差 {d:.4f}"
                      + ("  🛑 疑似 transform 未生效！" if d < 1e-6 else "  ✓"))

    # ---------------------------------------------------------- 逐视图读数
    def read(p: np.ndarray) -> tuple[float, float, float, float]:
        pred = p.argmax(1)
        acc = float((pred == yva).mean())
        mr = CLF.macro_recall(pred, yva, K)
        accL = float((pred[low] == yva[low]).mean()) if low.sum() else float("nan")
        mrL = CLF.macro_recall(pred[low], yva[low], K) if low.sum() else float("nan")
        return acc, mr, accL, mrL

    print("\n" + "=" * 92)
    print(f"逐视图（留出集 {len(yva)} 张；★ 取舍看「低相似」两列）")
    print("=" * 92)
    print(f"{'视图':<12}{'全体acc':>10}{'全体MR':>10}{'低相似acc':>11}{'低相似MR':>11}")
    print("-" * 92)
    base = None
    for w in want:
        acc, mr, accL, mrL = read(probs[w])
        if w == "base":
            base = probs[w]
        print(f"{w:<12}{acc:>10.4f}{mr:>10.4f}{accL:>11.4f}{mrL:>11.4f}")

    # ---------------------------------------------------------- 视图平均
    # ★ 组合按「假设」分组，别只堆一个"全部平均"：
    #     safe  = 训练分布内（base/flip/rc*）⇒ 只降方差，不引分布偏移
    #     reso  = safe + zoom/square      ⇒ 额外赌"等效分辨率"
    #   两组分别读数，才知道增益到底来自"降方差"还是"提分辨率"。
    print("-" * 92)
    # ★★ 按视图**属性**自动分组，不硬编码名字列表 ——
    #   硬编码名单的隐患是"加了新视图但忘了更新名单" ⇒ 新视图**静默掉出分组**，
    #   只出现在"全部视图"里，于是"扩展视图"的实验看起来毫无变化（本工程踩过多次
    #   同类的静默失效）。语义分界：
    #     safe = 训练分布内（base / flip / rc*）⇒ 只降方差，不引分布偏移
    #     reso = 其余（zoom* / square）⇒ 改了裁剪范围，赌"等效分辨率"
    safe = [w for w in want if by_name[w].rc or w in ("base", "flip")]
    reso = [w for w in want if w not in safe]
    combos: list[tuple[str, list[str]]] = []
    if "base" in probs:
        combos.append(("base+flip", [w for w in ("base", "flip") if w in probs]))
        if len(safe) > 2:
            combos.append(("safe(rc+flip)", safe))
        if reso and len(safe) > 1:
            combos.append(("safe+分辨率", safe + reso))
        if len(want) > 2:
            combos.append(("全部视图", list(want)))
    seen = set()
    for cn, ws in combos:
        if len(ws) < 2:
            continue
        key = tuple(ws)
        if key in seen:
            continue
        seen.add(key)
        pa = np.mean([probs[w] for w in ws], axis=0)
        acc, mr, accL, mrL = read(pa)
        print(f"{cn:<14}{acc:>10.4f}{mr:>10.4f}{accL:>11.4f}{mrL:>11.4f}"
              f"   ({len(ws)} 视图平均)")

    # ------------------------------------------------ ★ 诊断 A：rc 个数递增
    # 回答「rc 要几个才够」。若 k=2 与 k=8 差不多，提交时就没必要跑 8 个
    # （每个视图约 1 min/7537 张，生产端是 6 min/视图）。
    rcs = [w for w in want if by_name[w].rc]
    if len(rcs) >= 2 and "base" in probs:
        print("-" * 92)
        print("rc 个数递增（base+flip + 前 k 个 rc）—— 看边际收益在哪一档平掉：")
        for k in range(1, len(rcs) + 1):
            ws = [w for w in ("base", "flip") if w in probs] + rcs[:k]
            pa = np.mean([probs[w] for w in ws], axis=0)
            acc, mr, accL, mrL = read(pa)
            print(f"  k={k:<3d}{'':<8}{acc:>10.4f}{mr:>10.4f}"
                  f"{accL:>11.4f}{mrL:>11.4f}")

    # ------------------------------------------------ ★ 诊断 B：留一法
    # 从「全部视图」里逐个去掉，看掉多少分。
    # ★ 去掉后**反而更好**（Δ 为正且低相似列不降）⇒ 该视图是**净负担**，
    #   生产配置里应当拿掉 —— 视图不是越多越好。
    if len(want) >= 4:
        print("-" * 92)
        print("留一法（从「全部视图」里去掉一个）—— **正数 = 去掉它反而更好**：")
        all_p = np.mean([probs[w] for w in want], axis=0)
        a_all, _, aL_all, _ = read(all_p)
        for w in want:
            rest = [x for x in want if x != w]
            pa = np.mean([probs[x] for x in rest], axis=0)
            acc, _, accL, _ = read(pa)
            flag = ("  ← 净负担，建议拿掉"
                    if (acc > a_all and accL >= aL_all - 1e-9) else "")
            print(f"  -{w:<12}{acc:>10.4f}{'':>10}{accL:>11.4f}{'':>11}"
                  f"  Δ全体 {acc - a_all:+.4f}  Δ低相似 {accL - aL_all:+.4f}{flag}")

    # ---------------------------------------------------------- 存逐视图概率
    # ★ 存下来是为了**离线**做任意子集组合 / 两两配对 / 留一法 ——
    #   本轮的教训：8 视图 vs 15 视图的差值只有 0.0044~0.0049，
    #   与标准误同量级，靠"读两侧绝对值"是判不动的，只能靠配对检验。
    if a.save_probs:
        # ★ 2026-09-25：整段包 try/except。
        #   为什么：这一步排在**所有读数打印之后**，但不加保护的话，一旦写盘出错
        #   （磁盘满 / 路径不可写）就会抛异常 ⇒ eval_any.sh 按退出码判"评估失败"
        #   ⇒ 白白丢掉一次 20~60min 的评估（v19 就是这种场景）。
        #   ⇒ 保存是"附加收益"，绝不能让它拖垮主产物。
        try:
            out_p = Path(a.save_probs)
            if out_p.suffix != ".npz":
                out_p = out_p.with_suffix(".npz")
            out_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_p,
                **{f"p_{w}": probs[w].astype(np.float16) for w in want},
                yva=yva.astype(np.int64),
                low=np.asarray(low),
                va_idx=va_idx.astype(np.int64),
                views=np.array(want),
            )
            print(f"[tta] ★ 已存逐视图概率 {out_p}（{len(want)} 个视图、"
                  f"{out_p.stat().st_size / 1e6:.0f} MB）⇒ 可离线做任意组合与配对检验")
        except Exception as _e:                                   # noqa: BLE001
            print(f"[tta] ⚠️ 保存逐视图概率失败（**不影响上面的读数**）: {_e!r}")

    # ---------------------------------------------------------- 配对检验
    if base is not None:
        ok_b = base.argmax(1) == yva
        print("\n" + "=" * 92)
        print("配对比较（相对 base，即当前生产预处理）")
        print("=" * 92)
        for cn, ws in combos:
            if len(ws) < 2:
                continue
            pa = np.mean([probs[w] for w in ws], axis=0)
            ok_a = pa.argmax(1) == yva
            f, b, pv = mcnemar(ok_b, ok_a)
            d = float(ok_a.mean() - ok_b.mean())
            print(f"  {cn:<14} Δacc={d:+.4f}  错→对 {f:>4} / 对→错 {b:>4} "
                  f"净 {f - b:>+5}（McNemar p={pv:.3g}）")
        print("\n⚠️ 这是**配对**差值，比绝对值可信；但仍不是榜分预测。")
        print("⚠️ 决定采纳前，务必确认增益在「低相似」列上**未被反转**"
              "（只有全体涨 = 在吃近似重复的便宜）。")
        print("⚠️ 本留出集约 7537 张 ⇒ 全体acc 的标准误约 0.0055；"
              "低相似子集更小，标准误更大 ⇒ **1 个点以内的差异别当真**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
