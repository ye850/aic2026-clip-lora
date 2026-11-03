# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""全局配置：路径、CLIP 规格、噪声鲁棒训练超参。

把赛题硬约束写进代码，防止后续改错：
  1. 骨干必须是 CLIP ViT-B/32，且只能用 OpenAI 官方公开权重；
  2. 不得使用多模型集成 / 投票融合 —— 本流程只产出「单一模型 + 单一推理流程」；
  3. 不得引入任何额外数据集（复赛不得复用初赛数据）。

⚠️ 本模块**必须在 import numpy / torch 之前**被导入（见下方环境变量块）。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 线程限制
# ⚠️⚠️ 这一块必须在 numpy / torch 之前执行，因此放在文件最顶部、且不依赖任何
# 第三方库。config.py 是全工程最先被导入的模块，取巧地借用它来设环境变量。
#
# 为什么必须这么做（实测踩坑）：
#   本机 20 逻辑核 / 16GB 内存（可用仅约 4.5GB）。DataLoader 的每个 worker
#   进程在 import numpy 时会各自初始化一份 OpenBLAS，默认按核数开 20 条线程。
#   num_workers=6 就是 6 x 20 = 120 条线程 + 6 份线程栈，直接把内存打爆：
#       OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
#       RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly
#   结果不是变慢，而是**整个提特征流程崩掉**。
#   限制为单线程后，worker 之间靠进程并行（而不是线程超订），既稳又快。
#
# 用 setdefault 而不是直接赋值：留给用户/集群覆盖的空间。
import os as _os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_k, "1")

# 国内网络环境下的 HF 镜像（不覆盖用户已设置的值）
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path  # noqa: E402

# 工程根目录：默认取本文件所在包的上层目录（src/ 的父目录）。
# 可用环境变量 AIC_ROOT 覆盖，便于把同一份代码放到不同阶段目录（初赛/复赛/半决赛）
# 下运行，而**数据与缓存各自独立、互不可见** —— 这是「阶段隔离」最物理的实现方式。
_ROOT_ENV = _os.environ.get("AIC_ROOT")
ROOT = (Path(_ROOT_ENV).resolve() if _ROOT_ENV not in (None, "")
        else Path(__file__).resolve().parents[1])
_DATA_ENV = _os.environ.get("AIC_DATA_DIR")
DATA_DIR = (Path(_DATA_ENV).resolve() if _DATA_ENV not in (None, "")
            else ROOT / "data")
# ★ 输出目录也跟随 AIC_ROOT。
# 踩过的坑：早期版本把 OUT_DIR 写成 `Path(__file__).parents[1] / "outputs"`，
# 结果设了 AIC_ROOT 后**数据读对、缓存写错** —— 两个对照实验（如冻结 vs LoRA）
# 会共享同一份 feat_train.npy / train_meta.json，且**不报任何错**，
# 于是「实验 A 的结论」里混进了「实验 B 的特征」，结论完全不可信。
# 这类静默串味的危害和跨阶段复用数据是同一级别，所以这里一并锚到 ROOT。
OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "cache"
LOG_DIR = ROOT / "logs"
SUB_DIR = OUT_DIR / "submission"

for _d in (DATA_DIR, OUT_DIR, CACHE_DIR, LOG_DIR, SUB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------- 阶段隔离护栏（取消成绩级）
# ⚠️⚠️ 规则原文：「复赛期间不得使用初赛的数据，半决赛期间不得使用初赛和复赛的数据。」
# 违反 = 取消成绩。
#
# 本工程（复赛）与初赛工程是两个**物理隔离**的独立目录，
# 正常情况下 AIC_ROOT 指向本工程、初赛数据早已清空，不存在误用可能。
#
# 但有一个真实的风险入口：`AIC_ROOT` 是环境变量，一旦被误设成初赛工程路径
# （例如上一次做冒烟实验时 export 过、或从初赛工程的脚本里继承了环境），
# 后续所有读写都会落到那个目录上 —— 程序照跑、不报错，**静默跨阶段**。
# 因此这里加一道**默认开启**的护栏。
#
# 关卡名单取自「初赛工程里出现过的、足以标识该阶段的特征」：
#   * data 下存在 500 类（初赛类别数）的 train 目录
#   * cache 里存着 num_classes=500 的 train_meta.json
# 命中则直接报错退出，除非显式设 AIC_ALLOW_CROSS_STAGE=1 覆盖。
_STAGE_KNOWN = {500: "初赛", 750: "复赛"}
_CROSS_STAGE_GUARD = _os.environ.get("AIC_ALLOW_CROSS_STAGE", "") not in ("1", "true", "True")


def _detect_stage() -> str | None:
    """从 data/train 的类别数或缓存的 meta 反推本目录属于哪个阶段。"""
    tr = DATA_DIR / "train"
    if tr.is_dir():
        try:
            n = sum(1 for p in tr.iterdir() if p.is_dir())
        except OSError:
            n = 0
        if n in _STAGE_KNOWN:
            return _STAGE_KNOWN[n]
        # 兜底：只看目录数量级，判断是否明显属于另一阶段
        if 400 <= n <= 600:
            return "初赛"
        if 600 <= n <= 900:
            return "复赛"
    meta = CACHE_DIR / "train_meta.json"
    if meta.exists():
        try:
            import json as _json
            k = int(_json.loads(meta.read_text(encoding="utf-8")).get("num_classes", 0))
            if k in _STAGE_KNOWN:
                return _STAGE_KNOWN[k]
            if 400 <= k <= 600:
                return "初赛"
            if 600 <= k <= 900:
                return "复赛"
        except Exception:  # noqa: BLE001
            pass
    return None


def _detect_root_by_name() -> str | None:
    """从 ROOT 目录名反推阶段 —— 这是**最后一道兜底**，也是唯一不依赖数据存在的一道。

    ★ 为什么需要它：`_detect_stage()` 的两条线索（data/train 的类别数、
    cache/train_meta.json）**都要求那个阶段的数据或缓存还在**。
    而按规则要求把初赛数据清空之后，一旦 AIC_ROOT 被误设成初赛工程路径，
    `_detect_stage()` 会返回 None ⇒ 上面那个 `if _found and ...` 直接短路
    ⇒ **护栏在"最该生效的时候"恰好失效**。

    这是典型的"守卫依赖被守卫对象"缺陷：清理动作本身把守卫的判据抹掉了。
    所以这里补一条与数据无关的判据 —— 目录名。
    """
    name = ROOT.name
    if "半决赛" in name:
        return "半决赛"
    if "复赛" in name:
        return "复赛"
    if "初赛" in name or "挑战赛道" in name:
        return "初赛"
    return None


STAGE_NAME = _os.environ.get("AIC_STAGE", "复赛")
if _CROSS_STAGE_GUARD:
    _found = _detect_stage() or _detect_root_by_name()
    if _found and _found != STAGE_NAME:
        raise SystemExit(
            f"\n[stage-guard] 🛑 阶段隔离冲突，已阻止运行：\n"
            f"[stage-guard]    本代码声明所处阶段 = {STAGE_NAME}\n"
            f"[stage-guard]    但 {ROOT} 看起来属于「{_found}」阶段"
            f"（判据：{'数据/缓存' if _detect_stage() else '目录名'}）\n"
            f"[stage-guard] 规则禁止复用上一阶段数据（违者取消成绩），因此不继续。\n"
            f"[stage-guard]\n"
            f"[stage-guard] 若确认这是误判（例如正在做跨阶段对照实验），可设：\n"
            f"[stage-guard]    AIC_ALLOW_CROSS_STAGE=1\n"
            f"[stage-guard] 若只是想换阶段运行，请改 AIC_ROOT 指向正确目录，例如：\n"
            f"[stage-guard]    AIC_ROOT=<本阶段工程目录的绝对路径>\n"
        )

STAGE_INFO = {"stage": STAGE_NAME, "root": str(ROOT),
              "detected": _detect_stage() or _detect_root_by_name()}

# ---------------------------------------------------------------- CLIP
# 规则原文允许「openai/clip 库 或 HuggingFace Transformers 对应版本」。
# 该 repo 即 OpenAI 官方 ViT-B/32 权重的 HF 镜像，二者数值等价。
CLIP_HF_ID = "openai/clip-vit-base-patch32"
CLIP_OPENCLIP_ID = ("ViT-B-32", "openai")  # 备选实现，二选一

# 本地权重目录：优先使用。用 curl 从 hf-mirror 预下载后放这里，
# 好处 ① 不依赖 HF 缓存目录布局 ② 之后完全离线可复现（总决赛要复核）。
# 实测：huggingface.co 在本机**不可达**，hf-mirror.com 可达。
#
# ★ 权重目录跟随 ROOT，但**允许用 AIC_CLIP_DIR 单独覆盖**。
# 动机：做冒烟/对照实验时常把 AIC_ROOT 指到临时目录（如 _smoke/），
# 那里没有 605MB 的权重，若不做覆盖就只能联网重下 —— 既慢又依赖网络，
# 还会往 HF 缓存里塞东西。605MB 的官方权重跨实验复用完全合理（它不是数据）。
_clip_dir_env = _os.environ.get("AIC_CLIP_DIR")
if _clip_dir_env not in (None, ""):
    CLIP_LOCAL_DIR = Path(_clip_dir_env).resolve()
else:
    CLIP_LOCAL_DIR = ROOT / "models" / "clip-vit-base-patch32"
    # 若 ROOT 下没有权重，回退到「本文件所在工程的 models/」——
    # AIC_ROOT 被指到临时子目录时能自动找到真权重，避免无谓联网。
    if not (CLIP_LOCAL_DIR / "config.json").exists():
        _fallback = Path(__file__).resolve().parents[1] / "models" / "clip-vit-base-patch32"
        if (_fallback / "config.json").exists():
            CLIP_LOCAL_DIR = _fallback
IMG_SIZE = int(_os.environ.get("AIC_IMG_SIZE", "224"))
# ★ 2026-09-23 起 IMG_SIZE 可用环境变量覆盖（默认 224 = 历史行为）。
#   动机：细粒度识别对**输入分辨率**极敏感，而本项目此前始终锁在 224
#   （CLIP ViT-B/32 的预训练分辨率）。测试图是 500×375；短边缩到 224 时
#   每个 patch（32px）覆盖 32 个原图像素，升到 288 后每 patch 只覆盖 24 像素
#   ⇒ patch 粒度更细，更利于区分细粒度类别。
#   ⚠️ 三处必须同步（都由本变量驱动，改这一个即可）：
#     1. 训练/评估 transform（extract.build_transform、finetune_lora._build_tf）
#     2. TTA 视图几何（tta_lora_eval 里 shortest 按 IMG_SIZE/224 同比缩放）
#     3. `interpolate_pos_encoding=True` —— 位置编码是 7×7，换 9×9 时必须插值，
#        否则 transformers 直接抛 shape 不匹配。
#   ⚠️ 分辨率一改，**所有历史绝对分数都不可横比**（与换预处理缓存同理）。
FEAT_DIM = 512  # get_image_features 输出（投影后），与文本空间对齐

# 特征提取
EXTRACT_BATCH = 96

# DataLoader 工作进程数（做 JPEG 解码 + 预处理，与 GPU 前向重叠）。
#
# ★ 这个数字是**实测标定**出来的，不是拍脑袋（scripts/probe_workers.py）：
#   用模型 t(n) = F(W) + n/c(W) 在 n=1536 / 6144 两组上解出固定开销与稳态吞吐：
#       W=0 : F=  1.1s   c= 189 img/s
#       W=4 : F= 10.9s   c= 755 img/s   ← 正好 4.0x，线性扩展
#       W=8 : F= 19.9s   c=1140 img/s   ← 6.0x
#   固定开销 F 主要是 Windows 下 spawn 一个子进程要重新 import torch/numpy，
#   约 2.5s/worker，且**在 worker 数上线性累加**。
#   所以 worker 数只有在数据量大时才划算：
#       103,218 张 →  W=0: 9.1min   W=4: 2.5min   W=8: 1.8min
#       1,536   张 →  W=0: 9.2s     W=4: 12.9s    W=8: 21.2s（反而更慢！）
#   ⚠️ 小数据集冒烟测试时请手动设 AIC_EXTRACT_WORKERS=0，否则会被固定开销拖慢。
#
#   ★ 在真实尺寸（512px）数据上的对照（scripts/profile_loop.py，3072 张）：
#       W=0:  总计 34.3s  其中「等数据」占 89.2%   →  89.6 img/s
#       W=6:  总计  7.4s  其中「等数据」占 35.6%   → 415.4 img/s（4.6 倍）
#     说明瓶颈确实在 CPU 侧的 JPEG 解码 + 缩放，多进程是真收益。
#
# ⚠️ 解析环境变量时**不能用 `or` 兜底**：`int("0") or 默认值` 会因为 0 是 falsy
#    而悄悄把用户显式设置的 0 覆盖掉（这个坑真的踩过，导致一批对照实验全部失真）。
_EXTRACT_WORKERS_ENV = _os.environ.get("AIC_EXTRACT_WORKERS")
EXTRACT_WORKERS = (int(_EXTRACT_WORKERS_ENV)
                   if _EXTRACT_WORKERS_ENV not in (None, "") else
                   min(8, max(2, (_os.cpu_count() or 4) // 3)))

# 预处理实现：
#   "fast"     : 自写 torchvision 管线（PIL 双三次），单进程实测 201 img/s
#   "official" : 官方 CLIPImageProcessor，在 worker 内惰性创建，单进程 123 img/s
#
# 两者**数值差异极小**：换算回像素空间最大差约 2/255 灰阶（属舍入级），
# 实测（scripts/cmp_preprocess.py）差异量级 3.0e-2（归一化空间）。
# 默认 fast 是因为快 1.6 倍；若需要「与官方处理器逐位一致」的复核口径，
# 设 AIC_PREPROC=official 即可，代价是特征提取慢约 1.6 倍。
PREPROC_MODE = _os.environ.get("AIC_PREPROC", "fast")

# ★ 2026-10-07 标记为**死代码**（全仓无任何地方读取它；grep 仅此一处）。
#   它容易误导：会让人以为"关掉 fp16"要改这里。
#   实际控制推理精度的是 **autocast**：`scripts/tta_lora_eval.py` 里的
#   `torch.amp.autocast("cuda", enabled=...)`，已由 `AIC_NO_AMP=1` / `--no-amp` 控制。
#   （今天实测：关掉 autocast 后饱和率不变、预测差异 0.20% ⇒ 此项当前无收益）
#AMP_DTYPE = "float16"   # ← 已停用，勿删注释以免后人重新启用

# ------------------------------------------------- 噪声鲁棒分类头训练
# 头是在「缓存特征」上训练的：512 维之内的小矩阵，秒级一轮，
# 因此可以低成本地做多组消融、多折交叉验证。
#
# ★★★ HEAD_LR / HEAD_EPOCHS 的取值是**实测标定**的，不是默认值拍脑袋
#      （2026-09-22，`scripts/sweep_head_epochs.py`，真实 750 类全量数据、
#       5% 留出 7,406 张、macro recall 对齐榜单口径）。
#
# 原来的 HEAD_LR=1e-2 **明显过高**：在含噪标签上它很快把噪声也背下来 ——
# 末轮训练 loss 降到 **0.0035**（几乎完全拟合训练标签），而验证准确率反而
# 掉到 0.577。**训练损失照降、验证反向恶化，是"在噪声上过拟合"的典型指纹。**
#
# 固定其它一切、只改 lr（240 轮）：
#     lr      最优 acc / macroRecall   最优 epoch   末轮（未挑选）acc / MF1
#     1e-2    0.6335 / 0.6272          15           0.5890 / 0.5714
#     3e-3    0.6435 / 0.6380          30           0.6100 / 0.5892
#     1e-3    0.6568 / 0.6489          45           0.6348 / 0.6116
# 再延长到 600 轮：
#     3e-4    0.6630 / 0.6558          120          0.6369 / 0.6127
#     1e-4    0.6624 / 0.6569          210          0.6569 / 0.6277   ← 采用
#
# 两点结论：
#   (1) **「最优」与「末轮」两个口径同向单调** ⇒ 降 lr 是真实收益，
#       不是"按留出集挑最优 epoch"挑出来的虚高（挑得越多越虚高，见
#       train_head._train_loop 末尾 model.last_val 那段注释）。
#   (2) 3e-4 与 1e-4 已在抽样标准误（≈0.0056）内等价 ⇒ 到饱和点了，不必再降。
#       选 1e-4 是因为它的**末轮指标几乎等于最优**（0.6569 vs 0.6624），
#       对"挑哪个检查点"最不敏感 ⇒ 结论最经得起复核。
#
# ⇒ 相对旧的 60 轮 / 1e-2：macro recall **0.6137 → 0.6569（+0.0432）**，
#   约 8 倍标准误。按"本地 × 0.706 ≈ 榜分"的换算锚点约 +3 分。
#
# ⚠️ 注意 `OneCycleLR` 的 total_steps 与 epochs 挂钩 ⇒ **"加轮数"会同时改变
#    整条有效学习率轨迹**。两个变量混在一起时归因一定是错的 ——
#    本项目一度把功劳记给"轮数"，真凶是 lr 轨迹。
HEAD_EPOCHS = 600
HEAD_BATCH = 4096
HEAD_LR = 1e-4

# OOF 折内训练用的学习率 —— **刻意与 HEAD_LR 分开**。
#
# 为什么必须分开：OOF 只用来估噪声转移矩阵 T（confident learning），
# 每条折内模型只训 `epochs`(默认 20~25) 轮。若跟着 HEAD_LR 一起降到 1e-4，
# 20 轮根本训不起来 ⇒ OOF 概率接近均匀 ⇒ 估出的 T 退化 ⇒ 后续 CL 权重与
# 前向校正全部失真。**这是"为了调最终头而悄悄改坏了上游估计"的典型入口。**
OOF_HEAD_LR = 1e-2
HEAD_WD = 1e-4
HEAD_HIDDEN = 0  # 0 = 纯线性探针；>0 = 加一层 MLP 瓶颈
LABEL_SMOOTH = 0.0

# Confident Learning 噪声估计
CV_FOLDS = 5
SEED = 0

# 鲁棒损失权重
#   ★ 默认 "sce" 而非前向校正 —— 理由见下方 T_STRENGTH 注释。
# ★★★ 2026-10-05 重要变更：本项的含义已改变。
#   · `"sce"` = **标准 SCE**（RCE = -Σ p·log y，Wang et al. 2019），**这是新的默认含义**；
#   · 要复现 **2026-10-05 之前**已提交并得分的包（如榜分 74.3 的 `d_r448mlp1024`），
#     **必须显式用 `--kind sce_legacy`** —— 那是历史实现，
#     其 RCE 被写成了 -Σ p·log p（**预测熵，不含标签**），**不是 SCE**。
#     详见 `outputs/SCE实现错误_复核与后果_20261005.md`。
LOSS_KIND = "sce"  # ce | sce(标准) | sce_legacy(历史) | gce | forward_*
SCE_ALPHA = 0.1
# ⚠️⚠️ SCE_BETA 不能照搬到「端到端微调」场景。
#
# ★★★ 2026-10-05 更正两件事：
#
# (1) 下面这段解释原先**方向写反了**。历史实现（`sce_legacy`）的 RCE 是
#     **-Σ p·log p = 预测熵 H(p)**，**不含标签**。原注释说它"推向均匀"——**错**：
#       H(p) 在单纯形上**均匀取最大值**（=log K），**极小值(0)在一 hot**
#       ⇒ 最小化 H(p) 是「**锐化**」，不是「压平」。
#       准确描述：H(p) 是一股**无标签的锐化力**——谁当前 p 大就把谁压得更尖，不管对错。
#
# (2) **`SCE_BETA = 0.05` 是为"熵项"调出来的经验值，不适用于标准 SCE。**
#     标准 RCE 在 one-hot 下 = 9.21·(1−p_c)，量纲与 H(p) 完全不同
#     （H ∈ [0, log K]=[0,6.62]；标准 RCE ∈ [0, 9.21]，且与 p_c 单调反相关）。
#     同一 β 下两者的贡献量级差很多 ⇒ **换到标准 SCE 后 β 必须重新标定**。
#     官方论文（从头训 CNN）用 α=0.1 / β=1.0；本任务是微调，起点强，需实测。
#
# 那"val_acc 卡在 1/K"的现象为什么真的发生（历史实现下）？——用 H(p) 可定量解释：
#   设 K=20（logK=3.0），总损失 = α·CE + β·H：
#     · 均匀预测：          0.1·3.0 + β·3.0
#     · 全部塌缩成一个类（猜对 1/K）：0.1·[19/20·CE_wrong] ≈ 0.1·19 = 1.9，H≈0
#   β=1.0 时：均匀 3.30 > 塌缩 1.9 ⇒ **模型选择塌缩** ⇒ 观测到 acc≈1/K ✅
#   β=0.05 时：均匀 0.45 < 塌缩 1.9 ⇒ 塌缩不再划算 ⇒ acc 回到 0.5375 ✅
# ⇒ **现象与数字都对，只有"为什么"写反了。**
#
# 实测（K=20 子集 / 800 张 / LoRA r=16 / lr1e-4 / 8 轮 / 其余参数完全一致）：
#   | 损失        | val_acc |
#   | SCE beta=1.0（原默认）| 0.0375  ← ≈随机 0.05 |
#   | SCE beta=0.2         | 0.2500 |
#   | SCE beta=0.05        | 0.5375 |
#   | 朴素 CE              | 0.5312 |
#   | GCE q=0.7            | 0.2812 |
# beta 单调、是唯一变量 ⇒ 证据干净。结论：**端到端场景需要 beta ≲ 0.05**。
#
# 那 alpha/beta 该取多少？SCE 原论文（Wang et al. 2019）给的是 alpha=0.1/beta=1.0，
# 但那是在**从头训练 CNN** 的设定下标定的；本任务是从预训练 CLIP 出发的微调，
# 起点已经很强，过度依赖 RCE 的"防噪"价值低于它带来的"压平"代价。
# 现在这个 0.05 是**本机实测的经验值**，不是定理；复赛正式跑之前
# 应当在更大子集上再核一次（扫描见 scripts/run_ab_harness.py）。
SCE_BETA = 0.05
GCE_Q = 0.7

# 前向校正里 T 的可信度：T_use = (1-strength)*I + strength*T。
#
# 为什么要这个旋钮：噪声率估计在两个方向都有偏 ——
#   * confident_joint（Northcutt 原始）：本场景严重低估（真实 30% 估成 0.04%），
#     T 退化单位阵，校正失效；
#   * confusion（本工程默认）：把「模型自身误差」也算进噪声，倾向高估
#     （真实 30% 估成 46%）;
#   * 一个错的 T 会**主动伤害**性能（多种子实测出现过 -6 个百分点）。
# 因此前向校正默认用 strength<1 收缩，且只在排行榜确认有增益时才开满。
T_STRENGTH = 0.5

# CL 保真度降权：w_i = clip(T[y_i, y_i], CLIP_LO, CLIP_HI)
#
# ★ 这一项相对安全：combine_weights 会把权重按均值归一化，因此**只有类别间的
#   相对差异**才起作用；若估计出的保真度整体偏低（一律相等），权重全约等于 1，
#   即自动退化为「无操作」，不会误伤。
USE_CL_REWEIGHT = True
CLIP_LO = 0.30
CLIP_HI = 1.00

# 长尾类别平衡（复赛/半决赛有长尾）：
#   none | inv | cb      cb = class-balanced (Cui 2019)
USE_CLASS_BALANCE = "cb"

# ⚠️⚠️ CB_BETA 必须落在**开区间 (0, 1)**。设成 1.0 会让全部样本权重变成 NaN
# （分子 1-beta=0，随后 0/0），loss 静默变 nan、准确率掉到随机水平以下，
# 而且训练**不会崩**，只会白跑完还"正常"保存一个模型。实测白跑过 60 轮。
# noise.class_balanced_weights 与 train_head 现在都会主动拦截，不要依赖它们兜底。
#
# ★ 取值依据（本工程真实 750 类数据实测，见 scripts/sweep_cb_beta.py）：
#
#   CB_BETA | 类平衡跨度 | 组合跨度(×CL) | 覆盖率 | 覆盖类别数
#   --------+------------+---------------+--------+-----------
#   0.9     |    2.4x    |     8.1x      | 0.9907 |    743
#   0.99    |   18.7x    |    62.3x      | 0.9973 |    748   ← 推荐
#   0.999   |   44.0x    |   145.7x      | 0.9973 |    748
#   0.9999  |   49.0x    |   162.0x      | 0.9973 |    748
#   1.0     |    NaN     |     NaN       | 0.0013 |      1   ← 全崩
#
# 两点结论：
#   (1) **0.99 与 0.999 在覆盖度上完全等价**（都是 748/750 类，
#       熵比 0.9761 vs 0.9774），但 0.99 的权重跨度小得多（62.3x vs 145.7x）。
#       既然效果一样、极端程度低一半，就没有必要用 0.999。
#   (2) 本工程真实长尾**其实很轻**（头尾比 49.6x、Gini 0.0956、
#       有效类数 736.1/750）。0.999 是对着比实际严重得多的长尾设的，
#       把尾部类样本的梯度贡献压到了头部的约 1/146。
#   ⚠️ 但覆盖度只是"没塌"的证据，**不等于榜分更高**。含噪指标无法给抗噪
#      方案排名，所以 0.99 这个新默认值**仍须由排行榜确认**；
#      若榜分不如预期，回退到 0.999 是最省事的排查动作。
CB_BETA = 0.99

# 无监督先验修正：把预测分布对齐到「目标先验」。
# 复赛/半决赛类别不均衡，直接 argmax 会偏头类。
USE_PRIOR_CORRECTION = True

# ------------------------------------------------------------- 提交格式
# 类别编号统一四位数字，不足前补 0；文件名必须与测试集完全一致。
SUB_CSV_NAME = "pred_results.csv"
CLASS_ID_WIDTH = 4

# 数据目录约定（用户下载后按此摆放；scan 时也会自动识别变体）
TRAIN_ROOT = DATA_DIR / "train"
TEST_ROOT = DATA_DIR / "test"
TEST_LIST_FILE = DATA_DIR / "test_list.txt"
CLASS_MAP_FILE = DATA_DIR / "class_map.csv"


# ------------------------------------------------------------ 实验脚本专用
def load_transition_matrix(t_strength: float | None = None):
    """读 `cache/T.npy` 并按 `T_STRENGTH` 收缩 —— **实验脚本必须用这个入口**。

    ★ 为什么专门做这么一个函数（踩过的坑，2026-09-22）：

    生产路径 `pipeline.stage_train` 是这样的：
        T = np.load(T.npy) → T = temper_transition_matrix(T, T_STRENGTH)
        → 既用于 `combine_weights`（CL 保真度降权），也用于前向校正损失

    而新写的两个实验脚本（`probe_val_trust.py` / `sweep_head_epochs.py`）各自
    写成了「用**原始** T 算 CL 权重、用收缩后的 T 算损失」—— 一半对一半。
    这种差异**不会报错、指标看起来也完全正常**，但测出来的已经不是生产配方了：
    样本权重分布变了，而"哪个配置更好"的结论可能就建立在这上面。

    ⇒ 统一到这一个入口，脚本里不要再出现裸的 `np.load(C.CACHE_DIR / "T.npy")`。
    （这与「配置类事实必须回产物核对」是同一族纪律：**实验装置必须与生产同构**。）

    ``t_strength=None`` 时取 `C.T_STRENGTH`。传 0 表示完全不做前向校正。
    """
    import numpy as np  # 函数内 import：config 顶层刻意不引 numpy（见文件开头）

    import noise as _NZ
    T = np.load(CACHE_DIR / "T.npy")
    s = T_STRENGTH if t_strength is None else t_strength
    return _NZ.temper_transition_matrix(T, s)
