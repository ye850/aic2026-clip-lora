"""在同一份验证集上，比较「冻结线性头」与「LoRA 端到端微调」。

为什么必须共用同一份验证集
--------------------------
两条路线的验证集切分方式不同：
  · 冻结线性头（`train_head.train_final`）：**全局随机**抽 5%
  · LoRA（`finetune_lora.train`）：**按类分层**，每类至少 8 张
直接拿各自报告的 val_acc 对比，等于换了考卷比分数。本脚本统一复刻
**LoRA 那份分层切分**，把冻结头也在上面评一遍。

指标口径
--------
官方细则写「测试集**类别分布均衡**」⇒ 榜分 ≈ **macro recall**（各类召回率平均）。
本地验证集是长尾的，普通准确率会按训练先验加权、**系统性高估**榜分
（实测印证：本地 0.6187 vs 真实榜分 43.07）。
所以排序看 macro recall，普通 acc 只作参考。

自带校验：复刻出的验证集大小应为 **7537**（与训练日志 `验证 7537` 一致）；
冻结头在这份切分上的普通 acc 应与 LoRA 报告的 baseline 同量级。

用法::

    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/compare_lora_vs_frozen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C            # noqa: E402
import numpy as np            # noqa: E402
import torch                  # noqa: E402

import train_head as TH       # noqa: E402
from data import build_train_manifest  # noqa: E402

LORA_BEST = C.CACHE_DIR / "lora_full750_best.pt"
LORA_REPORT = C.CACHE_DIR / "lora_train_report_lora_full750.json"

# ★ 训练还没跑完时（`*_best.pt` 要等全部轮次结束才落盘），可以退而用
#   周期检查点 `lora_full750.pt` —— 它是**到目前为止的最优**。
#   代价：它还会继续变好，所以这一行是**下界**，不是最终成绩。
LORA_ROLLING = C.CACHE_DIR / "lora_full750.pt"


def resolve_lora(spec: str) -> tuple[Path, bool]:
    """把 `--lora` 的值解析成实际路径。返回 (路径, 是否只是进行中的检查点)。"""
    if spec != "auto":
        return Path(spec), False
    if LORA_BEST.exists():
        return LORA_BEST, False
    if LORA_ROLLING.exists():
        print(f"[cmp] ⚠️ {LORA_BEST.name} 还没落盘（要等全部轮次结束）。"
              f"退用**进行中的** {LORA_ROLLING.name} —— 它还会继续变好，"
              f"所以下面这一行是**下界**。")
        return LORA_ROLLING, True
    return LORA_BEST, False


def lora_val_mask(labels: np.ndarray, K: int, val_ratio: float = 0.05,
                  per_class_min: int = 8) -> np.ndarray:
    """**逐字复刻** finetune_lora.train 里的切分逻辑（改一处就对不上了）。

    对应源码：rng = RandomState(C.SEED)；对 k in range(K) 依次 rng.choice。
    注意 rng 的调用顺序必须与训练时完全一致 —— 所以这里不能跳过任何一类。
    """
    rng = np.random.RandomState(C.SEED)
    mask = np.zeros(len(labels), dtype=bool)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        want = min(len(idx), max(per_class_min,
                                 int(round(len(idx) * val_ratio))))
        # AUDIT_PATCH_FLOOR1：复刻 finetune_lora.py:576 的 2026-10-04 修
        # （保底每类至少留 1 张训练图）。缺这段，val 会多 1 张且成员不同
        # （只有 5 张的类183 会被整类划走，训练侧该类为 0 —— 已知 bug）。
        if want >= len(idx) and len(idx) > 1:
            want = len(idx) - 1
        sel = rng.choice(idx, size=want, replace=False)
        mask[sel] = True
    return mask


def macro_recall(pred: np.ndarray, y: np.ndarray, K: int) -> float:
    rec = [float((pred[y == k] == k).mean()) for k in range(K) if (y == k).any()]
    return float(np.mean(rec)) if rec else float("nan")


def nn_sim(X: np.ndarray, y: np.ndarray, val_idx: np.ndarray,
           train_idx: np.ndarray) -> np.ndarray:
    """每个验证样本到「**同类**训练样本」的最大余弦（全池，无抽样）。

    ★ 训练侧必须显式传 `train_idx`，且与 `val_idx` **不相交** ——
      否则会算出"自己和自己相似度 = 1.0"的假象（本项目已在抽取池那版踩过）。
    """
    out = np.full(len(val_idx), -1.0, dtype=np.float32)
    for k in np.unique(y[val_idx]):
        vi = np.where(y[val_idx] == k)[0]
        tr_k = train_idx[y[train_idx] == k]
        if len(tr_k) == 0:
            continue
        out[vi] = (X[val_idx[vi]] @ X[tr_k].T).max(axis=1)
    return out


def low_sim_mask(X: np.ndarray, y: np.ndarray, val_idx: np.ndarray,
                 train_idx: np.ndarray, thr: float = 0.90) -> np.ndarray:
    """在 `val_idx` 里挑出「同类最近邻余弦 ≤ thr」的那批。

    ★ 为什么每行都要单独算而不是共用一个掩码：**各行的切分不同**
      （① 用的是那份全局随机 5%，②③ 用的是 LoRA 分层切分），
      所以"谁算低相似"也随行而变。用同一个掩码会把别的切分的样本算进来。
    ★ 刻画的**标尺统一用冻结 CLIP 特征** —— 低相似标记是"这张图在数据里
      不典型"的**数据属性**，不是某个模型的属性；统一标尺才可比。
    """
    ms = nn_sim(X, y, val_idx, train_idx)
    return ms <= thr, ms


def eval_lora(paths: list[str], device: str, K: int,
              ckpt_path: Path = LORA_BEST) -> np.ndarray:
    """载入 LoRA 权重，返回按 paths 顺序的概率矩阵。"""
    import lora as LO
    from extract import load_clip
    from finetune_lora import (LabeledImageDataset, LoRAClipClassifier,
                               eval_transform)
    from torch.utils.data import DataLoader

    ck = torch.load(ckpt_path, map_location=device)
    kind = ck.get("backbone_kind", "transformers")
    r = int(ck.get("r", 16))
    tr_args = ck.get("args", {})
    hidden = int(tr_args.get("hidden", 0))
    targets = tuple(str(tr_args.get("targets", "q,k,v,out")).split(","))
    print(f"[lora] {ckpt_path.name} best_ep={ck.get('best_ep')} "
          f"best_val_acc={ck.get('best_val_acc')} r={r} hidden={hidden} kind={kind}")

    _, clip_model, _ = load_clip(device)
    clip_model = clip_model.float()
    if r > 0:
        LO.inject_lora(clip_model, r=r, alpha=tr_args.get("alpha"),
                       dropout=0.0, targets=targets, verbose=False)
    model = LoRAClipClassifier(clip_model, kind, K, hidden=hidden).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[lora] ⚠️ 载入有差异：missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = LabeledImageDataset(paths, np.zeros(len(paths), dtype=np.int64), train=False)
    ds.tf = eval_transform(paths)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4,
                    pin_memory=False, prefetch_factor=2)

    out = []
    with torch.no_grad():
        for bi, (x, _y) in enumerate(dl, 1):
            x = x.to(device, non_blocking=False)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            if bi % 10 == 0:
                print(f"[lora] {bi}/{len(dl)}", flush=True)
    return np.concatenate(out, 0)


def eval_frozen_on_lora_val(Xtr: np.ndarray, y: np.ndarray, m: np.ndarray,
                            K: int, device: str) -> np.ndarray:
    """在 LoRA 那份切分上，评一个**没见过它**的冻结线性头。

    ★★★ 为什么必须重训，不能直接用 `head.pt`
    ---------------------------------------------------------------
    `head.pt` 是用**全部 148,695 张**训的（它自己的验证集是那份"全局随机 5%"）。
    而 LoRA 的验证集是**按类分层**的另一份切分 —— 对 `head.pt` 来说，
    那批样本**就在它的训练集里**。直接评它等于在考它背过的题：

        实测踩到：head.pt 在 LoRA 切分上 acc = **0.9119**，
        在它自己那份上是 0.6187。而 0.9119 恰好等于它**在训练集上的
        拟合率 0.9118** —— 数字自己招了。
        （判别技巧：**异常偏高 + 与已知的训练集拟合率吻合 = 泄漏**。）

    所以这里用**同一套配置**（sce / α=0.1 / β=0.05 / CL 降权 / 类平衡 / T 收缩 0.5）
    在 `~m` 上重训一个头，只在 `m` 上评。

    ⚠️ 成本已随配置变化：`config.HEAD_EPOCHS` 现在是 **600**、`HEAD_LR` 是 **1e-4**
    （2026-09-22 实测标定，见 config 注释），所以重训一次约 **2~3 分钟**，不是当初的 22 秒。

    ⚠️⚠️ **必须用当前的最优配方重训，不能沿用旧的 `head_loraVal.pt`。**
    否则就是拿"lr=1e-2 / 60 轮的旧冻结头"去比"LoRA"，结论会**系统性偏袒 LoRA** ——
    而那个 lr 恰恰已被证明是错的。缓存命中时若配置已变，请先删掉这个文件。
    """
    fit = C.CACHE_DIR / "head_loraVal.pt"
    if fit.exists():
        ck = torch.load(fit, map_location="cpu")
        print(f"[cmp] 复用 {fit.name}（已在 ~该切分 上训练）"
              f"⚠️ 若 config 的 HEAD_LR/HEAD_EPOCHS 已变，请删掉它重训")
        model = TH.load_final(fit, device=device)
    else:
        # ★ 用 config 的统一入口读 T（按 T_STRENGTH 收缩），保证与生产同源。
        T = C.load_transition_matrix()
        print(f"[cmp] 在排除该切分的 {int((~m).sum())} 张上重训一个冻结头"
              f"（同一套配置：{C.LOSS_KIND} / lr={C.HEAD_LR:g} / "
              f"{C.HEAD_EPOCHS} 轮）…")
        # ★ train_final 期望的是**已在 device 上的 tensor**（内部会 .cpu()），
        #   传 numpy 会直接报错 —— 与 stage_train 的调用保持一致。
        Xfit = torch.as_tensor(Xtr[~m], device=device)
        yfit = torch.as_tensor(y[~m], device=device)
        model = TH.train_final(Xfit, yfit, T, int(K), device,
                               kind=C.LOSS_KIND, log_prefix="[cmp/frozen]",
                               save_name=fit.name)
    probs = TH.accuracy(model, torch.as_tensor(Xtr[m], device=device),
                        None, prior_q=None).cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="冻结头 vs LoRA 公平对照")
    ap.add_argument("--heads", default="head.pt,head_e600_lr1e-4.pt",
                    help="参考用的头文件（逗号分隔，位于 outputs/cache/）。"
                         "默认给「旧配方（已交 43.0723）」与「新配方」各一个，便于对照")
    ap.add_argument("--lora", default="auto",
                    help="LoRA 权重路径；auto = 优先 %%_best.pt，没有就用进行中的检查点")
    ap.add_argument("--save-probs", default="",
                    help="把 LoRA 在留出集上的%%%%逐样本概率%%%%存成 "
                         "`<前缀>_lora.npy`（顺序 = va_paths，即复刻出的那 7537 张）。"
                         "★ 存下来是为了能在 `reweight_to_test.py` 里"
                         "用%%%%同一把相似度尺子%%%%统一读数。")
    a = ap.parse_args()
    lora_p, lora_partial = resolve_lora(a.lora)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = build_train_manifest()
    K = int(tr.attrs.get("num_classes", 0))
    labels = tr["label"].to_numpy().astype(np.int64)
    paths = tr["path"].tolist()

    m = lora_val_mask(labels, K)
    print(f"[cmp] 复刻 LoRA 验证集：{int(m.sum())} 张 / 训练集 {len(labels)} 张"
          f"（应等于训练日志里的「验证 7537」）")
    if int(m.sum()) != 7537:
        print("[cmp] ⚠️ 与日志的 7537 不一致 ⇒ 切分复刻有偏差，下面的比较不可用。")
        return 1
    va_paths = [p for p, keep in zip(paths, m) if keep]
    yva = labels[m]
    print(f"[cmp] 验证集覆盖 {len(np.unique(yva))}/{K} 类")

    Xtr = np.load(C.CACHE_DIR / "feat_train.npy")
    all_idx = np.arange(len(labels))
    rows = []   # (标签, probs, 真值, 低相似掩码)

    # ---- 参考行：我们实际提交用的那几个头，各评在**它自己**那份随机 5% 上（无泄漏）
    #   两行都留着，是为了把「旧配方（已交 43.0723）」和「新配方」放在同一张表里，
    #   直观看出改 lr 到底值多少。
    rng = np.random.RandomState(C.SEED)
    own = rng.rand(len(labels)) < 0.05
    own_idx = np.where(own)[0]
    low_own, _ = low_sim_mask(Xtr, labels, own_idx, all_idx[~own])
    heads = [s.strip() for s in a.heads.split(",") if s.strip()]
    for i, hn in enumerate(heads, 1):
        hp = C.CACHE_DIR / hn
        if not hp.exists():
            print(f"[cmp] ⚠️ 找不到 {hn}，跳过")
            continue
        model = TH.load_final(hp, device=device)
        p = TH.accuracy(model, torch.as_tensor(Xtr[own], device=device),
                        None, prior_q=None).cpu().numpy()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        rows.append((f"参{i} {hn}\n     评在它自己的随机5%上",
                     p, labels[own], low_own))

    # ---- 公平对照：重训的冻结头 vs LoRA，都评在**同一份** LoRA 分层切分上
    low_lora, _ = low_sim_mask(Xtr, labels, np.where(m)[0], all_idx[~m])
    p_frozen = eval_frozen_on_lora_val(Xtr, labels, m, K, device)
    rows.append(("② 冻结线性头（按当前配方重训）", p_frozen, yva, low_lora))

    if lora_p.exists():
        p_lora = eval_lora(va_paths, device, K, lora_p)
        rows.append(("③ LoRA 端到端微调" +
                     ("（进行中的检查点 ⇒ 下界）" if lora_partial else ""),
                     p_lora, yva, low_lora))
        if a.save_probs:
            np.save(f"{a.save_probs}_lora.npy", p_lora)
            print(f"[cmp] 已存 {a.save_probs}_lora.npy {p_lora.shape}"
                  f"（--split lora）")
    else:
        print(f"[cmp] ⚠️ 未找到 {lora_p}，跳过 LoRA 一侧。")

    print("\n" + "=" * 96)
    print("比较（标签含噪 ⇒ 只看相对排序，绝对值不可外推）")
    print("=" * 96)
    print(f"{'路线':<34}{'全体acc':>9}{'全体MR':>9}{'覆盖':>6}"
          f"{'低相似n':>8}{'低相似acc':>11}{'低相似MR':>10}")
    print("-" * 96)
    for name, p, yy, low in rows:
        pred = p.argmax(1)
        acc = float((pred == yy).mean())
        mr = macro_recall(pred, yy, K)
        accL = float((pred[low] == yy[low]).mean()) if low.sum() else float("nan")
        mrL = macro_recall(pred[low], yy[low], K) if low.sum() else float("nan")
        tag = name.split("\n")[0]
        print(f"{tag:<34}{acc:>9.4f}{mr:>9.4f}{len(np.unique(pred)):>6}"
              f"{int(low.sum()):>8}{accL:>11.4f}{mrL:>10.4f}")
    print("-" * 96)
    print("★ **取舍看「低相似」两列** —— `probe_stratified.py` 实测：本地 `全体` 指标")
    print("  被「近似重复图」撑着（>0.95 相似度的 16% 样本 acc 0.87），而官方测试集")
    print("  是另来源、无近似重复 ⇒ 更像「低相似 ≤0.90」那一块。")
    print("★ 只有 ② 与 ③ 可比（同一份切分，且两边都没在它上面训练过）。")
    print("  ① 是另一份切分，用来说明「我们实际提交的那个模型」处在什么水平。")
    print("\n⚠️ 本验证集 7537 张 / 750 类 ⇒ 每类约 10 张，单类召回率标准差约 0.155，")
    print("   平均后 macro recall 的标准误约 **0.006** ⇒ **小于这个量级的差异不要当真**；")
    print("   低相似子集更小 ⇒ 标准误更大，别拿 1~2 个点说事。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
