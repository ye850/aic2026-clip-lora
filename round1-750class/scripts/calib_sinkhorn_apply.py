#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""均衡校正升级版：**Sinkhorn 不动点**（CPU，零训练成本），替代会过冲的幂校正。

为什么要换掉幂校正
=====================================================================
`scripts/calib_apply.py` 的做法是**一轮**边际估计：
    m = 各列概率均值归一化;   p' ∝ p / m^τ
线上它确实有效（τ=0.8 → **72.1023**，τ=1.2 → **72.32**），但驱动它的是
「一次幂校正 = Sinkhorn 的第一步」，没有迭代到不动点。

⚠️ **更正（09-27，自测发现的错误）**：我最初在 `outputs/cache/probs_test_tta15.npy`
（**9月22日的旧模型概率，不是生产那份**）上量到"τ>1.6 后过冲发散"——
    τ=1.2 超额 8.67% → τ=1.6 最低 7.51% → τ=2.5 飙到 17.6% → τ=4.0 到 30.3%
**换到生产概率 `probs_test_d_r448ul4.npy` 后，这个现象不存在**，τ 单调收敛：

    来源                文件                          基线超额   τ=1.2      τ=4.0
    ❌ 旧模型(09-22)    probs_test_tta15.npy            17.61%   8.67%      30.30%(发散)
    ✅ 生产(复现 cal12) probs_test_d_r448ul4.npy        13.70%   11.20%      6.90%(单调)

⇒ 教训：**换概率文件会静默改变一切结论**。本地 cache 里有多个历史
  `probs_test*.npy`，**出包前必须用 `--ref-csv` 反查**：
  「我这套 base+τ 能不能**逐位复现**已提交包」——本脚本与
  `_diag_calib_headroom.py --check` 都会打印这一行，差异必须是 0.00%。

本脚本改用幂校正的**收紧版本**：直接解 Sinkhorn 不动点，找类偏置 b 使
    pred = argmax_k [ log p_ik + b_k ]
之后的**列计数**在给定目标边际下尽可能均衡。
好处：① 有不动点、不过冲 ② 剂量连续可读 ③ 目标边际可以不是均匀分布。
生产概率上的实测落点：基线 超额 13.70%/标准差 19.7 **且有 10 个类拿到 0 张**；
Sinkhorn(uniform) 把标准差压到 **3.7**、超额压到 **2.09%**（改动 16.83% vs base）。

两种目标边际（都要跑一次，用提交裁决 —— 经验：τ 轴上本地读数与线上分数可能反号）
=====================================================================
  `--target uniform`   每类 37444/750 = 49.925。若官方测试集是均衡采样，这是正解。
  `--target train`     按**训练集类频**同比缩放到 37444。
                       依据：`build_train_manifest()` 实测训练集类频
                       中位 196 / 均值 198.3 / min 5 / max 248（近均匀但有尾）
                       ⇒ 若测试集与训练集同分布，尾部类只该拿 1~15 张，
                         硬按 50 张分配就会把尾部类灌爆。
  ⚠️ 两个目标对**尾部类**的差别很大，必须分开实测，不能凭推理选。

与既有校正的关系（同一族，剂量不同）
=====================================================================
    τ=0.8  改 9.54%  → 榜分 72.1023
    τ=1.2  改 14.43% → 榜分 72.32
    τ=1.35 改 16.24%（已出包待提交）
    Sinkhorn(uniform) 改 24.20% vs base、13.19% vs τ=1.2、超额压到 3.73%
⇒ Sinkhorn 是"把软校正迭代到不动点"，属于同一族里更靠端点的剂量。

安全闸（与 calib_apply.py 同等，不放松）
=====================================================================
1. 概率形状要跟 `train_meta.json` 的类数、测试清单行数逐一对上
2. 改动比例 > `--max-change`（默认 0.30）⇒ 中止；默认比 calib_apply 的 0.20 宽，
   因为 Sinkhorn 的落点本就更大；**跑之前先看 _diag_calib_headroom.py 的落点**
3. `out_dir` 已有 `pred_results.zip` ⇒ 拒绝覆盖（--force 放行）
4. 写盘复用 `src/data.py::write_submission`（提交编号映射全项目最高风险，
   绝不在这里另起一套）

用法
----
    PYTHONPATH=src python scripts/calib_sinkhorn_apply.py \
        --probs outputs/cache/probs_test_tta15.npy \
        --out-dir outputs/submission_d_r448ul4_balU \
        --target uniform \
        --ref-csv outputs/_cal_pull/submission_d_r448ul4_cal12/pred_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

import config as C                      # noqa: E402
from data import ClassMap, build_test_manifest, write_submission   # noqa: E402


def load_ref_labels(path: Path) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    labs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                labs.append(int(line.split(",")[-1].strip()))
    return np.array(labs, dtype=np.int64)


def hist_stats(labels: np.ndarray, K: int) -> str:
    h = np.bincount(labels, minlength=K)
    exc = float((h - h.mean()).clip(0).sum())
    return (f"max={h.max():4d} 中位={int(np.median(h)):4d} 均值={h.mean():6.1f} "
            f"min={h.min():3d} 标准差={h.std():5.1f} 超额={100*exc/h.sum():5.2f}%")


def sinkhorn_bias(p: np.ndarray, target: np.ndarray, n_iter: int = 60,
                  damp: float = 0.5, tol: float = 0.6, verbose: bool = True
                  ) -> tuple[np.ndarray, np.ndarray]:
    """求类偏置 b 使 argmax(log p + b) 的列计数逼近 target。

    ★ 目标边际用**软化分配**的列和去驱动（Sinkhorn 的标准做法），
      不是直接用硬 argmax 的计数 —— 后者不可微、会振荡。
    ★ 阻尼 0.5 防振荡；上限 60 轮（每轮过一遍 (N,K) 的 exp，200 轮要 5 分钟以上）。
    """
    logp = np.log(np.maximum(p, 1e-30))
    b = np.zeros(p.shape[1], dtype=np.float64)
    target = target / target.sum() * len(p)
    for it in range(n_iter):
        col = np.zeros(p.shape[1], dtype=np.float64)
        for i in range(0, len(p), 4096):
            s = logp[i:i + 4096] + b[None, :]
            s -= s.max(1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(1, keepdims=True)
            col += w.sum(0)
        b += np.log(target / np.maximum(col, 1e-30)) * damp
        if verbose and it % 10 == 9:
            h = np.bincount((logp + b[None, :]).argmax(1), minlength=p.shape[1])
            print(f"    [sinkhorn] 第 {it+1:>3} 轮：实得列计数 标准差={h.std():6.2f} "
                  f"min={h.min()} max={h.max()}", flush=True)
            if h.std() < tol:
                break
    return b, col


def main() -> int:
    ap = argparse.ArgumentParser(description="Sinkhorn 均衡校正出包（CPU）")
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", default="uniform", choices=["uniform", "train"],
                    help="目标类别边际：uniform=每类 N/K；train=按训练集类频缩放")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--max-change", type=float, default=0.30,
                    help="改动比例上限，超过即中止（历史事故 --prior em 改掉 61.5%%）")
    ap.add_argument("--ref-csv", default="")
    ap.add_argument("--allow-unverified-map", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    pf = Path(a.probs)
    if not pf.exists():
        raise SystemExit(f"[bal] 🛑 找不到概率文件 {pf}")
    probs = np.load(pf)
    print(f"[bal] 概率 {pf.name}  shape={probs.shape} dtype={probs.dtype}")

    meta_p = C.CACHE_DIR / "train_meta.json"
    if not meta_p.exists():
        raise SystemExit(f"[bal] 🛑 缺少 {meta_p}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    K = int(meta["num_classes"])
    names = meta["class_names"]
    if probs.ndim != 2 or probs.shape[1] != K or len(names) != K:
        raise SystemExit(f"[bal] 🛑 形状不符 probs{probs.shape} / K={K} / names={len(names)}")

    te = build_test_manifest()
    img_names = te["img_name"].tolist()
    if probs.shape[0] != len(img_names):
        raise SystemExit(f"[bal] 🛑 行数不符 {probs.shape[0]} vs {len(img_names)}")

    class_map = ClassMap({k: int(v) for k, v in meta["class_map"].items()})
    class_map.confidence = meta.get("class_map_confidence", "unknown")
    class_map.source = meta.get("class_map_source", "?")
    col2id = np.array([class_map[n] for n in names], dtype=np.int64)

    p = probs.astype(np.float64)
    p /= np.maximum(p.sum(1, keepdims=True), 1e-30)
    base = p.argmax(1)
    N = len(p)

    # ---------- 目标边际 ----------
    if a.target == "uniform":
        target = np.ones(K, dtype=np.float64)
        src = "均匀（每类 N/K）"
    else:
        lb_p = _HERE.parent / "outputs" / "oof_local" / "oof_labels.npy"
        if not lb_p.exists():
            raise SystemExit(f"[bal] 🛑 --target train 需要 {lb_p}（全量训练标签）")
        cnt = np.bincount(np.load(lb_p).astype(np.int64), minlength=K).astype(np.float64)
        if len(cnt) != K:
            raise SystemExit(f"[bal] 🛑 训练标签类数 {len(cnt)} ≠ {K}")
        target = cnt
        src = (f"训练集类频缩放（中位 {int(np.median(cnt))} / min {int(cnt.min())} "
               f"/ max {int(cnt.max())}）")
    print(f"[bal] 目标边际 = {src}")

    # ---------- 校正 ----------
    print(f"[bal] 解 Sinkhorn 不动点（iters={a.iters} damp={a.damp}）...")
    b, _ = sinkhorn_bias(p, target, a.iters, a.damp)
    pred = (np.log(np.maximum(p, 1e-30)) + b[None, :]).argmax(1)

    chg = float((pred != base).mean())
    print(f"\n[bal] 校正前: {hist_stats(base, K)}")
    print(f"[bal] 校正后: {hist_stats(pred, K)}")
    print(f"[bal] 改动预测 {chg*100:.2f}%（{int((pred != base).sum())}/{N}）")
    if a.ref_csv:
        ref = load_ref_labels(Path(a.ref_csv))
        if ref is not None and len(ref) == N:
            same = int((col2id[pred] == ref).sum())
            print(f"[bal] 与对照包 {Path(a.ref_csv).parent.name}："
                  f"{N - same} 张不同（{(N - same)/N*100:.2f}%）")
            print("[bal] ⚠️ 测试集无标签 ⇒ 这里只能报**改动量**，报不出涨跌")
        elif ref is not None:
            print(f"[bal] ⚠️ 对照包行数 {len(ref)} ≠ {N}，跳过对比")

    hb = np.bincount(base, minlength=K).astype(np.float64)
    sw = pred != base
    if sw.any():
        print(f"[bal] 改动样本原预测类平均热度 {hb[base[sw]].mean():.1f} 张 → "
              f"新预测类 {hb[pred[sw]].mean():.1f} 张（应以后者更大 = 往冷门搬）")

    if chg > a.max_change:
        raise SystemExit(f"[bal] 🛑 改动 {chg*100:.2f}% 超上限 {a.max_change*100:.0f}% ⇒ 中止")

    out_dir = Path(a.out_dir)
    if (out_dir / "pred_results.zip").exists() and not a.force:
        raise SystemExit(f"[bal] 🛑 {out_dir} 已有 zip，拒绝覆盖")
    write_submission(img_names, col2id[pred], out_dir=out_dir,
                     class_map=class_map, class_names=names,
                     allow_unverified=a.allow_unverified_map)
    print(f"[bal] ★ 已写 {out_dir}")
    print(f"[bal] 下一步：scripts/check_submission.py {out_dir}/pred_results.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
