"""伪标签打标器：用指定 LoRA 权重对【训练子集】（排除 val）跑多视图 TTA，
存每样本概率 + 对齐索引，供伪标签回收（噪声标签校正）离线分析。

为什么存在
----------
数据集真噪声率 ~33.8%（outputs/cache/T.npy 对角估计）。冠军配方未开筛选，
高噪声标签原样进了训练。思路：用现役模型 + TTA 对训练子集打标，
"模型一致地不同意原标签"的样本 = 疑似噪声 ⇒ 重训时用伪标签替换原标签。
配套第二信号（特征空间最近邻标签自洽，无自训练偏置）由 pseudo_review.py
在本地 CPU 上计算，两信号交叉后才回收，不单靠本产物拍板。

自训练偏置警示
--------------
打标权重若在训练子集上训过，会"记住"部分噪声标签 ⇒ 打标偏保守
（校正空间被低估）。缓解：15 视图非平凡增强仍能扰动记忆；且最终回收
以双信号一致为准。本脚本的 probs 是原料，不是判决。

对齐安全
----------------
build_train_manifest() 类内不排序 ⇒ 顺序随文件系统迭代，跨机器/跨版本
可能不同。本脚本把 paths 数组与 probs 行号一起落盘（pseudo_index.json），
下游一切分析按**文件路径**对齐，禁止按行号裸对齐。

val 绝对不碰
------------
val 参与逐轮选优。打标只对训练子集（val_mask 排除后的部分），
切分参数必须与训练时逐字一致：RandomState(C.SEED)、分层、
val_ratio=0.05、val_per_class=8（finetune_lora.py L374-384 的复刻）。

用法（云端 GPU）::
    PYTHONPATH=src python scripts/pseudo_label.py \
        --lora outputs/d_r448up/lora_best.pt \
        --views best --fusion power --fusion-t 4 \
        --workers 4 --batch 128 --out-dir outputs/pseudo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C            # noqa: E402  ★ 必须早于 numpy/torch（设 BLAS 线程）
import numpy as np            # noqa: E402
import torch                  # noqa: E402

from data import build_train_manifest  # noqa: E402
from tta_lora_eval import (            # noqa: E402  单一真源，与提交管线同源
    build_view_tf, infer, load_lora_model,
)
from predict_tta import resolve_views  # noqa: E402  视图集解析同源


def main() -> int:
    ap = argparse.ArgumentParser(description="训练子集 TTA 打标（伪标签原料）")
    ap.add_argument("--lora", required=True, help="打标用 LoRA 权重")
    ap.add_argument("--views", default="best",
                    help="视图集（与出包口径一致 ⇒ best=15 视图）")
    ap.add_argument("--fusion", default="power", choices=["arith", "power"])
    ap.add_argument("--fusion-t", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/pseudo")
    ap.add_argument("--limit", type=int, default=0,
                    help="分层抽样上限（冒烟用；先切 val 再抽，val 永不混入）")
    ap.add_argument("--val-ratio", type=float, default=0.05,
                    help="必须与打标目标训练时的值一致")
    ap.add_argument("--val-per-class", type=int, default=8,
                    help="必须与打标目标训练时的值一致")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--expect-k", type=int, default=0,
                    help="类别数守卫：manifest 类数 ≠ 此值 ⇒ 中止（复赛 =750，"
                         "防数据根指错或数据不全）")
    ap.add_argument("--probs-dtype", default="float16",
                    choices=["float16", "float32"],
                    help="★ 落盘精度。%%%%默认 float16%%%% 保持与历史读数一致；"
                         "但 float16 会让78.57%% 的 conf 饱和到分不出⇒ "
                         "τ 阈值筛选形同虚设。做新分析请用 float32。")
    a = ap.parse_args()

    tr = build_train_manifest()
    K = int(tr.attrs.get("num_classes", 0))
    paths_all = tr["path"].tolist()
    labels_all = tr["label"].to_numpy().astype(np.int64)
    print(f"[plab] 全量 {len(paths_all)} 张 / {K} 类", flush=True)
    if a.expect_k and K != a.expect_k:
        raise SystemExit(f"[plab] 🛑 manifest 类数 {K} ≠ 期望 {a.expect_k} "
                         f"⇒ 数据根不对或数据不全，中止")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    views = resolve_views(a.views)
    print(f"[plab] device={device}；视图 {len(views)} 个："
          f"{', '.join(v.name for v in views)}", flush=True)

    rng = np.random.RandomState(C.SEED)
    val_mask = np.zeros(len(labels_all), dtype=bool)
    for k in range(K):
        idx = np.where(labels_all == k)[0]
        if len(idx) == 0:
            continue
        want = min(len(idx), max(a.val_per_class,
                                 int(round(len(idx) * a.val_ratio))))
        sel = rng.choice(idx, size=want, replace=False)
        val_mask[sel] = True
    paths = [p for p, m in zip(paths_all, val_mask) if not m]
    labels = labels_all[~val_mask]
    print(f"[plab] 训练子集 {len(paths)} 张（val {int(val_mask.sum())} 张已排除，"
          f"val_ratio={a.val_ratio} per_class={a.val_per_class} seed={C.SEED}）", flush=True)

    if a.limit and a.limit < len(paths):
        keep = []
        per = max(1, a.limit // max(K, 1))
        r2 = np.random.RandomState(C.SEED)
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            sel = r2.choice(idx, size=min(per, len(idx)), replace=False)
            keep.append(sel)
        keep = np.concatenate(keep)
        r2.shuffle(keep)
        paths = [paths[i] for i in keep]
        labels = labels[keep]
        print(f"[plab] ⚠️ --limit {a.limit} ⇒ 冒烟只打 {len(paths)} 张", flush=True)

    model = load_lora_model(Path(a.lora), K, device)

    # ---- 视图循环（predict_tta.py 主循环同构：float64 累加 + 幂平均 + 行归一化）
    if a.fusion == "power" and a.fusion_t <= 0:
        raise SystemExit("[plab] 🛑 --fusion power 必须配 --fusion-t > 0")
    acc = None
    t0 = time.time()
    for i, v in enumerate(views, 1):
        tv0 = time.time()
        p = infer(model, paths, build_view_tf(v), device, a.batch, a.workers,
                  v.name, seed_base=(v.seed_base if v.rc else None), head_ckpt=None)
        if p.shape[0] != len(paths):
            raise SystemExit(f"[plab] 🛑 视图 {v.name} 返回 {p.shape[0]} 行，"
                             f"期望 {len(paths)} ⇒ 顺序错了，中止")
        contrib = (np.power(p.astype(np.float64), a.fusion_t)
                   if a.fusion == "power" else p.astype(np.float64))
        acc = contrib if acc is None else acc + contrib
        print(f"[plab] 视图 {i}/{len(views)} {v.name} 完成（{time.time() - tv0:.0f}s）",
              flush=True)
    row = acc.sum(1, keepdims=True)
    if np.any(row <= 0):
        raise SystemExit(f"[plab] 🛑 {int((row <= 0).sum())} 行融合后全 0（t 过大下溢？）")
    probs = (acc / row).astype(np.float32)
    print(f"[plab] 融合完成，总用时 {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- 落盘：probs + 对齐索引（下游按路径对齐，禁按行号）
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ★ 2026-10-06：落盘精度改为可选。
    #   实测这批数据 float16 ⇒ **78.57% 的样本 conf ≥ 0.9999**（饱和到分不出），
    #   而 float16 在 1.0 附近分辨率仅 2^-11≈4.9e-4 ⇒
    #   「真置信 0.9998」与「真置信 1.0000」**落盘后不可分**。
    #   ⇒ 直接后果：任何基于 τ 阈值的筛选**形同虚设**（实测 τ=0.9 仍留下 95%）。
    #   默认保持 float16 以复现历史读数；**做新分析必须用 float32**。
    _dt = np.float32 if a.probs_dtype == "float32" else np.float16
    np.save(out_dir / "probs_train.npy", probs.astype(_dt))
    index = {
        "lora": str(a.lora),
        "views": ",".join(v.name for v in views),
        "fusion": a.fusion, "fusion_t": a.fusion_t,
        "val_ratio": a.val_ratio, "val_per_class": a.val_per_class,
        "seed": int(C.SEED), "num_classes": K, "n_train": len(paths),
        "paths": paths, "labels": labels.tolist(),
    }
    (out_dir / "pseudo_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"[plab] 已存 {out_dir}/probs_train.npy {probs.shape} + pseudo_index.json",
          flush=True)

    # ---- 快速读数：一致率与三个置信阈值的回收规模预告
    pred = probs.argmax(1)
    conf = probs.max(1)
    agree = pred == labels
    print(f"[plab] 打标 vs 原标签：一致率 {agree.mean():.4f}"
          f"（不一致 {int((~agree).sum())} 张 = 疑似噪声池上限）", flush=True)
    for tau in (0.6, 0.8, 0.9):
        n = int((~agree & (conf >= tau)).sum())
        print(f"[plab]   回收预告 τ≥{tau}: {n} 张"
              f"（{n / max(len(paths), 1):.1%} of 训练子集）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
