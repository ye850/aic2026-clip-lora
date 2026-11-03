# -*- coding: utf-8 -*-
"""在固定留出集（下称"尺子"）上计算准确率。

用法::

    python scripts/eval_acc.py <tag> [prefix]

读取 ``outputs/_rt_<tag>/pred_results.csv``（每行 ``文件名, 类别id``）与
``outputs/_valpin/val_labels.txt``（每行 ``文件名, 类别id``），按文件名求交集后
计算准确率，并打印一行标准格式：

    STAR <tag> pinned-val = 81.97% (n=4620)

说明
----
* **只用文件名求交集**，不依赖顺序，避免两份清单排序不同导致错配。
* 该留出集由训练时的 ``--val-ratio 0.05 --val-per-class 8`` 分层产生（4,620 张）。
* ⚠️ 用「真全量（``--val-ratio 0``）」训练出来的权重在这套留出集上是**训练召回**，
  不能当作泛化指标 —— 此时应直接比对提交文件的 md5，而不是看这里的分数。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def _read_pairs(p: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            d[parts[0]] = parts[1]
    return d


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/eval_acc.py <tag> [prefix]", file=sys.stderr)
        return 2
    tag = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "_rt_"

    pred_p = OUT / f"{prefix}{tag}" / "pred_results.csv"
    label_p = OUT / "_valpin" / "val_labels.txt"
    for p in (pred_p, label_p):
        if not p.exists():
            print(f"[eval_acc] 缺少文件：{p}", file=sys.stderr)
            return 1

    pred = _read_pairs(pred_p)
    gold = _read_pairs(label_p)
    common = set(pred) & set(gold)
    if not common:
        print("[eval_acc] 预测与标签没有交集，请检查文件名格式", file=sys.stderr)
        return 1

    hit = sum(1 for k in common if pred[k] == gold[k])
    acc = 100.0 * hit / len(common)
    print(f"  STAR {tag} pinned-val = {acc:.2f}% (n={len(common)})")
    if len(common) < len(gold):
        print(f"  [warn] 标签共 {len(gold)} 条，只匹配到 {len(common)} 条", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
