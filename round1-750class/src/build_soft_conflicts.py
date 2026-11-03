#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""构造「冲突组软目标 + 组总权重=1」清单 + 各类权重影响检查
=====================================================================
依据（用户 2026-06-06 定的可审计实验规格）
--------------------------------------------------------------------
① **只依据文件哈希**处理冲突组（不用 CLIP 特征、不用近邻、不用模型预测）
② **每个冲突组总权重设为 1**
③ **软目标在组内不同标签之间均分**（不是复制到每张图再保留原权重——那已被证明无效）
④ 训练/验证**按哈希分组**，同一文件不跨集合（配套 `split_by_hash.py`）
⑤ **同步检查对各类样本权重的影响**

★ 与已被否掉的方案的区别（用户 10-06 指出）
   ❌ 旧：软标签分布复制给组内每张图 + 保留原总权重
        ⇒ 合计交叉熵梯度**基本等价于原冲突硬标签总和**，矛盾并未减少
   ✅ 新：组内每张权重 1/m（m=组内样本数），组总权重 = 1
        ⇒ 矛盾组的**监督力度被压到原来的 1/m²**（梯度 + 权重双重衰减）
        ⇒ 软目标**均分**给不同标签 ⇒ 承认"不知道哪个类对"

输入：`outputs/文件级重复标签冲突_确定错误.json`（1,877 组，已 SHA-256 逐字节复核）
输出：`outputs/dedup_soft_conflicts.json`（训练脚本直接消费）
      `outputs/各类权重影响检查_20261006.md`

用法::
    .venv/Scripts/python.exe scripts/build_soft_conflicts.py
=====================================================================
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs", "文件级重复标签冲突_确定错误.json")
IDX = os.path.join(ROOT, "outputs", "pseudo", "pseudo_feat_index.json")
OUT = os.path.join(ROOT, "outputs", "dedup_soft_conflicts.json")
MD_REPORT = os.path.join(ROOT, "outputs", "各类权重影响检查_20261006.md")

# ★ 与 manifest 行序一致的基线：伪标签索引的 paths 即 manifest 顺序
BASE_WEIGHT = 1.0


def base(p: str) -> str:
    """★ 跨平台约定：取 basename 必须先 replace('\\','/')"""
    return str(p).replace("\\", "/").rsplit("/", 1)[-1]


def main() -> int:
    if not os.path.exists(SRC):
        print(f"🛑 缺输入：{SRC}")
        return 1
    conf = json.load(open(SRC, encoding="utf-8"))
    idx = json.load(open(IDX, encoding="utf-8"))
    paths = idx["paths"]
    labels = np.asarray(idx["labels"])
    N = len(labels)

    groups = conf["groups"]
    print(f"[soft] 冲突组 = {len(groups)}（输入自文件 md5 判定）")

    # ---- 构造：每组选 1 个"下采样代表"，但**所有组内成员都保留**（权重 1/m）----
    # ★ 用户规格：组总权重=1 + 软目标均分 ⇒ 组内 m 张各持 1/m
    recs = []
    path2w = {}
    path2soft = {}
    cls_delta = collections.defaultdict(float)     # 每类 权重变化（相对原始 1.0/张）
    touched_cls = collections.Counter()
    for gi, g in enumerate(groups, 1):
        pl = g["paths"]
        lb = g["labels"]
        m = len(pl)
        w = BASE_WEIGHT / m
        # 组内**不同**标签集合（软目标在这些标签上均分）
        uniq = sorted(set(int(x) for x in lb))
        soft = {c: 1.0 / len(uniq) for c in uniq}
        for p, l in zip(pl, lb):
            path2w[base(p)] = w
            path2soft[base(p)] = soft
            cls_delta[int(l)] += (w - 1.0)      # 相对"每张 1.0"的增量
            touched_cls[int(l)] += 1
        recs.append({
            "group": gi, "n_members": m, "labels": uniq,
            "weight_each": round(w, 6), "weight_total": 1.0,
            "soft_target": {str(c): round(s, 6) for c, s in soft.items()},
            "members": [{"path": p, "label": int(l)} for p, l in zip(pl, lb)],
        })
        if gi % 300 == 0:
            print(f"  {gi}/{len(groups)} 组", flush=True)

    # ---- 各类权重影响检查（用户第⑤条）----
    print("\n[soft] 各类权重影响检查")
    rows = []
    for c in sorted(touched_cls):
        n_touched = touched_cls[c]
        # 该类总权重变化
        delta = cls_delta[c]
        # 该类受影响的样本占该类总量
        rows.append((c, n_touched, delta))
    # 全局
    n_aff = len(path2w)
    tot_w_before = float(n_aff)                      # 每张 1.0
    tot_w_after = sum(path2w.values())
    print(f"  受影响样本 = {n_aff} / {N} ({100*n_aff/N:.3f}%)")
    print(f"  这些样本的权重总和：{tot_w_before:.1f} → {tot_w_after:.4f} "
          f"（降为 {tot_w_after/tot_w_before*100:.2f}%）")
    print(f"  受影响类别数 = {len(rows)} / 750")
    print(f"  {'类别':>7}{'受影响张数':>12}{'权重变化':>12}")
    for c, n, d in rows[:12]:
        print(f"  {c:>7}{n:>12}{d:>12.4f}")
    if len(rows) > 12:
        print(f"  … 其余 {len(rows)-12} 类见报告")

    # ---- 落盘 ----
    out = {
        "desc": ("文件 md5 判定的冲突组：组总权重=1（组内每张 1/m），"
                 "软目标在组内不同标签间均分。★ 不在清单中的样本权重=1.0、标签=原硬标签"),
        "basis": "文件 md5（SHA-256 已逐字节复核）；不依赖任何模型预测/特征相似度",
        "n_groups": len(groups),
        "n_affected_samples": n_aff,
        "weight_total_before": tot_w_before,
        "weight_total_after": tot_w_after,
        "n_classes_touched": len(rows),
        "groups": recs,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[落盘] {OUT}")

    md = ["# 各类样本权重影响检查（冲突组去重权重 + 软目标）", "",
          "> 依据：文件 md5 判定的 1,877 个冲突组（SHA-256 已逐字节复核）",
          "> 处理：**组总权重 = 1**（组内每张 1/m），软目标在组内不同标签间均分", "",
          "## 一、总体", "",
          "| 项 | 值 |", "|---|---|",
          f"| 冲突组数 | {len(groups)} |",
          f"| 受影响样本 | {n_aff}（{100*n_aff/N:.3f}%） |",
          f"| 这些样本的权重总和 | {tot_w_before:.1f} → **{tot_w_after:.4f}**"
          f"（降为 {tot_w_after/tot_w_before*100:.2f}%） |",
          f"| 受影响类别数 | {len(rows)} / 750 |",
          f"| 全局权重总和 | {N:.1f} → **{N - tot_w_before + tot_w_after:.4f}**"
          f"（降 {(tot_w_before-tot_w_after)/N*100:.3f}%） |", "",
          "## 二、逐类权重变化（受影响的类）", "",
          "| 类别 | 受影响张数 | 权重变化（相对每张1.0） |", "|---|---|---|"]
    for c, n, d in rows:
        md.append(f"| {c:04d} | {n} | {d:+.4f} |")
    md += ["", "---", "",
           "★ **读法**：负值＝该类因冲突组被降权（这些张是**互相矛盾**的同一文件）。",
           "  未列出的类不受影响（权重仍为 1.0/张）。",
           "★ 本表只回答『权重被改了哪些类、改了多少』，**不回答『这样做对不对』**——",
           "  后者需榜分裁决（本地留出集读数未必代表线上）。"]
    with open(MD_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"[落盘] {MD_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
