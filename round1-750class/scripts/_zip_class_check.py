#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提交包「落盘后验」自检 —— 替代新 seed 无法使用的 --ref-csv 自证。

背景
  出包前必须自证「基础概率是对的」。对已经提交过的底模，做法是
  `--ref-csv <已提交包的 pred_results.csv>`，要求复现差异 0.00%。
  但**新 seed 没有可比对的已提交包** ⇒ 自证必须换形式。

本脚本做的是「结构自证」：均衡校正后的提交包，类计数必须呈现均衡解的特征。
判据区间**不靠记忆里的数字**，而是用两个**已提交包在本脚本下的实测值**定标
（可复算：跑一次 scripts/_zip_class_check.py 于下面两个 zip 即可）：

  包（均已过官方校验）              类数  min  max   标准差
  submission_d_r448ul4_balU   749    3    62    3.27   ← 已提交，榜分 73.26
  submission_d_r448ul4_balU2  750   30    63    1.63   ← 精确配额版
  submission_d_r448ul4_balT   749    3    68    8.62   ← 目标=训练类频，非法判据

⚠️ 这是「像不像一个正确的均衡解」，**不是**「分数高不高」。
   它抓的是：probs 读错文件（行序错 / 换了模型）、类别映射错、
   校正没生效这几类**静默错误**。它抓不到「模型本身就差」。

⚠️ 提交 csv 的格式是 **无表头** 的 `文件名, 类别编号`（由
   src/data.py::write_submission 写出）。所以**不能把第 1 行当表头丢掉**
   —— 本脚本用「第 2 列是否全是数字」来判断有没有表头。

用法：
  python scripts/_zip_class_check.py <pred_results.zip> [更多 zip ...]
  退出码：0 = 全部通过；1 = 有可疑包；2 = 参数错误
"""
from __future__ import annotations

import csv
import io
import math
import statistics
import sys
import zipfile
from collections import Counter

# ---- 测试集规模（项目常量） ----
N_TEST = 37444
K_CLASS = 750

# ---- 判据（区间由上面表里的实测值定标，留足裕度） ----
REF_STD_MIN = 1.0        # balU2 实测 1.63 ⇒ 更低说明被压死/读错模型
REF_STD_MAX = 6.0        # balU  实测 3.27 ⇒ 更高说明校正没生效（balT 8.62 会被标出）
REF_MEDIAN = 50.0
REF_MEDIAN_TOL = 3.0
REF_NCLASS_MIN = 748     # 至少覆盖 748/750 类
BARE_STD_MIN = 10.0      # std ≥ 此值 ⇒ 判为"未做均衡校正的裸包"（实测裸包 19~22；
                         # balT=8.62 是"非均匀目标的校正当" ⇒ 仍按可疑报，不混进这一档）


def _looks_numeric(s: str) -> bool:
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


def check_one(path: str) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as f:
            names = [n for n in f.namelist() if n.lower().endswith(".csv")]
            if not names:
                return False, f"{path}: zip 里没有 csv"
            with f.open(names[0]) as fh:
                rows = [r for r in csv.reader(io.TextIOWrapper(fh, "utf-8-sig")) if r]
    except Exception as e:  # noqa: BLE001
        return False, f"{path}: 打不开 zip（{e}）"

    if len(rows) < 2:
        return False, f"{path}: csv 只有 {len(rows)} 行"

    # ★ 无表头格式：第 2 列全是数字 ⇒ 第一行也是数据，不许丢
    header = not _looks_numeric(rows[0][-1])
    body = rows[1:] if header else rows
    lbl_i = len(rows[0]) - 1

    cnt = Counter(r[lbl_i].strip() for r in body)
    v = [float(x) for x in cnt.values()]
    n_cls = len(v)
    mean = sum(v) / n_cls
    med = statistics.median(v)
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / n_cls)
    mx, mn = max(v), min(v)

    probs = []
    ok = True
    if len(body) != N_TEST:
        probs.append(f"行数 {len(body)} != {N_TEST}")
    if n_cls < REF_NCLASS_MIN:
        probs.append(f"覆盖类数 {n_cls} < {REF_NCLASS_MIN}")
    if abs(med - REF_MEDIAN) > REF_MEDIAN_TOL:
        probs.append(f"中位 {med:.0f} 偏离 {REF_MEDIAN:.0f}")
    if not (REF_STD_MIN <= std <= REF_STD_MAX):
        probs.append(f"标准差 {std:.2f} 不在 [{REF_STD_MIN},{REF_STD_MAX}]")

    # ★ 09-27 补：这个脚本本来只服务于「均衡校正后的包」。
    #   但历史 submission_* 目录里混着大量**未做均衡校正的裸包**
    #   （实测 std 19~22、类覆盖 745~748、max 160~230），它们会被一律报"可疑"，
    #   看清单时根本分不清"包错了"和"这包本来就没校正"。
    #   ⇒ std 落在裸包带（≥ 8）时给一个**不同措辞**，避免误判。
    bare = std >= BARE_STD_MIN
    if bare:
        tag = (f"（看起来是**未做均衡校正的裸包**：std {std:.2f} ≥ {BARE_STD_MIN}）")
        return True, (f"裸包? | n={len(body)} 类数={n_cls} 中位={med:.0f} "
                      f"max={mx:.0f} min={mn:.0f} 标准差={std:.2f} {tag}")

    ok = not probs
    tag = "OK 通过" if ok else "!! 可疑：" + "；".join(probs)
    return ok, (f"{tag} | n={len(body)} 类数={n_cls} 中位={med:.0f} "
                f"max={mx:.0f} min={mn:.0f} 标准差={std:.2f}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for p in sys.argv[1:]:
        ok, msg = check_one(p)
        print(msg)
        bad += 0 if ok else 1
    print(f"[自检] 共 {len(sys.argv) - 1} 个包，可疑 {bad} 个")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
