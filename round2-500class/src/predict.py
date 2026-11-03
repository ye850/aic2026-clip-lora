"""推理并生成提交文件 pred_results.csv。

严格按赛题格式：
    xxxxxxxxxxxx.jpg, 0001
  * 每行两个字段：图片文件名、类别编号
  * 类别编号统一四位数字，不足前补 0
  * 文件名必须与测试集完全一致（含大小写与扩展名）
  * 文件固定命名 pred_results.csv，压成 zip 提交

单一模型、单一推理流程 —— 不做任何集成或投票（赛题明确禁止）。
"""
from __future__ import annotations

from pathlib import Path

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import torch
import torch.nn.functional as F

from data import (ClassMap, build_test_manifest, remap_to_official_ids,
                  write_submission)
from extract import get_or_build_features
from train_head import accuracy, estimate_test_prior, load_final


def run(device: str = "cuda",
        prior_mode: str = "none",
        head_path: Path | None = None,
        out_dir: Path | None = None,
        id_shift: int = 0,
        allow_unverified_map: bool = False) -> Path:
    """prior_mode: none | train_prior | em（长尾阶段用 em / train_prior）。

    ⚠️ **默认务必先用 "none"。** 实测（见下方内联表）表明：
    当测试集与训练集**同分布**时，`train_prior` 修正会稳定掉分，
    尾巴越长掉得越狠（Zipf s=1.5 时掉 0.33）。它只在「测试集比训练集更均衡」
    时才帮忙 —— 而这个前提官方没说，属未验证假设。
    **拿榜单分数比出差距之前不要开。**

    ``id_shift``            提交编号整体偏移（见 pipeline.py 的说明）
    ``allow_unverified_map`` 允许映射未经官方确认时仍写出提交
    """
    test_df = build_test_manifest()
    paths = test_df["path"].tolist()

    Xte = torch.as_tensor(get_or_build_features(paths, "feat_test", device), device=device)
    model = load_final(head_path, device=device)

    probs = accuracy(model, Xte, None, prior_q=None).cpu().numpy()

    if prior_mode != "none":
        # ★★ 使用前先读这段。
        #
        # 长尾场景下「乘 1/train_prior」看起来像是标准的 logit adjustment，
        # 但**它的成立条件很窄**：只有当「测试集分布比训练集更均衡」时才帮忙。
        # 若测试集与训练集同分布（或测试集更偏），它会**主动把分数扔掉**。
        #
        # 已在受控模拟上实测（K=200、d=16、等方差高斯 ⇒ 用真贝叶斯后验，
        # 即模型的判别能力已是最优，考察的纯粹是修正这一步）：
        #
        #   | 训练/测试同分布 | 最优 argmax | 乘 1/train_prior | 乘 1/EM |
        #   |---|---|---|---|
        #   | Zipf s=0.5 | 0.4924 | 0.4642 (**-0.0282**) | 0.4549 (-0.0375) |
        #   | Zipf s=1.0 | 0.6324 | 0.4678 (**-0.1646**) | 0.0273 (-0.6051) |
        #   | Zipf s=1.5 | 0.8074 | 0.4728 (**-0.3346**) | 0.0032 (-0.8042) |
        #
        # ⇒ 结论：**同分布时 `--prior train_prior` 必然掉分，越长的尾巴掉得越狠。**
        #   连 `--prior em` 也是负的：EM 能估准测试先验（实测与真值差 ~3e-3），
        #   但「估准」之后它又去掉了先验对正确推理有用的那部分信息 ——
        #   **EM 只在训练/测试确实不同分布时才有意义。**
        #
        # 那赛题到底同不同分布？**不知道**。官方只说「含长尾」，没说测试集
        # 是否与训练集同分布。这是个**未验证的假设**，不能当成已知条件。
        #
        # 所以这里的处理是：
        #   · 既保留机制（不同分布时它是救命的）
        #   · 又强制把「改变了多少预测」打出来，让效果可被观测
        #   · 默认值保持 "none" —— 在拿到榜单证据前不动它
        # 复赛每日只有 2 次提交，**第一次提交务必用 `--prior none`**，
        # 之后拿 `train_prior` 的结果去和它比，用量化差距决定保留哪个。
        tr_prior_path = C.CACHE_DIR / "train_prior.npy"
        if not tr_prior_path.exists():
            raise FileNotFoundError(
                f"缺少训练先验 {tr_prior_path}；请先跑 pipeline.py 的 scan 阶段")
        q_tr = np.load(tr_prior_path)
        if prior_mode == "em":
            q = estimate_test_prior(probs, q_tr)
            print("[predict] EM 估计测试先验，头/尾概率比 "
                  f"{q.max()/max(q.min(),1e-12):.2f}")
        else:
            q = q_tr
        before = probs.argmax(axis=1).copy()
        probs = probs / np.maximum(q, 1e-12)
        probs = probs / probs.sum(axis=1, keepdims=True)
        changed = int((probs.argmax(axis=1) != before).sum())
        print(f"[predict] 先验修正（{prior_mode}）改变 {changed}/{len(before)} 条预测"
              f"（{changed/max(len(before),1):.1%}）；"
              f"q 跨度 {q.max()/max(q.min(),1e-12):.1f}x")
        if changed == 0:
            print("[predict] ⚠️ 未改变任何预测 —— 这次修正等于没做。")

    pred_idx = probs.argmax(axis=1)

    # 内部下标 -> 官方四位类别编号
    import json
    meta = json.loads((C.CACHE_DIR / "train_meta.json").read_text(encoding="utf-8"))
    class_names: list[str] = meta["class_names"]
    class_map = ClassMap({k: int(v) + int(id_shift)
                          for k, v in meta["class_map"].items()})
    class_map.confidence = meta.get("class_map_confidence", "unknown")
    class_map.source = meta.get("class_map_source", "?") + \
        (f"（已整体偏移 {id_shift:+d}）" if id_shift else "")

    class_ids = remap_to_official_ids(pred_idx, class_names, class_map)

    # 一致性自检
    assert len(class_ids) == len(test_df), "预测数与测试集条数不一致"
    n_tok = len(set(map(str, test_df["img_name"])))
    if n_tok != len(test_df):
        print(f"[predict] ⚠️ 图片名有重复：{len(test_df)} 条 / {n_tok} 唯一")

    zip_path = write_submission(test_df["img_name"].tolist(), class_ids, out_dir,
                                class_map=class_map, class_names=class_names,
                                allow_unverified=allow_unverified_map)
    uniq, cnt = np.unique(class_ids, return_counts=True)
    print(f"[predict] 预测覆盖 {len(uniq)}/{len(class_names)} 类；"
          f"最频繁类占 {cnt.max()/len(class_ids):.1%}")
    return zip_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prior", default="none",
                    choices=["none", "train_prior", "em"])
    ap.add_argument("--id-shift", type=int, default=0)
    ap.add_argument("--allow-unverified-map", action="store_true")
    a = ap.parse_args()
    run(device=a.device, prior_mode=a.prior, id_shift=a.id_shift,
        allow_unverified_map=a.allow_unverified_map)
