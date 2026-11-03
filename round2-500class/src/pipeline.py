"""端到端流水线编排。

    scan     建立训练/测试清单与类别编号映射
    extract  提取并缓存 CLIP 特征（train / test）
    oof      K 折 OOF 概率 -> 估计噪声转移矩阵 T
    train    用 CL 权重 + 前向校正损失训练最终分类头
    predict  推理并生成 pred_results.csv(.zip)
    eval     若测试集标签可得（冒烟数据），报告准确率
    zeroshot 零样本基线（类别名为语义名时可用）

典型用法：
    python src/pipeline.py all                 # 一条龙
    python src/pipeline.py train --kind ce     # 消融：换成朴素交叉熵
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import torch

import noise as NZ
import train_head as TH
from data import (ClassMap, build_test_manifest, build_train_manifest,
                  write_submission)
from extract import build_transform, get_or_build_features

META_P = C.CACHE_DIR / "train_meta.json"


# ------------------------------------------------------------------ scan
def stage_scan() -> tuple:
    from data import audit_class_map, infer_class_map

    tr = build_train_manifest()
    class_names = tr.attrs["class_names"]
    dirs_names = sorted(tr["class_name"].unique())
    raw = infer_class_map([Path(x) for x in dirs_names])

    by_name = tr.groupby("class_name")["class_id"].first().to_dict()
    class_map = ClassMap({k: int(by_name[k]) for k in class_names})
    class_map.confidence = getattr(raw, "confidence", "unknown")
    class_map.source = getattr(raw, "source", "?")

    # ★ 把映射自检结果打印出来并写进 meta —— 编号错了就是全错，必须醒目
    issues = audit_class_map(class_map, len(class_names))
    if issues:
        print("[scan] ⚠️ 提交前请消除上述问题，否则可能得 0 分。")

    prior = np.bincount(tr["label"].to_numpy(), minlength=len(class_names)).astype(np.float64)
    prior = prior / prior.sum()
    np.save(C.CACHE_DIR / "train_prior.npy", prior)

    META_P.write_text(json.dumps({
        "class_names": class_names,
        "class_map": class_map,
        "class_map_confidence": class_map.confidence,
        "class_map_source": class_map.source,
        "class_map_issues": issues,
        "num_classes": len(class_names),
        "n_train": int(len(tr)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scan] 写出 {META_P}（{len(class_names)} 类）")
    return tr, class_names, class_map


def _load_meta() -> dict:
    if not META_P.exists():
        raise FileNotFoundError("请先运行 scan 阶段")
    return json.loads(META_P.read_text(encoding="utf-8"))


# --------------------------------------------------------------- extract
# --------------------------------------------------- 阶段数据隔离（重要）
# ⚠️⚠️ 规则原文：「复赛期间不得使用初赛的数据，半决赛期间不得使用初赛和复赛的数据。」
# 违反 = 取消成绩。因此换阶段时必须把**所有由上一阶段训练集派生**的产物清干净。
#
# 踩过的坑：早先的 --rebuild 只删 feat_train/feat_test 两个文件，漏掉了
# oof_probs.npy（412MB）、T.npy（转移矩阵）、train_meta.json（类别表）、
# train_prior.npy（类别先验）、head_*.pt（分类头）。
# 后果有两层，都很致命：
#   1. T.npy 没有任何 shape 自检 → 会被静默复用到新阶段，
#      等于用初赛的噪声结构去训练复赛模型；
#   2. train_meta.json 里存着初赛的 num_classes（500），
#      而复赛是 750 类 → 分类头规模错配，报错或静默出错。
#
# 这里定义「派生自训练集」的完整清单，供 --rebuild 与阶段切换检查共用。
STAGE_DERIVED_FILES = (
    "feat_train.npy",     # 训练集特征
    "feat_test.npy",      # 测试集特征（测试集也随阶段更换）
    "oof_probs.npy",      # OOF 概率（由训练集算出）
    "T.npy",              # 噪声转移矩阵（由训练集算出）
    "train_prior.npy",    # 类别先验（由训练集算出）
    "train_meta.json",    # 类别表 / num_classes（由训练集算出）
)


def clear_stage_cache(verbose: bool = True) -> list[str]:
    """清空所有「由当前阶段数据派生」的缓存，避免跨阶段数据混用。

    用于换阶段（初赛→复赛→半决赛）时，**必须调用**。
    返回被删除的文件名列表。
    """
    removed = []
    for n in STAGE_DERIVED_FILES:
        p = C.CACHE_DIR / n
        if p.exists():
            size = p.stat().st_size
            p.unlink()
            removed.append(n)
            if verbose:
                print(f"[rebuild] 删除 {n}（{size:,} 字节）")
    # 分类头一律在特征/元信息之后训练，故一并清除
    for p in sorted(C.CACHE_DIR.glob("head*.pt")):
        size = p.stat().st_size
        p.unlink()
        removed.append(p.name)
        if verbose:
            print(f"[rebuild] 删除 {p.name}（{size:,} 字节）")
    if verbose:
        print(f"[rebuild] 共清除 {len(removed)} 个阶段派生文件"
              f"（阶段隔离，防止违反『不得复用上一阶段数据』）")
    return removed


def stage_extract(device: str = "cuda", rebuild: bool = False):
    tr = build_train_manifest()
    te = build_test_manifest()
    if rebuild:
        clear_stage_cache()
        # ★ clear_stage_cache 会连 train_meta.json 一起删掉（它属于阶段派生文件，
        # 不清就有跨阶段复用的风险）。但 meta 里存着 class_map ——
        # 后续 train / predict / finetune_lora **都要用它把索引映射回官方编号**，
        # 缺了就直接 "请先运行 scan 阶段" 报错。
        #
        # 早先的实现没做这一步，于是 `extract --rebuild` 跑完接 `train` 必挂，
        # 用户得记得手动补一条 scan —— 而这种"文档里没写、忘了就报错"的依赖
        # 正是最容易在换阶段当天踩坑的地方。这里直接顺手重建，让 rebuild 自洽。
        print("[rebuild] 重建 train_meta.json（class_map 后续阶段必需）")
        stage_scan()
    else:
        # 非 rebuild 模式下也做一次「元信息与当前数据是否自洽」的检查：
        # 若 train_meta.json 的类别数与当前训练集类数不一致，说明换了阶段
        # 却忘了 --rebuild，此时**必须报错**，不能静默复用。
        meta_p = C.CACHE_DIR / "train_meta.json"
        if meta_p.exists():
            import json as _json
            old_k = _json.loads(meta_p.read_text(encoding="utf-8")).get("num_classes")
            new_k = int(tr.attrs.get("num_classes", 0))
            if old_k and new_k and old_k != new_k:
                raise SystemExit(
                    f"\n[stage-guard] 🛑 检测到阶段数据已更换但缓存未清理：\n"
                    f"[stage-guard]    缓存 train_meta.json 类别数 = {old_k}\n"
                    f"[stage-guard]    当前训练集类别数       = {new_k}\n"
                    f"[stage-guard] 规则禁止复用上一阶段数据（违者取消成绩）。\n"
                    f"[stage-guard] 请改跑：pipeline.py extract --rebuild\n")
    Xtr = get_or_build_features(tr["path"].tolist(), "feat_train", device)
    Xte = get_or_build_features(te["path"].tolist(), "feat_test", device)
    print(f"[extract] train {Xtr.shape} / test {Xte.shape}")


# ------------------------------------------------------------------- oof
def stage_oof(device: str = "cuda", folds: int | None = None, epochs: int = 25):
    tr = build_train_manifest()
    Xtr = torch.as_tensor(np.load(C.CACHE_DIR / "feat_train.npy"), device=device)
    y = torch.as_tensor(tr["label"].to_numpy(), device=device)
    K = int(y.max().item() + 1)

    oof_p = C.CACHE_DIR / "oof_probs.npy"
    if oof_p.exists():
        oof = np.load(oof_p)
        if oof.shape[0] == len(y):
            print(f"[oof] 复用 {oof_p.name} {oof.shape}")
        else:
            oof = TH.get_oof_probs(Xtr, y, K, folds or C.CV_FOLDS, device, epochs=epochs)
    else:
        oof = TH.get_oof_probs(Xtr, y, K, folds or C.CV_FOLDS, device, epochs=epochs)

    T = NZ.estimate_transition_matrix(oof, y.cpu().numpy(), K)
    np.save(C.CACHE_DIR / "T.npy", T)
    print(f"[oof] T 已保存（对角线均值 {np.diag(T).mean():.4f}）")


# ----------------------------------------------------------------- train
def stage_train(device: str = "cuda", kind: str | None = None,
                use_cl: bool | None = None, use_cb: str | None = None,
                epochs: int | None = None, tag: str = "main",
                t_strength: float | None = None):
    meta = _load_meta()
    K = meta["num_classes"]
    tr = build_train_manifest()
    Xtr = torch.as_tensor(np.load(C.CACHE_DIR / "feat_train.npy"), device=device)
    y = torch.as_tensor(tr["label"].to_numpy(), device=device)

    kind = kind or C.LOSS_KIND
    if use_cl is not None:
        C.USE_CL_REWEIGHT = use_cl
    if use_cb is not None:
        C.USE_CLASS_BALANCE = use_cb

    T = None
    if kind.startswith("forward") or C.USE_CL_REWEIGHT:
        tp = C.CACHE_DIR / "T.npy"
        if not tp.exists():
            raise FileNotFoundError(f"需要 {tp}；请先跑 oof 阶段（或 --no-cl）")
        T = np.load(tp)
        s = C.T_STRENGTH if t_strength is None else t_strength
        T = NZ.temper_transition_matrix(T, s)
        print(f"[train] T 收缩强度 = {s}（0=不校正，1=全量校正）")

    # ★ 训练产物直接落到「按 tag 命名」的文件，**绝不覆盖 head.pt**。
    #
    # 踩过的坑：原先的实现是先无条件写 head.pt、再在 tag != "main" 时复制一份
    # 副本。后果是**每跑一个备选版本都会把主版本覆盖掉** —— 跑完三轮消融后，
    # head.pt 变成了最后一轮的模型，已提交版本无法复现。
    # 现在改为：主版本写 head.pt，其余一律写 head_<tag>.pt，互不干扰。
    out_name = "head.pt" if tag == "main" else f"head_{tag}.pt"
    m = TH.train_final(Xtr, y, T, K, device, kind=kind,
                       epochs=epochs or C.HEAD_EPOCHS, log_prefix=f"[train/{tag}]",
                       save_name=out_name)
    print(f"[train/{tag}] 权重 -> {C.CACHE_DIR / out_name}")
    return m


# ---------------------------------------------------------------- scores
def _report_probs(probs: np.ndarray, tag: str):
    meta = _load_meta()
    K = meta["num_classes"]
    gt_p = C.DATA_DIR / "test_labels.csv"
    if not gt_p.exists():
        print(f"[{tag}] 测试集无标签（正式赛正常），跳过打分；"
              f"预测覆盖 {len(np.unique(probs.argmax(1)))}/{K} 类")
        return None
    import pandas as pd
    gt = pd.read_csv(gt_p)
    col = "class_id" if "class_id" in gt.columns else gt.columns[-1]
    name_col = "img_name" if "img_name" in gt.columns else gt.columns[0]
    te = build_test_manifest()
    m = dict(zip(gt[name_col].astype(str), gt[col].astype(int)))
    y_true = np.array([m[str(n)] for n in te["img_name"]], dtype=np.int64)

    names = meta["class_names"]
    cmap = {k: int(v) for k, v in meta["class_map"].items()}
    lut = np.array([cmap[n] for n in names], dtype=np.int64)
    y_pred_ids = lut[probs.argmax(1)]
    acc = float((y_pred_ids == y_true).mean())
    print(f"[{tag}] ★ 测试集准确率 = {acc:.4f}  ({len(y_true)} 条)")
    return acc


# --------------------------------------------------------------- predict
def stage_predict(device: str = "cuda", prior: str = "none",
                  head: str | None = None, id_shift: int = 0,
                  allow_unverified_map: bool = False,
                  out_dir=None, tta: str = "none"):
    # ★ 护栏：非 main 的备选头**不许**写进默认提交目录。
    #
    # 踩过的坑：`--tag cb0.9` 这类消融跑完后，`head_cb0.9.pt` 的预测结果会
    # **直接覆盖** `outputs/submission/pred_results.csv` —— 那个文件是真正的
    # 提交件，也是对照实验的评分依据。一旦被备选配置覆盖，要么误交错版本，
    # 要么事后无法复现某次分数来自哪个配置。
    # 现在强制备选 tag 写到 `outputs/submission_<tag>/`，主版本才用默认目录。
    if out_dir is None and head:
        out_dir = C.SUB_DIR.parent / f"submission_{head}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[predict/{head}] ⚠️ 备选头不写默认提交目录；输出 -> {out_dir}")

    if head:
        head_p = C.CACHE_DIR / f"head_{head}.pt"
    else:
        head_p = None
    te = build_test_manifest()
    Xte = torch.as_tensor(
        get_or_build_features(te["path"].tolist(), "feat_test", device), device=device)
    model = TH.load_final(head_p, device=device)
    probs = TH.accuracy(model, Xte, None, prior_q=None).cpu().numpy()

    # ---------------------------------------------------------- TTA（水平翻转）
    # 单一模型、对同一批输入做两种预处理后平均概率 —— **不是模型集成**，
    # 没有第二个模型、没有投票，符合"不得多模型集成/融合"的约束。
    #
    # ★ 必须断言"确实改变了预测"：如果翻转特征因为任何原因没真的翻转
    #   （最容易的是 extract 的降级分支漏传 transform），probs_f 会与 probs
    #   逐位相同，TTA 静默变成空操作 —— 那正是最该拦的假成功。
    if tta == "flip":
        Xte_f = torch.as_tensor(
            get_or_build_features(te["path"].tolist(), "feat_test_flip", device,
                                  transform=build_transform(flip=True)),
            device=device)
        probs_f = TH.accuracy(model, Xte_f, None, prior_q=None).cpu().numpy()
        before_tta = probs.argmax(1).copy()
        probs = 0.5 * (probs + probs_f)
        ch = int((probs.argmax(1) != before_tta).sum())
        print(f"[predict] TTA(flip) 概率平均生效：改变 {ch}/{len(before_tta)} 条"
              f"（{ch/max(len(before_tta),1):.1%}）")
        if ch == 0:
            raise SystemExit(
                "[predict] 🛑 TTA 没有改变任何预测 —— 翻转特征多半没真翻转。\n"
                "          检查 feat_test_flip.npy 是不是在加了 transform 之前生成的"
                "（删掉该缓存重跑）。")

    if prior != "none":
        q_tr = np.load(C.CACHE_DIR / "train_prior.npy")
        q = TH.estimate_test_prior(probs, q_tr) if prior == "em" else q_tr

        # ==============================================================
        # ★★ 先验合理性护栏 —— 拦「先验极端到会反向塌缩」的情况
        # ==============================================================
        # 踩过的坑（2026-09-21，差点提交出去）：
        #   本工程训练先验**近乎均匀**（熵比 0.9972、头尾比 49.6x），
        #   而无约束的 EM 迭代 200 轮后把头尾比放大到 **6.4e49 x**
        #   （最小类先验 1.9e-52）。随后 `p / q` 修正会**反向塌缩**：
        #   `p / 1e-52` 溢出到 5e48，于是**先验越小的类越必然胜出** ——
        #   被预测最多的第 728 类占 25.8% 的样本，而它的先验比均值低 8 个数量级。
        #   实测 `none` vs `em` 改动了 **61.5%** 的预测，全提交出去基本全错。
        #
        # ★ 判据为什么是「方向」而不是「头尾比」：
        #   第一版护栏我写的是「头尾比 > 1e4 就拦」。实测发现**那是错的** ——
        #   加了 min_ratio=0.1 夹子之后，em 的头尾比仍有 6.2e05，但方向是对的：
        #   胜出类（被预测最多的 5 个类）的先验均值是全体均值的 **5.37 倍**，
        #   即预测确实集中在**高先验**类上，符合先验修正该有的行为。
        #   头尾比大只说明「估计出的先验形状本身就尖」，**不是病态**。
        #   ⇒ 真正的病态特征是**方向反了**：胜出类的先验低于全体均值。
        #     所以这里直接量方向，而不是拿一个形状阈值去卡。
        if not np.isfinite(q).all() or q.min() <= 0:
            raise SystemExit(
                f"\n[predict] 🛑 先验向量非法（存在非有限值或非正值），拒绝用它修正。\n"
                f"[predict]    min={q.min():.3e} max={q.max():.3e}\n")

        # 预演一次修正，检查**方向**是否正确（不落地，只用于判定）
        _pre = probs / np.maximum(q, 1e-12)
        _pe = _pre.argmax(1)
        _u, _c = np.unique(_pe, return_counts=True)
        _top = _u[np.argsort(_c)[::-1][:5]]
        _dir = float(q[_top].mean() / max(q.mean(), 1e-300))
        if _dir < 0.5:
            raise SystemExit(
                f"\n[predict] 🛑 先验修正**方向反了**，已阻止 —— 这会产出错误提交。\n"
                f"[predict]    现象：修正后「被预测最多」的 5 个类，其先验均值\n"
                f"[predict]          只有全体的 {_dir:.3f} 倍（正常应 > 1）。\n"
                f"[predict]    含义：`p / q` 让**先验最小的类几乎必然胜出** ——\n"
                f"[predict]          这是先验极端（或数值发散）导致的**反向塌缩**。\n"
                f"[predict]    q 范围 [{q.min():.3e}, {q.max():.3e}]"
                f"，头尾比 {q.max()/max(q.min(),1e-300):.3e} x\n"
                f"[predict]    处理办法：\n"
                f"[predict]      1) 用 --prior em 时，检查 train_head.estimate_test_prior\n"
                f"[predict]         的 min_ratio 是否被改小（默认 0.1，单类调整 ≤10 倍）。\n"
                f"[predict]      2) 若训练先验本身就长尾，改用 --prior train_prior。\n")
        if _dir > 50:
            print(f"[predict] ⚠️ 先验修正方向正确但偏强：胜出类先验均值是全体的 "
                  f"{_dir:.1f} 倍（一般 1~10 倍较常见）。")
            print("[predict]    建议留意是否需要收紧 min_ratio。")

        # ★ 先验修正必须打印「实际改变了多少预测」。
        #
        # 这个模块踩过一回"写了但没生效"：`LOSS_KIND` 的默认值是不使用 T 的
        # `sce`，于是「前向校正」在默认路径下根本没跑，却被当成卖点写进自检。
        # 教训是——**谈某个模块的效果前，先确认它真的在跑**。
        # 对先验修正，最直接的判据就是「修正前后 argmax 有多少条变了」：
        # 若为 0，说明这次修正等于没做，无论先验从哪来。
        before = probs.argmax(1).copy()
        probs = probs / np.maximum(q, 1e-12)
        probs = probs / probs.sum(1, keepdims=True)
        changed = int((probs.argmax(1) != before).sum())
        q_str = (f"q 范围 [{q.min():.3e}, {q.max():.3e}]，比值 {q.max()/max(q.min(),1e-12):.1f}x")
        print(f"[predict] 先验修正（{prior}）生效：{q_str}；"
              f"预测改变 {changed}/{len(before)} 条（{changed/max(len(before),1):.1%}）")
        if changed == 0:
            print("[predict] ⚠️ 先验修正未改变任何预测 —— 等于没做。"
                  "若本次是指望它救长尾，请换 --prior em 或检查先验是否退化为均匀。")
        # ★ 改动量过大同样要警告：真正"救长尾"的修正通常只动一小部分预测，
        #   动了一半以上往往意味着先验出了问题（参见上面的塌缩事故）。
        elif changed / max(len(before), 1) > 0.5:
            print(f"[predict] ⚠️⚠️ 先验修正改变了 {changed/max(len(before),1):.1%} 的预测 —— "
                  "改动量异常大。")
            print("[predict]    真正救长尾的修正一般只动一小部分样本。"
                  "改动过半通常意味着先验失真（例如 EM 发散）。")
            print("[predict]    建议先跑 scripts/probe_prior_effect.py 核对先验形状，"
                  "再决定是否提交。")

    acc = _report_probs(probs, "predict")

    meta = _load_meta()
    names = meta["class_names"]
    class_map = ClassMap(
        {k: int(v) + int(id_shift) for k, v in meta["class_map"].items()})
    class_map.confidence = meta.get("class_map_confidence", "unknown")
    class_map.source = meta.get("class_map_source", "?") + \
        (f"（已整体偏移 {id_shift:+d}）" if id_shift else "")
    class_ids = np.array([class_map[n] for n in names],
                         dtype=np.int64)[probs.argmax(1)]
    write_submission(te["img_name"].tolist(), class_ids, out_dir=out_dir,
                     class_map=class_map, class_names=names,
                     allow_unverified=allow_unverified_map)
    return acc


# -------------------------------------------------------------------- all
def run_all(device: str = "cuda", **kw):
    t0 = time.time()
    stage_scan()
    stage_extract(device)
    stage_oof(device)
    stage_train(device)
    stage_predict(device, id_shift=kw.get("id_shift", 0),
                  allow_unverified_map=kw.get("allow_unverified_map", False))
    print(f"[all] 完成，用时 {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "extract", "oof", "train",
                                      "predict", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kind", default=None,
                    help="ce | sce | gce | forward_ce | forward_sce | forward_gce")
    ap.add_argument("--no-cl", action="store_true", help="关闭 CL 保真度降权")
    ap.add_argument("--cb", default=None, choices=["cb", "inv", "none"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--prior", default="none", choices=["none", "train_prior", "em"],
                    help="先验修正。⚠️ 默认 none：实测测试集与训练集%%%%同分布%%%%时"
                         "train_prior 修正必然掉分（Zipf s=1.5 时 -0.33），"
                         "只在测试集更均衡时才帮忙 —— 该前提未经验证。"
                         "要开请先拿一次榜单分数作对照，见 predict.py 内联说明。")
    ap.add_argument("--t-strength", type=float, default=None,
                    help="T 收缩强度 0..1（前向校正用；0=不校正）")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tta", default="none", choices=["none", "flip"],
                    help="测试时增强。flip=水平翻转后平均概率（单模型，非集成）。"
                         "首次使用会提一遍翻转特征（测试集约 1~2 分钟）")
    ap.add_argument("--id-shift", type=int, default=0,
                    help="提交编号整体偏移量。若文件夹是 0001..0500 而官方标签是 "
                         "0..499，用 --id-shift -1 纠正")
    ap.add_argument("--allow-unverified-map", action="store_true",
                    help="允许在类别映射未经官方确认时仍写出提交文件")
    a = ap.parse_args()

    print(f"[cfg] device={a.device} cuda={torch.cuda.is_available()}")
    if a.stage == "scan":
        stage_scan()
    elif a.stage == "extract":
        stage_extract(a.device, a.rebuild)
    elif a.stage == "oof":
        stage_oof(a.device, a.folds, a.epochs or 25)
    elif a.stage == "train":
        stage_train(a.device, a.kind, (False if a.no_cl else None), a.cb,
                    a.epochs, a.tag, a.t_strength)
    elif a.stage == "predict":
        stage_predict(a.device, a.prior, None if a.tag == "main" else a.tag,
                      id_shift=a.id_shift,
                      allow_unverified_map=a.allow_unverified_map,
                      tta=a.tta)
    else:
        run_all(a.device, id_shift=a.id_shift,
                allow_unverified_map=a.allow_unverified_map)


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    main()
