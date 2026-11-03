"""多视图 TTA 版的提交生成：对**测试集**跑 N 个视图，概率平均，出 `pred_results`。

为什么必须单独一个脚本
----------------------
`finetune_lora.py --predict-only` 走的是**单视图**（`Resize(224) + CenterCrop(224)`）。
而 `tta_lora_eval.py` 已经在留出集上实测出：

| 组合 | 全体 acc | 低相似 acc | Δacc 配对 | McNemar p |
|---|---|---|---|---|
| base（单视图，= 当前提交口径） | 0.6997 | 0.5044 | 0 | — |
| safe（rc+flip，6 视图） | 0.7200 | 0.5264 | +0.0203 | 1.6e-13 |
| **safe+分辨率（8 视图）** | **0.7248** | **0.5338** | **+0.0251** | **1.4e-19** |

而且**增益在「低相似」列上没有被反转**（+0.0294 > 全体 +0.0251）
⇒ 是真泛化，不是"吃近似重复的便宜"。

★★ 关键：复赛提交的就是**我们自己跑出来的预测文件**（`pred_results.zip`），
所以这个增益是**直接写进提交里的**，不依赖评测方怎么推理 ——
这一点与"训练侧技巧"有本质区别（后者要赌评测环境）。

⚠️ 但**绝不能把留出集上的 +0.025 当成榜分预测**。测试集是另来源，
增益大概率会缩水。本脚本只负责把"已验证更好的推理方式"落地，
榜分仍以排行榜为准。

用法::

    # 先小样本冒烟（30 张，约 1 分钟）
    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/predict_tta.py --limit 30 --dry-run

    # 正式生成（8 视图，37,444 张，约 12 min）
    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/predict_tta.py \
        --lora outputs/cache/lora_full750_best.pt \
        --out-dir outputs/submission_tta
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C            # noqa: E402  ★ 必须早于 numpy/torch（设 BLAS 线程）
import numpy as np            # noqa: E402
import torch                  # noqa: E402

from data import ClassMap, build_test_manifest, write_submission  # noqa: E402

# ★★ 复用 TTA 评估脚本的视图定义与推理循环 —— **单一真源**。
#    在这里重写一遍的风险是"评估时量的是这套视图，提交时用的却是另一套"，
#    那样 8 分钟的评估就白做了（本项目已经在"低相似标尺"上踩过同类坑）。
from tta_lora_eval import (  # noqa: E402
    ALL_VIEWS, DEFAULT_VIEWS, V8_VIEWS, build_view_tf, infer, load_lora_model,
)

# 实测最优组合（15 视图）：safe（训练分布内，rc+flip）+ 分辨率那一路。
# ★ 直接用评估脚本的默认值，**不在这里重新抄一遍字符串** ——
#   抄一遍就会漂移，导致"评估时量的是这套视图、提交时用的却是另一套"。
#   （2026-09-22 实测：15 视图相对 8 视图 Δ全体 +0.0044 / Δ低相似 +0.0049，
#    配对 McNemar p=0.0053 ⇒ 显著，故把 15 视图设为默认。）
BEST_VIEWS = DEFAULT_VIEWS

# 语义化别名，避免每次都手打一长串
PRESETS = {
    "best": BEST_VIEWS,          # 实测最优（15 视图）
    "v8": V8_VIEWS,              # 上一轮 8 视图（= 已生成的 `submission_tta` 包，可复现）
    "safe": "base,flip," + ",".join(f"rc{i}" for i in range(8)),  # 只降方差、不引分布偏移
    "none": "base",              # 等价于原来的单视图（用于对照）
}


def resolve_views(spec: str) -> list:
    spec = PRESETS.get(spec.strip().lower(), spec)
    by_name = {v.name: v for v in ALL_VIEWS}
    out = []
    for nm in [s.strip() for s in spec.split(",") if s.strip()]:
        if nm not in by_name:
            raise SystemExit(f"[ptta] 🛑 未知视图 {nm!r}；可选："
                             f"{', '.join(by_name)}（或预设 {', '.join(PRESETS)}）")
        out.append(by_name[nm])
    if not out:
        raise SystemExit("[ptta] 🛑 视图列表为空")
    return out


def health_report(probs: np.ndarray, p_base: np.ndarray | None,
                  K: int, tag: str) -> dict:
    """提交前的健康检查。

    口径与项目既有的 `probe_prior_effect` 一致：**覆盖度**（有没有塌缩到少数类）
    与**分布形状**（有没有被压成均匀分布）。
    ★ 注意这些指标**不能替代准确率** —— 它们只证明"没塌"，不证明"更准"。
    """
    pred = probs.argmax(1)
    cnt = np.bincount(pred, minlength=K)
    covered = int((cnt > 0).sum())
    ratio = float(cnt.max() / max(cnt.mean(), 1e-9))
    ent = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum(1).mean())
    rep = {
        "tag": tag,
        "n": int(len(pred)),
        "covered_classes": covered,
        "coverage": covered / K,
        "empty_classes": int(K - covered),
        "pred_per_class_mean": float(cnt.mean()),
        "pred_per_class_max": int(cnt.max()),
        "pred_per_class_min": int(cnt.min()),
        "max_over_mean": ratio,
        "mean_entropy": ent,
        "mean_max_prob": float(probs.max(1).mean()),
    }
    if p_base is not None:
        rep["agree_with_base"] = float((pred == p_base.argmax(1)).mean())
    print(f"[ptta] ---- 健康检查（{tag}）----")
    print(f"[ptta] 预测 {rep['n']} 条；覆盖 {covered}/{K} 类"
          f"（覆盖率 {rep['coverage']:.4f}，空类 {rep['empty_classes']}）")
    print(f"[ptta] 每类预测数：均值 {rep['pred_per_class_mean']:.1f}、"
          f"最大 {rep['pred_per_class_max']}、最小 {rep['pred_per_class_min']}、"
          f"最大/均值 {ratio:.2f}x")
    print(f"[ptta] 平均熵 {ent:.3f}（均匀分布为 {np.log(K):.3f}）、"
          f"平均最大概率 {rep['mean_max_prob']:.3f}")
    if p_base is not None:
        print(f"[ptta] 与 base 单视图的预测一致率 {rep['agree_with_base']:.4f}"
              f"（TTA 只应在少数样本上改动预测）")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="多视图 TTA 提交生成")
    ap.add_argument("--lora", default=str(C.CACHE_DIR / "lora_full750_best.pt"))
    ap.add_argument("--views", default="best",
                    help="视图列表，或预设 best/safe/none")
    ap.add_argument("--out-dir", default=str(C.CACHE_DIR.parent / "submission_tta"))
    ap.add_argument("--batch", type=int, default=256)
    # ★★ 2026-09-24 新增：显式设备开关（默认 auto，行为与从前完全一致）。
    #   动机：云端那张 4090 在跑训练时只剩几百 MiB 显存，**GPU 出包根本插不进去**；
    #   而"等训练跑完再出包"要白等几小时。云端有 128 个逻辑核基本闲置，
    #   完全可以**用 CPU 出包** —— 既不与训练争显存，也不用排队。
    #   ⚠️ 走 CPU 时必须**在启动命令里**显式给 `OMP_NUM_THREADS`：
    #     `config.py` 用 `setdefault` 把 OMP/MKL/OPENBLAS 等锁到 1
    #     （那是为 DataLoader 多 worker 防线程超订设的），而 `setdefault`
    #     **不会覆盖 shell 里已有的值** ⇒ 命令行给值即可生效。
    #     不给的话前向就是单线程，慢一个数量级。
    #   ⚠️ 也别给太大：DataLoader 的 worker 是 fork 出来的、会继承这个值，
    #     `workers × OMP_NUM_THREADS` 一旦超过物理核数就是线程超订（更慢）。
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="推理设备：auto（默认）= 有 CUDA 就用 CUDA；cpu = 强制 CPU")
    # ★ 融合规则：默认算术平均 —— 历史一切结论都建立在它上面，**不许偷偷改默认值**。
    #   `power` = 幂平均（只取 argmax 时等价于 argmax(mean(p^t))）：
    #   t>1 叫锐化，抑制「发散/低置信」视图、突出「自信」视图。
    #   实测（留出集低相似列）：V15 在 t=4 见峰 0.5448（算术 0.5387），
    #   但 **t 的最优值依赖视图集**（V8 要 t≥8、V23 在 t=4 也不如 V15）
    #   ⇒ 因此 `--fusion-t` **不设默认值**，必须显式给，
    #     免得后来的人以为「4」是个通用常数。
    ap.add_argument("--fusion", default="arith", choices=["arith", "power"],
                    help="视图融合方式：arith=算术平均（默认/历史口径）；power=幂平均（锐化）")
    ap.add_argument("--fusion-t", type=float, default=0.0,
                    help="幂平均指数 t（仅 --fusion power 用；t>1 为锐化）。★ 无默认值")
    ap.add_argument("--workers", type=int, default=6,
                    help="推理是纯前向、瓶颈在 CPU 解码 ⇒ worker 有正收益"
                         "（与训练相反，见 finetune_lora 的 --workers 说明）。"
                         "★ 默认从 4 提到 6：出包一次是几十分钟，别省这点。"
                         "★ 改这个数%%%%不影响结果%%%%（视图用逐样本播种，与 worker 数无关）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 张（冒烟用）。⚠️ 与训练侧的 --limit 语义不同，"
                         "这里只截断测试集，不影响任何缓存")
    ap.add_argument("--dry-run", action="store_true",
                    help="只报健康检查，%%%%不写提交文件%%%%")
    ap.add_argument("--allow-unverified-map", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖 --out-dir 里已存在的 pred_results.zip（默认拒绝）")
    ap.add_argument("--save-probs", default="",
                    help="把平均后的概率矩阵存成 npy供后续分析")
    ap.add_argument("--probs-dtype", default="float16",
                    choices=["float16", "float32", "float64"],
                    help="★ 落盘精度。%%%%默认 float16 保持与已提交包一致%%%%"
                         "（复现路径不可改）；但 float16 在 1.0 附近分辨率仅 4.9e-4，"
                         "实测 78~99%% 的元素被冲成 0/1 ⇒ 任何基于置信度的分析都会失效。"
                         "做新分析时用 float32。")
    ap.add_argument("--head", default="",
                    help="改用外部头（train_head 产物）接在 LoRA 骨干后")
    a = ap.parse_args()

    if a.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = a.device
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("[ptta] 🛑 指定了 --device cuda，但本机没有可用 CUDA")
    if device == "cpu":
        print(f"[ptta] ⚠️ CPU 推理：OMP_NUM_THREADS="
              f"{os.environ.get('OMP_NUM_THREADS')!r}、"
              f"torch.get_num_threads()={torch.get_num_threads()}、"
              f"workers={a.workers}")
    views = resolve_views(a.views)
    print(f"[ptta] device={device}；视图 {len(views)} 个："
          f"{', '.join(v.name for v in views)}")

    meta_p = C.CACHE_DIR / "train_meta.json"
    if not meta_p.exists():
        raise SystemExit("[ptta] 🛑 缺少 train_meta.json，请先跑 scan 阶段")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    K = int(meta["num_classes"])
    names = meta["class_names"]

    te = build_test_manifest()
    paths = te["path"].tolist()
    img_names = te["img_name"].tolist()
    if a.limit:
        paths, img_names = paths[:a.limit], img_names[:a.limit]
        print(f"[ptta] ⚠️ --limit {a.limit} ⇒ 只跑 {len(paths)} 张（冒烟）")
    print(f"[ptta] 测试集 {len(paths)} 张 / {K} 类")

    ckpt = Path(a.lora)
    if not ckpt.exists():
        raise SystemExit(f"[ptta] 🛑 找不到检查点 {ckpt}")
    model = load_lora_model(ckpt, K, device)
    head_ckpt = Path(a.head) if a.head else None
    if head_ckpt is not None:
        print(f"[ptta] 使用外部头 {head_ckpt}")

    # ---------------------------------------------------------- 逐视图累加
    # 用 float64 累加：单视图概率已经有 1e-7 量级的小值，
    # 8 个视图相加时 float32 的相对误差会累积到可见量级。
    # ★ 且幂平均要算 p^t：float32 下 p 的 1e-7 量级小值 ^4 就贴近下溢边界，
    #   所以**先升到 float64 再取幂**，而不是先幂后升。
    if a.fusion == "power":
        if a.fusion_t <= 0:
            raise SystemExit("[ptta] 🛑 --fusion power 必须同时给 --fusion-t > 0"
                             "（t=4 是 V15 上的实测最优点；t 依赖视图集，别照抄）")
        if a.fusion_t > 30:
            print(f"[ptta] ⚠️ --fusion-t = {a.fusion_t} 偏大：float64 下 1e-7 的 t 次幂会"
                  f"下溢到 0 ⇒ 该视图的低概率部分整段消失、只剩 argmax 附近的几项。"
                  f"（不是错误，但你要知道自己测的是什么。）")
        print(f"[ptta] 融合规则 = 幂平均 t={a.fusion_t}（锐化）")
    else:
        print("[ptta] 融合规则 = 算术平均（历史口径）")

    # ---------------------------------------------------------- 覆盖保护（先拦，再跑）
    # ★ 一个包是二十几分钟的 GPU。等跑完才发现目录被覆盖，那二十几分钟就白烧了；
    #   更糟的是历史包（`submission_tta15/` 等）要留作对照，不能被静默盖掉。
    # ★ `--dry-run` 不写任何文件 ⇒ 不该被这条拦（冒烟测试要能在原目录上跑）。
    out_dir = Path(a.out_dir)
    if not a.dry_run and (out_dir / "pred_results.zip").exists() and not a.force:
        raise SystemExit(
            f"[ptta] 🛑 {out_dir} 里已经有 pred_results.zip —— 拒绝覆盖。\n"
            f"        历史包要留作对照，请换一个 --out-dir"
            f"（例如 `outputs/submission_tta15_t4`）；\n"
            f"        确实要重写同一个包，再加 --force。")
    if a.dry_run:
        print("[ptta] --dry-run：覆盖保护已豁免（不写文件）")

    acc = None
    p_base = None
    per_view_acc: list[tuple[str, np.ndarray]] = []
    t0 = time.time()
    for i, v in enumerate(views, 1):
        tv0 = time.time()
        p = infer(model, paths, build_view_tf(v), device, a.batch, a.workers,
                  v.name, seed_base=(v.seed_base if v.rc else None),
                  head_ckpt=head_ckpt)
        if p.shape[0] != len(paths):
            raise SystemExit(f"[ptta] 🛑 视图 {v.name} 返回 {p.shape[0]} 行，"
                             f"期望 {len(paths)} ⇒ 顺序错了，中止")
        contrib = (np.power(p.astype(np.float64), a.fusion_t)
                   if a.fusion == "power" else p.astype(np.float64))
        if acc is None:
            acc = contrib
        else:
            acc += contrib
        if v.name == "base":
            p_base = p
        per_view_acc.append((v.name, p))
        print(f"[ptta] 视图 {i}/{len(views)} {v.name} 完成"
              f"（{time.time() - tv0:.0f}s）", flush=True)

    # ★ 按行归一化，而不是除以视图数：
    #   算术平均下两者等价（每行本来就和为 1）；幂平均下 `mean(p^t)` **不再是分布**，
    #   而健康度报告要算归一化熵/最大概率 ⇒ 必须先还原成一个分布。
    #   除以逐行常数**不改变 argmax**，所以对最终预测无影响，只是让报告仍有意义。
    row = acc.sum(1, keepdims=True)
    if np.any(row <= 0):
        raise SystemExit(f"[ptta] 🛑 有 {int((row <= 0).sum())} 行融合后权重全 0"
                         f"（--fusion-t 太大导致下溢？）⇒ 中止")
    probs = (acc / row).astype(np.float32)
    print(f"[ptta] {len(views)} 视图融合完成，总用时 {(time.time() - t0) / 60:.1f} min")

    # ---------------------------------------------------------- 逐视图一致性
    if len(per_view_acc) > 1 and p_base is not None:
        print("[ptta] 各视图与 base 的预测一致率（仅供判断是否真有分歧）：")
        for nm, p in per_view_acc:
            ag = float((p.argmax(1) == p_base.argmax(1)).mean())
            print(f"[ptta]   {nm:<10}{ag:.4f}")

    rep = health_report(probs, p_base, K,
                        f"{len(views)} 视图 · "
                        + (f"幂平均 t={a.fusion_t}" if a.fusion == "power" else "算术平均"))

    if a.save_probs:
        # ★ 2026-10-06：精度改为可选，默认 float16 保持复现路径不变。
        #   float16 在 1.0 附近分辨率 2^-11≈4.9e-4 ⇒「真置信 0.9998」与
        #   「真置信 1.0000」落盘后不可分（实测 probs_train.npy 78.57% 饱和）。
        _dt = {"float16": np.float16, "float32": np.float32,
               "float64": np.float64}[a.probs_dtype]
        np.save(f"{a.save_probs}.npy", probs.astype(_dt))
        print(f"[ptta] 已存 {a.save_probs}.npy {probs.shape}（{a.probs_dtype}）")

    if a.dry_run:
        print("[ptta] --dry-run ⇒ 不写提交文件")
        return 0

    # ---------------------------------------------------------- 写提交
    # ★ 与 `finetune_lora.predict` 走同一套写入路径（含 class_map 闸门），
    #   不另起一套 —— 提交文件的编号映射是全项目最高风险的一环。
    class_map = ClassMap({k: int(v) for k, v in meta["class_map"].items()})
    class_map.confidence = meta.get("class_map_confidence", "unknown")
    class_map.source = meta.get("class_map_source", "?")
    class_ids = np.array([class_map[n] for n in names],
                         dtype=np.int64)[probs.argmax(1)]
    write_submission(img_names, class_ids, out_dir=out_dir, class_map=class_map,
                     class_names=names, allow_unverified=a.allow_unverified_map)
    print(f"[ptta] ★ 提交文件已写到 {out_dir}")
    print(f"[ptta] 下一步：`scripts/check_submission.py` 校验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
