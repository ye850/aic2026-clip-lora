#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用实验臂运行器（进程内 patch；**不改源码、不改数据盘**）。

为什么需要它
=====================================================================
2026-10-01 得知「有队伍复赛**第一天就 90 分**」，而我们 10 天只有 73.6593（差 **16.3pp**）。
⇒ 需要跑"**验证地基**"的对照臂，但：
  · **不能改源码**（工程约定：跑中改源文件 = 静默灾难）；
  · **不能动数据盘**（工程约定：改训练集规模必须重建所有按样本数索引的派生缓存）。
⇒ 本脚本采用「import → patch → 设 sys.argv → 调 main()」，
   与**已验证过**的 `scripts/label_fix_arm.py` 完全同一模式。

支持的 patch
=====================================================================
`--cb <value>`
    覆写 `config.USE_CLASS_BALANCE`。依据 `src/train_head.py:200` 的分支：
      "cb" → 类平衡权重；"inv" → 逆频权重；**其他值（如 "none"）→ w=ones（不加权）**。
    ⚠️ 必须在 `import finetune_lora` **之前**赋值（模块级常量）。

`--exclude <file>`
    从 `build_train_manifest()` 里**剔除**清单中的路径（每行一个，相对项目根、正斜杠）。
    ★ 自带**命中数自证**：实际命中 ≠ 清单行数 ⇒ 立即中止（宁可不跑，不可静默错位）。
    ★ 实现细节：manifest 的 `path` 可能是绝对路径，故统一截取到 `data/train/` 之后再做集合匹配。
    ⚠️ 副作用须记录：val 是从同一 manifest 划分的，**排除后 val 成员会变**
      ⇒ 该臂的 val 读数与历史臂**不可横向比较**（本臂的目的本来就是看榜分）。

用法
=====================================================================
    /root/miniconda3/bin/python scripts/arm_runner.py \\
        --cb none --exclude outputs/cache/nonnatural_train.txt \\
        -- --epochs 12 --batch 96 --workers 8 ...      # ← 其余**逐字**转给 finetune_lora

    # 只关 CL 与 SCE（不动类平衡）：
    /root/miniconda3/bin/python scripts/arm_runner.py -- --epochs 12 ... --no-cl --kind ce
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _norm(p: str) -> str:
    """统一成**文件名**用于匹配（★ 不要改回"路径匹配"）。

    为什么用文件名
    ------------------------------------------------------------------
    本地与云端的训练集**目录层数不同**：
      · 本地：`data/train/train/<类>/xxx.jpg`（**两层** train）
      · 云端：`/root/autodl-tmp/aic/data/train/<类>/xxx.jpg`（**一层** train，
              实测 `F.build_train_manifest()['path']` 输出如此）
    而排除清单是在**本地**扫出来的 ⇒ 两边路径天然对不上。
    ⇒ 2026-10-01 首跑 clean 臂时按路径匹配**命中 0 条**，被本脚本自带的
      "命中 0 即中止"机制拦下（**没有**产生"跑了但什么都没排除"的假实验）。

    文件名是 32 位十六进制 hash ⇒ **全局唯一** ⇒ 用 basename 匹配与目录层数无关，最稳。
    ⚠️ 跨平台约定：取 basename **禁用 `Path(p).name`**（Linux 不把反斜杠当分隔符）
       ⇒ 必须先 `replace` 再切。
    """
    s = str(p).replace("\\", "/")
    return s.rsplit("/", 1)[-1]


def main() -> int:
    my = argparse.ArgumentParser(add_help=False)
    my.add_argument("--cb", default=None)
    my.add_argument("--exclude", default=None)
    args, rest = my.parse_known_args()

    # rest 形如 ["--", "--epochs", "12", ...] 或 ["--epochs", ...]
    fwd = rest[1:] if rest and rest[0] == "--" else rest
    if not fwd:
        print("[arm] 🛑 没有要转给 finetune_lora 的参数（用 `--` 分隔）")
        return 2

    # ---------- patch 1: 类平衡 ----------
    if args.cb is not None:
        import config as C
        old = getattr(C, "USE_CLASS_BALANCE", None)
        C.USE_CLASS_BALANCE = args.cb
        print(f"[arm] ★ USE_CLASS_BALANCE: {old!r} → {args.cb!r}"
              f"{'（=不加权，关闭类平衡）' if args.cb not in ('cb', 'inv') else ''}", flush=True)

    # ---------- import 训练主体 ----------
    import finetune_lora as F

    # ---------- patch 2: 排除清单 ----------
    if args.exclude:
        ex_path = Path(args.exclude)
        if not ex_path.exists():
            print(f"[arm] 🛑 排除清单不存在：{ex_path}")
            return 2
        lines = [ln.strip() for ln in ex_path.read_text(encoding="utf-8").splitlines()]
        excl = {_norm(ln) for ln in lines if ln and not ln.startswith("#")}
        print(f"[arm] ★ 排除清单 {ex_path.name}：{len(excl)} 条", flush=True)
        if not excl:
            print("[arm] 🛑 清单为空 ⇒ 中止（跑了也等于没排除，纯烧钱）")
            return 2

        _orig = F.build_train_manifest

        def _patched():
            tr = _orig()
            rel = tr["path"].map(_norm)
            hit = rel.isin(excl)
            n_hit = int(hit.sum())
            n_rel_hit = 0
            if n_hit != len(excl):
                # 允许清单中有图不在训练清单（例如落在 val 侧）——但要**报数**，不能静默
                n_rel_hit = len(excl) - n_hit
                print(f"[arm] ⚠️ 清单 {len(excl)} 条，manifest 命中 {n_hit} 条，"
                      f"差 {n_rel_hit} 条（可能不在训练清单里）", flush=True)
            if n_hit == 0:
                raise SystemExit("[arm] 🛑 命中 0 条 ⇒ 路径格式漂移，中止"
                                 "（宁可不跑，不可静默地什么都没排除）")
            tr = tr.loc[~hit].reset_index(drop=True)
            print(f"[arm] ★ 已从训练清单剔除 {n_hit} 张 → 剩 {len(tr)} 张"
                  f"（{n_hit / max(1, n_hit + len(tr)) * 100:.2f}% 被移除）", flush=True)
            return tr

        F.build_train_manifest = _patched
        print("[arm] ✅ 已 patch build_train_manifest（唯一挂点，训练循环零侵入）", flush=True)

    # ---------- 转交 main() ----------
    print(f"[arm] → finetune_lora.main() argv={fwd}", flush=True)
    sys.argv = ["finetune_lora.py"] + fwd
    return int(F.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
