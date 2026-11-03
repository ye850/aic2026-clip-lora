"""导出"本机/云端各自算出的 val 名单" —— 只读、零副作用、两侧通用。

【为什么需要它】
已确认（云端日志 vs 本机）：数据规模 148695/750 类、val 大小 7537、缓存"缺失回落 0 张"、
`src/finetune_lora.py` md5 逐位一致 —— **全部相同**。
但同一批权重在两边量的 val acc 差 6.1~6.8pp，且**只发生在新代**（旧代吻合到 0.2~0.5pp）。

统计上"两个同尺寸、同类分布的分层随机子集"之间的 acc 差只有 σ≈0.68pp（10σ 才够 6.8pp），
所以必须先把一个事实钉死：**两边的 val 到底是不是同一批样本？**

机制上它**应该**不同：`build_train_manifest()` 用 `d.iterdir()`（**不排序**）建行序，
而 `rng.choice(idx, size=want)` 抽的是**位置** ⇒ 行序一变，成员就变。
自估 overlaps：148695/750 类、每类约 198 张、每类抽 10 张 ⇒ 两边每类重叠期望仅 ~0.5 张。

【本脚本输出】
- val 张数
- **集合 md5**（basename 排序后，与顺序无关）⇒ 相同即"完全同一批"
- 行序 md5（按 df 行序）⇒ 反映占位顺序
- 全量训练集文件名集合 md5（确认数据本身是同一批）
- 完整 val 名单文件（供逐条比对重叠率）

【用法】
    python scripts/dump_val_names.py --out outputs/_val_names_local.txt
    # 云端（无卡）：
    AIC_ROOT=/root/autodl-tmp/aic python scripts/dump_val_names.py --out outputs/_val_names_remote.txt
    # 对比（本机）：
    python scripts/dump_val_names.py --compare outputs/_val_names_local.txt <remote.txt>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _base(p: str) -> str:
    """跨平台 basename —— Linux 的 Path 不把 `\\` 当分隔符（本项目踩过）。"""
    return str(p).replace("\\", "/").rstrip("/").split("/")[-1]


def collect() -> dict:
    import numpy as np
    from data import build_train_manifest
    import compare_lora_vs_frozen as CLF

    tr = build_train_manifest()
    K = int(tr.attrs.get("num_classes", 0))
    paths = tr["path"].tolist()
    labels = tr["label"].to_numpy().astype(np.int64)
    mask = CLF.lora_val_mask(labels, K)          # 逐字复刻训练侧切分
    idx = np.where(mask)[0]
    names = [_base(paths[i]) for i in idx]
    all_names = sorted({_base(p) for p in paths})

    return {
        "root": str(ROOT),
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "?"),
        "n_train_total": int(len(labels)),
        "n_classes": K,
        "n_val": int(len(idx)),
        "val_set_md5": hashlib.md5("\n".join(sorted(names)).encode()).hexdigest(),
        "val_order_md5": hashlib.md5("\n".join(names).encode()).hexdigest(),
        "train_set_md5": hashlib.md5("\n".join(all_names).encode()).hexdigest(),
        "val_names": names,
    }


def compare(fa: Path, fb: Path) -> int:
    def _rd(p: Path):
        # 跳过以 `#` 开头的指纹注释行（脚本自己写的第一行）
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#")]
    A, B = _rd(fa), _rd(fb)
    sa, sb = set(A), set(B)
    inter = sa & sb
    print("=" * 96)
    print("val 名单比对")
    print("=" * 96)
    print(f"  A(local) : {len(A)} 张   文件 {fa.name}")
    print(f"  B(remote): {len(B)} 张   文件 {fb.name}")
    print()
    print(f"  集合 md5 : {hashlib.md5(chr(10).join(sorted(A)).encode()).hexdigest()[:16]}…  vs  "
          f"{hashlib.md5(chr(10).join(sorted(B)).encode()).hexdigest()[:16]}…")
    print(f"  交集     : {len(inter)} 张")
    print(f"  重叠率   : {len(inter)/max(1,len(sa)):.2%}  (相对 A)   /   "
          f"{len(inter)/max(1,len(sb)):.2%}  (相对 B)")
    print(f"  A 独有   : {len(sa-sb)}    B 独有: {len(sb-sa)}")
    print()
    if sa == sb:
        print("  ⇒ ★★ 两个 val 集合**完全相同** ⇒ 6.8pp **不可能**来自'换了留出集'，")
        print("     必须转向『图源内容』或『评估过程本身』。")
    elif len(inter) / max(1, len(sa)) > 0.90:
        print("  ⇒ 高度重叠（>90%）⇒ 与'完全同一批'近似，6.8pp 同样不能由换子集解释。")
    else:
        print(f"  ⇒ 重叠仅 {len(inter)/max(1,len(sa)):.1%} ⇒ 两边 val 是**两批不同样本**。")
        print("     ★ 但注意：两个同尺寸同类分布的分层随机子集，acc 差的标准差只有 ~0.68pp")
        print("       （见 scripts/val_subset_noise.py 的 bootstrap）⇒ **6.8pp = 10σ，仍不能由换子集解释**。")
        print("       ⇒ 若确认不同，下一步只能查『两边 val 样本的图内容（320 缓存 md5）』。")
    print("=" * 96)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A_TXT", "B_TXT"))
    a = ap.parse_args()

    if a.compare:
        return compare(Path(a.compare[0]), Path(a.compare[1]))

    d = collect()
    print(f"[val] root={d['root']} host={d['host']}")
    print(f"[val] 全量 {d['n_train_total']} 张 / {d['n_classes']} 类")
    print(f"[val] 验证 {d['n_val']} 张")
    print(f"[val] val 集合 md5（baename 排序）: {d['val_set_md5']}")
    print(f"[val] val 行序 md5               : {d['val_order_md5']}")
    print(f"[val] 全量集合 md5                : {d['train_set_md5']}")
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 第一行放指纹，便于不拉全名单也能比
        head = (f"# host={d['host']} n_val={d['n_val']} "
                f"set_md5={d['val_set_md5']} order_md5={d['val_order_md5']} "
                f"train_set_md5={d['train_set_md5']}")
        p.write_text(head + "\n" + "\n".join(d["val_names"]) + "\n", encoding="utf-8")
        print(f"[val] 已写 {p}（{p.stat().st_size} B）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
