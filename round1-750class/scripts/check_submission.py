"""提交文件格式核验（正式提交前必跑）。

对照赛题要求逐条检查：
  1. 文件必须命名为 pred_results.csv，且压缩为 zip；
  2. 每行恰好两个字段：图片文件名、类别编号；
  3. 类别编号为四位数字、不足前补 0；
  4. 图片文件名与测试集**完全一致**（含大小写与扩展名）；
  5. 行数与测试集一致、无重复、无缺失。

    python scripts/check_submission.py outputs/submission/pred_results.zip
"""
from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from data import build_test_manifest  # noqa: E402

FOUR_DIGIT = re.compile(r"^\d{4}$")
ok_all = True


def fail(msg: str) -> None:
    global ok_all
    ok_all = False
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def load_rows(src: Path) -> list[tuple[str, str]]:
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            print(f"  zip 内含 {len(names)} 个条目：{names}")
            target = [n for n in names if n.endswith(C.SUB_CSV_NAME)]
            if not target:
                fail(f"zip 内未找到 {C.SUB_CSV_NAME}")
                return []
            if len(target) > 1:
                fail(f"zip 内有多个 {C.SUB_CSV_NAME}")
            raw = z.read(target[0]).decode("utf-8-sig", errors="replace")
    else:
        raw = src.read_text(encoding="utf-8-sig", errors="replace")

    rows = []
    for ln, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            fail(f"第 {ln} 行字段数 = {len(parts)}（应为 2）：{line[:80]}")
            continue
        rows.append((parts[0], parts[1]))
    return rows


def main() -> int:
    # 2026-10-06 补：本脚本原来直接读 `sys.argv[1]` ⇒ `--help` 会被当成文件名，
    # 报「文件不存在：--help」。交付文档里把它列为第一道自检命令，
    # 评委第一个敲的就是 --help ⇒ 必须能正常响应。
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__ or __file__)
        print("用法：python scripts/check_submission.py [pred_results.zip]")
        print("  不带参数时默认核验 outputs/submission/pred_results.zip")
        return 0
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        C.SUB_DIR / (C.SUB_CSV_NAME.replace(".csv", "") + ".zip"))
    print(f"=== 核验提交文件：{src} ===")
    if not src.exists():
        print(f"  [FAIL] 文件不存在：{src}")
        return 2

    rows = load_rows(src)
    print(f"  解析出 {len(rows)} 行")

    # 3) 四位数字
    bad_fmt = [r for r in rows if not FOUR_DIGIT.match(r[1])]
    if bad_fmt:
        fail(f"{len(bad_fmt)} 行类别编号不是四位数字，例：{bad_fmt[:3]}")
    else:
        print("  [OK] 类别编号均为四位数字")

    # 5) 与测试集比对
    try:
        te = build_test_manifest()
        expect = list(map(str, te["img_name"]))
        got = [r[0] for r in rows]
        if len(got) != len(expect):
            fail(f"行数不符：提交 {len(got)}，测试集 {len(expect)}")
        dup = len(got) - len(set(got))
        if dup:
            fail(f"存在 {dup} 个重复文件名")
        else:
            print("  [OK] 无重复文件名")
        miss = set(expect) - set(got)
        extra = set(got) - set(expect)
        if miss:
            fail(f"缺失 {len(miss)} 个测试文件名，例：{sorted(miss)[:3]}")
        if extra:
            fail(f"多出 {len(extra)} 个非测试文件名，例：{sorted(extra)[:3]}")
        if not miss and not extra and len(got) == len(expect):
            print("  [OK] 文件名与测试集完全一致")

        # 顺序提示（多数评测不要求顺序，但一致更稳）
        if len(got) == len(expect) and got != expect:
            warn("行顺序与测试清单不同（通常可接受，但建议按测试清单顺序输出）")

        # 类别覆盖
        ids = {int(r[1]) for r in rows if FOUR_DIGIT.match(r[1])}
        print(f"  [信息] 预测覆盖 {len(ids)} 个不同类别编号")
        cm = C.DATA_DIR / "class_map.csv"
        if cm.exists():
            with open(cm, encoding="utf-8-sig") as f:
                official = {int(r["class_id"]) if "class_id" in r else int(r[list(r)[1]])
                            for r in csv.DictReader(f)}
            unknown = ids - official
            if unknown:
                fail(f"出现官方类别表之外的编号 {len(unknown)} 个，例：{sorted(unknown)[:5]}")
            else:
                print("  [OK] 全部类别编号在官方类别表内")
    except Exception as e:  # noqa: BLE001
        warn(f"无法与测试集比对（{type(e).__name__}: {e}）；请确认数据已就位")

    print("=== 结论：" + ("全部通过，可提交" if ok_all else "存在问题，勿提交") + " ===")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
