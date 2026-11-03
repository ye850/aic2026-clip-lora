"""数据集扫描、类别编号映射与稳健图像读取。

数据组织（按赛题说明）：训练数据按类别文件夹组织；测试集给出待预测图像列表，
标签不公开。因此本模块要解决两件事：
  1. 建立 train 清单 (img_path, label_idx, class_name)；
  2. 建立「类别名 -> 官方四位类别编号」的映射 —— 这是提交能否被正确计分的关键。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import pandas as pd
from PIL import Image, ImageFile


# 部分图片文件可能被截断；赛题说明确认 Pillow 仍可读取。
# 打开此开关让 Pillow 容忍截断文件，避免整批中断。
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_train_root(root: Path | None = None) -> Path:
    """定位真实训练根目录（容忍压缩包多套一层目录）。"""
    root = Path(root or C.TRAIN_ROOT)
    if not root.exists():
        raise FileNotFoundError(
            f"未找到训练目录：{root}\n"
            "请把赛题训练数据放到 data/train/ 下（每类一个子文件夹）。"
        )
    # 若 root 下没有子目录但有唯一的子目录，下钻
    for _ in range(3):
        subs = [d for d in root.iterdir() if d.is_dir()]
        if len(subs) == 1 and not any(root.glob("*.[jJ][pP][gG]")):
            root = subs[0]
        else:
            break
    return root


def list_class_dirs(train_root: Path) -> list[Path]:
    return sorted([d for d in train_root.iterdir() if d.is_dir()],
                  key=lambda p: p.name)


_LEADING_NUM = re.compile(r"^\s*(\d+)")


class ClassMap(dict):
    """带元信息的类别映射（confidence / source 挂在属性上，不污染键值）。"""

    confidence: str = "unknown"
    source: str = "?"


def infer_class_map(class_dirs: list[Path], out_csv: Path | None = None) -> dict[str, int]:
    """推断「类别文件夹名 -> 四位类别编号」。

    ⚠️⚠️ 这是整个流程里**唯一"错了就全错"**的环节：类别编号映射错一位，
    提交 CSV 的每一行标签都是错的，线上得分直接归零。
    因此本函数除了返回映射，还会把「映射可信度」写进 class_map 的
    ``.confidence`` 属性（dict 的子类），供 write_submission 做硬拦截。

    优先级：
      1. data/class_map.csv（官方映射文件，最可靠）
      2. 文件夹名前缀数字（如 "0001_robin" / "1" / "0012"）
      3. 按文件夹名排序后顺序编号 1..N  ——  ⚠️ 几乎一定不等于官方编号

    返回 {class_name: class_id}
    """
    out_csv = out_csv or C.CLASS_MAP_FILE

    # 1) 官方映射文件
    for cand in (out_csv, C.DATA_DIR / "class_map.csv",
                 C.DATA_DIR / "classes.csv", C.DATA_DIR / "label_map.csv"):
        if cand.exists():
            df = pd.read_csv(cand)
            cols = {c.lower().strip(): c for c in df.columns}
            name_col = next((cols[k] for k in ("class_name", "name", "folder",
                                               "label_name", "class")
                             if k in cols), None)
            id_col = next((cols[k] for k in ("class_id", "id", "label",
                                             "cls_id", "index")
                           if k in cols), None)
            if name_col and id_col:
                m = ClassMap({str(r[name_col]).strip(): int(r[id_col])
                              for _, r in df.iterrows()})
                m.confidence = "official_file"
                m.source = cand.name
                print(f"[class_map] ✅ 使用官方映射文件 {cand.name}，共 {len(m)} 类")
                return m

    # 2) 前缀数字
    prefixed = {}
    ok = True
    for d in class_dirs:
        mt = _LEADING_NUM.match(d.name)
        if not mt:
            ok = False
            break
        prefixed[d.name] = int(mt.group(1))
    if ok and prefixed:
        m = ClassMap(prefixed)
        ids = sorted(prefixed.values())
        contiguous = (ids == list(range(ids[0], ids[0] + len(ids))))
        m.confidence = "prefix_contiguous" if contiguous else "prefix_sparse"
        m.source = "文件夹名前缀数字"
        print(f"[class_map] 依据文件夹名前缀数字建立映射，共 {len(prefixed)} 类；"
              f"编号范围 {ids[0]}..{ids[-1]}"
              f"{'（连续）' if contiguous else '（⚠️ 不连续）'}")
        return m

    # 3) 顺序编号（需要人工核对）
    seq = ClassMap({d.name: i + 1 for i, d in enumerate(class_dirs)})
    seq.confidence = "sequential"
    seq.source = "按名称排序顺序编号"
    print("[class_map] ⚠️⚠️ 文件夹名没有数字前缀，已按名称排序顺序编号 1..N。")
    print("[class_map] ⚠️⚠️ 该编号**几乎一定不等于官方类别编号**，"
          "直接提交将被判全错。")
    print("[class_map]      请整理官方类别表为 data/class_map.csv"
          "（两列：class_name,class_id）后重跑 scan。")
    return seq


# ------------------------------------------------------------------ 映射自检
def audit_class_map(class_map: dict[str, int], num_classes: int,
                    verbose: bool = True) -> list[str]:
    """审查类别映射，返回「问题清单」（空列表 = 通过）。

    检查项：
      * 来源可信度（来源为顺序编号 → 直接不通过）
      * 编号是否唯一（两位类别撞到同一编号 = 一定错）
      * 编号范围是否连续、是否从 0 或 1 开始
      * 类数与编号跨度是否自洽
    """
    issues: list[str] = []
    conf = getattr(class_map, "confidence", "unknown")
    ids = sorted(class_map.values())

    if conf == "sequential":
        issues.append("映射来自「按名称排序顺序编号」，未经官方确认 —— 提交将被判全错")
    if conf == "prefix_sparse":
        issues.append(f"文件夹前缀编号不连续（{ids[0]}..{ids[-1]}，{len(ids)} 个类）"
                      f"，需确认是否与官方编号一致")
    if len(set(ids)) != len(ids):
        dup = len(ids) - len(set(ids))
        issues.append(f"存在 {dup} 个重复的类别编号（两个类共用一个编号）")
    if len(ids) != num_classes:
        issues.append(f"映射条目数 {len(ids)} != 实际类别数 {num_classes}")

    # 0/1 起点的歧义
    if ids and ids[0] == 1:
        issues.append("编号从 1 开始：如果官方是 0 起编号，整体会偏移一位 → "
                      "全错。务必确认（可用 pipeline.py predict --id-shift -1 纠正）")
    if ids and ids[0] > 1:
        issues.append(f"最小编号是 {ids[0]}（既不是 0 也不是 1），确认无误？")

    if verbose:
        print(f"[audit] 映射来源：{getattr(class_map, 'source', '?')}"
              f"（可信度 {conf}）")
        print(f"[audit] 共 {len(ids)} 类，编号 {ids[0]}..{ids[-1]}"
              f"（连续跨度 {len(ids)}）")
        sample = sorted(class_map.items(), key=lambda kv: kv[1])[:3]
        tail = sorted(class_map.items(), key=lambda kv: kv[1])[-2:]
        print(f"[audit] 样例：{sample} ... {tail}")
        if issues:
            print("[audit] ⚠️ 发现以下问题：")
            for i in issues:
                print(f"[audit]    - {i}")
        else:
            print("[audit] ✅ 未发现问题")
    return issues


def build_train_manifest(train_root: Path | None = None,
                         class_map: dict[str, int] | None = None) -> pd.DataFrame:
    root = find_train_root(train_root)
    dirs = list_class_dirs(root)
    class_map = class_map or infer_class_map(dirs)

    rows = []
    missing = []
    for d in dirs:
        cid = class_map.get(d.name)
        if cid is None:
            missing.append(d.name)
            continue
        for f in d.iterdir():
            if f.is_file() and _is_img(f):
                rows.append((str(f), d.name, int(cid)))
    if missing:
        raise KeyError(f"以下类别文件夹在映射中缺失编号：{missing[:10]}")

    df = pd.DataFrame(rows, columns=["path", "class_name", "class_id"])
    if df.empty:
        raise RuntimeError(f"训练目录扫描为空：{root}")
    names = sorted(df["class_name"].unique())
    name_to_idx = {n: i for i, n in enumerate(names)}
    df["label"] = df["class_name"].map(name_to_idx).astype(int)
    df.attrs["class_names"] = names
    df.attrs["num_classes"] = len(names)
    print(f"[train] {len(df)} 张图 / {len(names)} 类；目录 {root}")
    return df


def find_test_root(root: Path | None = None) -> Path | None:
    """定位真实测试根目录（容忍压缩包多套一层目录）。

    ★ 这里原本是 `return root`（只要目录非空就直接返回），**等于没有下钻** ——
    而测试清单用的是 `root.rglob("*")`，会递归扫到子目录，于是数据多套一层时
    **能跑通、不报错**，看起来一切正常。

    但 `_report_probs` 用 `test_labels.csv` 和清单做「按文件名」的字典匹配，
    不依赖目录层级，所以这个缺陷在**有真值**的场景下完全暴露不出来。
    一直到正式比赛（无真值）才可能在别的按路径处理的地方出问题。

    `build_train_manifest` 的同名函数 `find_train_root` 是实打实下钻的，
    两者行为不一致本身就是隐患，现在补齐。
    """
    root = Path(root or C.TEST_ROOT)
    if not root.exists():
        return None
    # 与 find_train_root 同一套判据：只有一个子目录、且当前层没有图片时下钻。
    # 上限 3 层，避免 "全是一层层空目录" 时无限往下走。
    for _ in range(3):
        subs = [d for d in root.iterdir() if d.is_dir()]
        has_img = any(f.is_file() and _is_img(f) for f in root.iterdir())
        if len(subs) == 1 and not has_img:
            root = subs[0]
        else:
            break
    return root if any(root.rglob("*")) else None


def build_test_manifest(test_root: Path | None = None) -> pd.DataFrame:
    """测试集清单。图片名（不含目录）即为提交 CSV 的第一列。

    若存在官方测试列表文件（一行一个文件名），以它为准并严格保持其顺序，
    避免因排序差异导致行数不匹配。
    """
    # 官方列表优先
    for lst in (C.TEST_LIST_FILE, C.DATA_DIR / "test.txt",
                C.DATA_DIR / "test_list.csv"):
        if lst.exists():
            txt = lst.read_text(encoding="utf-8", errors="ignore")
            names = [ln.strip().split(",")[0].strip()
                     for ln in txt.splitlines() if ln.strip()]
            names = [n for n in names if n.lower().endswith(tuple(IMG_EXTS)) or "." in n]
            if names:
                root = find_test_root(test_root)
                if root is None:
                    # 允许只给文件名，图片在 data/test/
                    root = C.TEST_ROOT
                # ★ 递归找是**故意的**：官方列表可能只给文件名，而图片在
                # data/test/ 下面某一层（压缩包常多套一层 test/test/）。
                # 这里若改成"只扫当前层"，多套一层就会一张都匹配不到。
                idx = {}
                for f in Path(root).rglob("*"):
                    if f.is_file() and _is_img(f):
                        idx.setdefault(f.name, f)
                rows = [(n, str(idx[n])) for n in names if n in idx]
                # ★★ 匹配率过低必须**报错退出**，不能只警告。
                #
                # 踩过的坑：这里原先是「只打印一句 ⚠️ 就继续」，然后把
                # 匹配到的部分行写成提交 CSV。后果是**静默产出残缺提交**：
                # 行数比官方列表少，但文件本身格式完全合法，check_submission
                # 这类"格式校验"也查不出来 —— 一直要到排行榜给个低分才发现。
                # 正式比赛没有 test_labels.csv 兜底，这个残缺文件就是最终提交。
                #
                # 判据用 99%：允许极个别文件名异常（如大小写差异），
                # 但拒绝"系统性匹配失败"。
                if len(names) and len(rows) < 0.99 * len(names):
                    miss = [n for n in names if n not in idx][:5]
                    raise RuntimeError(
                        f"\n[test] 🛑 官方测试列表 {lst.name} 有 {len(names)} 条，"
                        f"但只匹配到 {len(rows)} 张图（丢失 {len(names)-len(rows)} 张）。\n"
                        f"[test]     这几乎一定是**测试图片目录摆放不对**，"
                        f"不是文件缺失。\n"
                        f"[test]     示例未命中文件名：{miss}\n"
                        f"[test]     请确认图片在 {C.TEST_ROOT} 下（可多套一层）。\n"
                        f"[test]     若确实要接受残缺提交，请人工核对后再手动放行。\n"
                    )
                df = pd.DataFrame(rows, columns=["img_name", "path"])
                print(f"[test] 使用官方列表 {lst.name}，{len(df)} 张")
                df.attrs["from_official_list"] = True
                return df

    root = find_test_root(test_root)
    if root is None:
        raise FileNotFoundError(
            f"未找到测试目录：{C.TEST_ROOT}\n请把测试图片放到 data/test/ 下。"
        )
    rows = [(f.name, str(f)) for f in sorted(root.rglob("*")) if f.is_file() and _is_img(f)]
    df = pd.DataFrame(rows, columns=["img_name", "path"])
    df.attrs["from_official_list"] = False
    print(f"[test] {len(df)} 张（未发现官方列表，按文件名排序）")
    return df


def load_image(path: str) -> Image.Image | None:
    """稳健读取单张图；失败返回 None（调用方用占位图兜底）。"""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def remap_to_official_ids(pred_idx: np.ndarray, class_names: list[str],
                          class_map: dict[str, int]) -> np.ndarray:
    """把内部 0..N-1 的类别下标映射回官方四位类别编号。"""
    lut = np.array([class_map[n] for n in class_names], dtype=np.int64)
    return lut[pred_idx]


def write_submission(img_names, class_ids, out_dir: Path | None = None,
                     class_map: dict[str, int] | None = None,
                     class_names: list[str] | None = None,
                     allow_unverified: bool = False) -> Path:
    """写 pred_results.csv 并压成 zip（赛题要求：文件固定名 pred_results.csv）。

    ⚠️ 逗号后**保留一个空格**，与赛题示例 `xxxxxxxxxxxx.jpg, 0001` 逐字对齐。
    虽然标准 CSV 解析器不区分，但若评测端做字符串匹配则空格会成为失分点。

    ``class_map`` 非空时会做闸门检查：如果映射来源不可信（顺序编号）或存在
    严重问题，**默认拒绝写出提交文件** —— 因为错的编号会拿 0 分，
    而漏交一次和交错的，损失是一样的。确认无误时传 allow_unverified=True 放行。
    """
    import zipfile

    out_dir = Path(out_dir or C.SUB_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if class_map is not None:
        issues = audit_class_map(class_map, len(class_map))
        blocking = [i for i in issues
                    if "顺序编号" in i or "重复的类别编号" in i]
        if blocking and not allow_unverified:
            raise SystemExit(
                "\n[submit] 🛑 已阻止写出提交文件：类别编号映射不可信。\n"
                + "\n".join(f"[submit]    - {i}" for i in blocking) +
                "\n[submit] 请先把官方类别表整理成 data/class_map.csv"
                "（列名 class_name,class_id），或确认无误后用"
                " --allow-unverified-map 强制放行。\n")
        if issues:
            print("[submit] ⚠️ 映射存在以下待确认项（已放行）：")
            for i in issues:
                print(f"[submit]    - {i}")

    csv_path = out_dir / C.SUB_CSV_NAME
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        for n, c in zip(img_names, class_ids):
            f.write(f"{n}, {int(c):0{C.CLASS_ID_WIDTH}d}\n")
    zip_path = out_dir / (C.SUB_CSV_NAME.replace(".csv", "") + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, C.SUB_CSV_NAME)

    # 留一份「本次提交实际用了什么映射」的审计记录，便于复现与事后核对
    if class_map and class_names:
        audit = out_dir / "class_map_used.csv"
        with open(audit, "w", encoding="utf-8", newline="") as f:
            f.write("class_name,class_id,source,confidence\n")
            for n in class_names:
                f.write(f"{n},{int(class_map[n])},"
                        f"{getattr(class_map, 'source', '')},"
                        f"{getattr(class_map, 'confidence', '')}\n")
        print(f"[submit] 映射审计 {audit}")

    print(f"[submit] {csv_path}")
    print(f"[submit] {zip_path}  ({len(img_names)} 行)")
    return zip_path


def dump_meta(df: pd.DataFrame, path: Path) -> None:
    meta = {
        "class_names": list(df.attrs.get("class_names", [])),
        "num_classes": int(df.attrs.get("num_classes", 0)),
        "n_samples": int(len(df)),
    }
    Path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                          encoding="utf-8")
