"""零样本 CLIP 基线（免费的先验强度探针）。

用途有二：
  1. 若类别文件夹名是**语义名**（如 "robin" / "sunflower"），零样本分类本身
     就是一个不弱于线性探针的基线，且**完全不需要训练** —— 适合在等数据、
     等环境时先拿一个"榜上有分"的兜底结果；
  2. 若类别名是纯数字编号，零样本无意义，本模块会直接告警并退出。

注意：零样本用的文本编码器与骨干同属一个 CLIP（ViT-B/32），不构成多模型集成。
"""
from __future__ import annotations

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import torch
import torch.nn.functional as F

from extract import load_clip, take_embeds

PROMPT_TEMPLATES = [
    "a photo of a {}, a type of fine-grained object.",
    "a close-up photo of a {}.",
    "a photo of a {} in its natural habitat.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
]

DIGIT_ONLY = set("0123456789_-. ")


def looks_semantic(names: list[str]) -> bool:
    """判断类别名是否为语义名（非纯数字编号）。"""
    n_num = sum(1 for s in names if set(s.strip()) <= DIGIT_ONLY)
    return (n_num / max(len(names), 1)) < 0.5


@torch.no_grad()
def zeroshot_probs(class_names: list[str], image_features: np.ndarray,
                   device: str = "cuda", templates: list[str] | None = None
                   ) -> np.ndarray:
    """用多模板平均的文本嵌入做零样本分类，返回 (N, K) 概率。"""
    templates = templates or PROMPT_TEMPLATES
    kind, model, proc = load_clip(device)

    texts = []
    for t in templates:
        texts += [t.format(c.replace("_", " ")) for c in class_names]

    if kind == "transformers":
        inp = proc(text=texts, return_tensors="pt", padding=True,
                   truncation=True).to(device)
        tf = take_embeds(model.get_text_features(**inp))
    else:
        tk = __import__("open_clip").get_tokenizer(C.CLIP_OPENCLIP_ID[0])
        toks = tk(texts).to(device)
        tf = model.encode_text(toks)

    tf = F.normalize(tf.float(), dim=-1)
    K = len(class_names)
    tf = tf.view(len(templates), K, -1).mean(dim=0)
    tf = F.normalize(tf, dim=-1)

    xf = torch.as_tensor(image_features, dtype=torch.float32)
    xf = F.normalize(xf, dim=-1)

    logits = 100.0 * (xf @ tf.T.cpu())
    probs = F.softmax(logits, dim=1).numpy().astype(np.float64)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs


def run_from_cache(device: str = "cuda") -> np.ndarray | None:
    import json

    meta_p = C.CACHE_DIR / "train_meta.json"
    if not meta_p.exists():
        print("[zeroshot] 缺少 train_meta.json，请先跑 scan 阶段")
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    names = meta["class_names"]

    if not looks_semantic(names):
        print("[zeroshot] ⚠️ 类别名看似为纯数字编号，零样本不可用（文本提示无意义）。")
        print("[zeroshot]    跳过，改走线性探针路线（pipeline 主体）。")
        return None

    feat_p = C.CACHE_DIR / "feat_test.npy"
    if not feat_p.exists():
        print("[zeroshot] 缺少 feat_test.npy，请先跑 extract 阶段")
        return None
    Xte = np.load(feat_p)
    probs = zeroshot_probs(names, Xte, device)
    np.save(C.CACHE_DIR / "zeroshot_test_probs.npy", probs)
    pred = probs.argmax(1)
    print(f"[zeroshot] 完成；预测覆盖 {len(np.unique(pred))}/{len(names)} 类")
    return probs
