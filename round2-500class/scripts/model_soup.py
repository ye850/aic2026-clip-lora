# -*- coding: utf-8 -*-
"""模型汤 (Model Soup)：把多个 LoRA 检查点的权重做元素平均，合成单一模型。

合规说明：只平均【权重】，推理时仍是单模型、不聚合多个模型输出
（与已在栈内的 SWA 同源思路），因此不触发"禁集成"判据（判据=提交端是否聚合）。

用法：
  python scripts/model_soup.py \
    --ckpts outputs/runA_best.pt outputs/px560_best.pt outputs/sce0_best.pt \
    --out outputs/soup_v0_best.pt

说明：
  - 输入是训练侧存的 *_best.pt（含 "model" 键，结构见 finetune_lora.py:2037）。
  - 各 ckpt 必须同架构（同 r / hidden / targets / img_size），否则张量形状对不上会报错。
  - 平均后另存为 best 风格 ckpt，predict_tta.py 可直接 --lora 加载。
"""
import argparse, torch, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="参与 soup 的 *_best.pt 列表")
    ap.add_argument("--out", required=True, help="输出的 soup 权重路径")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="可选：各 ckpt 的加权系数（默认等权）。数量须与 --ckpts 一致")
    args = ap.parse_args()

    if args.weights is not None and len(args.weights) != len(args.ckpts):
        sys.exit("[soup] --weights 数量须与 --ckpts 一致")

    w = args.weights or [1.0] * len(args.ckpts)
    tw = sum(w)

    sd_acc = None
    meta = None
    for path, wi in zip(args.ckpts, w):
        ck = torch.load(path, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if sd_acc is None:
            sd_acc = {k: v.float() * wi for k, v in sd.items()}
            meta = ck
        else:
            for k in sd_acc:
                if k not in sd:
                    sys.exit(f"[soup] 键缺失于 {path}: {k}")
                sd_acc[k] = sd_acc[k] + sd[k].float() * wi
    sd_acc = {k: (v / tw).half() for k, v in sd_acc.items()}

    out = {
        "model": sd_acc,
        "num_classes": meta.get("num_classes"),
        "r": meta.get("r"),
        "backbone_kind": meta.get("backbone_kind"),
        "img_size": meta.get("img_size"),
        "best_ep": -1, "best_val_acc": None,
        "best_is_ema": False,
        "args": meta.get("args", {}),
        "soup": True,
    }
    torch.save(out, args.out)
    print(f"[soup] 已平均 {len(args.ckpts)} 个 ckpt -> {args.out}")
    print(f"[soup] 架构: r={out['r']} hidden={out['args'].get('hidden')} "
          f"img={out['img_size']} 类={out['num_classes']}")

if __name__ == "__main__":
    main()
