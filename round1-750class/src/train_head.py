"""在缓存特征上训练噪声鲁棒分类头。

流程
----
1. **K 折 OOF**：分层交叉验证得到每条的 out-of-fold 预测概率，
   用于 Confident Learning 估计噪声转移矩阵 T（同一份数据不能既训练又估噪）。
2. **估 T**：`noise.estimate_transition_matrix`。
3. **训练最终头**：在**全部训练数据**上用「CL 保真度降权 x 类平衡」加权，
   损失用前向校正形式 `-sum_j T[y,j] log p_j`。

关键实现细节
------------
特征已 L2 归一化（模长 1）。若用不带温度缩放的线性层，logit 量级只有 O(1)，
softmax 接近均匀分布，交叉熵梯度极小、**根本训不动**。CLIP 官方预训练用的
logit_scale = exp(4.6052) ≈ 100（即温度 1/100）。因此这里给头加一个可学习
的 `logit_scale`，并按 CLIP 惯例初始化。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import noise as NZ


# ------------------------------------------------------------------ 模型
class RobustHead(nn.Module):
    """线性/单隐层分类头 + 可学习 logit 温度（CLIP 风格）。"""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 0,
                 init_scale: float = 100.0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
        else:
            self.net = nn.Linear(in_dim, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(init_scale))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        # 上限 100 与 CLIP 预训练一致，避免温度无界漂移
        return self.logit_scale.clamp(max=np.log(100.0)).exp() * z


# ------------------------------------------------------------------ 评估
@torch.no_grad()
def accuracy(model, X: torch.Tensor, y: torch.Tensor | None = None,
             batch: int = 8192, prior_q: np.ndarray | None = None):
    model.eval()
    logits_all = []
    for i in range(0, X.shape[0], batch):
        logits_all.append(model(X[i:i + batch]).float())
    logits = torch.cat(logits_all, 0)
    if prior_q is not None:
        probs = NZ.prior_correction(logits, prior_q)
    else:
        probs = F.softmax(logits, dim=1)
    if y is None:
        return probs
    pred = probs.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    return acc, probs


def macro_f1(pred: np.ndarray, y: np.ndarray, K: int) -> float:
    """宏平均 F1（长尾场景比 accuracy 更能反映头尾均衡）。"""
    f1s = []
    for k in range(K):
        tp = np.sum((pred == k) & (y == k))
        fp = np.sum((pred == k) & (y != k))
        fn = np.sum((pred != k) & (y == k))
        if tp + fn == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


# ------------------------------------------------------------------ 训练
def _train_loop(Xtr, ytr, wtr, Xva, yva, T, kind: str, device,
                epochs: int, lr: float, wd: float, hidden: int,
                alpha: float, beta: float, q: float,
                label_smooth: float, log_prefix: str = "",
                seed: int = 0, num_classes: int | None = None) -> RobustHead:
    # ★ 固定随机种子：权重初始化与样本洗牌都依赖全局 RNG。
    # 不固定的话，两次「同配置」跑出的差异会混进来，消融表就不可信
    # （实测会让同一配置的标准差膨胀到 5 个百分点）。
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    # ★ 类别数必须显式传入，不能只看数据里的最大标签。
    # 长尾数据下最低频的类别常被排在最高编号，若某个折里它恰好样本极少、
    # 甚至全部落进验证集，max(label)+1 就会偏小 —— 输出维度错、评估直接崩。
    if num_classes is not None:
        K = int(num_classes)
    else:
        K = int(max(ytr.max().item(),
                    yva.max().item() if yva is not None else 0) + 1)
    in_dim = Xtr.shape[1]
    if ytr.max().item() >= K or (yva is not None and yva.max().item() >= K):
        raise ValueError(
            f"标签超出类别数：num_classes={K}，数据里出现了更大的标签，"
            f"请核对类别编号映射。")
    model = RobustHead(in_dim, K, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * max(1, Xtr.shape[0] // C.HEAD_BATCH + 1)),
        pct_start=0.15)

    n = Xtr.shape[0]
    steps_per_epoch = max(1, n // C.HEAD_BATCH)
    best = {"acc": -1.0, "state": None, "ep": -1}
    last = None  # 末轮（未经挑选）的指标，见函数末尾的说明

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(steps_per_epoch):
            sel = perm[s * C.HEAD_BATCH:(s + 1) * C.HEAD_BATCH]
            if sel.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr[sel])
            loss = NZ.compute_loss(logits, ytr[sel], sample_w=wtr[sel], T=T,
                                   kind=kind, alpha=alpha, beta=beta, q=q,
                                   label_smooth=label_smooth)
            # ★★ NaN 拦截 —— 必须在 backward 之前。
            # 踩过的坑：`CB_BETA = 1.0` 会让类平衡权重整体变成 NaN（分子
            # 1-beta=0，随后 0/0），于是 loss 直接是 nan。此时训练**不会崩**，
            # 只会静默跑完：每轮 loss=nan、val_acc 掉到 0.0016（≈全押一类），
            # 最后还"正常"保存了一个模型。实测就是这样白跑了 60 轮。
            # 所以这里主动炸掉，把排查成本从「事后看曲线」降到「当场报错」。
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[train_head] loss 在第 {ep} 轮变成非有限值（{loss.item()}）。\n"
                    f"  最常见的两个原因：\n"
                    f"    1) 类平衡权重出现 NaN —— 检查 CB_BETA 是否被设成了 1.0\n"
                    f"       （必须是开区间 (0,1) 内的值，例如 0.999）。\n"
                    f"    2) 样本权重里混入了 nan/inf —— 检查 T 与 CL 相关配置。\n"
                    f"  另可临时加 --lr-warmup 或降低 --lr 排除数值溢出。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            try:
                sched.step()
            except Exception:  # noqa: BLE001
                pass
            tot += loss.item()

        if Xva is not None and yva is not None and (ep % 5 == 0 or ep == epochs):
            acc, probs = accuracy(model, Xva, yva)
            pred = probs.argmax(1).cpu().numpy()
            mf1 = macro_f1(pred, yva.cpu().numpy(), K)
            print(f"{log_prefix} ep{ep:3d} loss={tot/steps_per_epoch:.4f} "
                  f"val_acc={acc:.4f} val_macroF1={mf1:.4f}", flush=True)
            last = {"ep": ep, "acc": float(acc), "mf1": float(mf1)}
            if acc > best["acc"]:
                best = {"acc": acc, "state": copy.deepcopy(model.state_dict()), "ep": ep}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"{log_prefix} 取最优 epoch={best['ep']} val_acc={best['acc']:.4f}")
    # ★★ 把「末轮（未经挑选）」的指标挂到模型上。
    #
    # 为什么必须留这个数：`best` 是**按留出集挑出来的最优 epoch**。
    # 轮数越多 ⇒ 评估过的候选点越多（60 轮 12 个、480 轮 96 个），
    # **"挑到偶然高分"的机会也随之变大**，最优值本身会随轮数虚高。
    # 所以「多加轮数后最优值变好了」这句结论**天然可疑** ——
    # 必须能把两者分开看：
    #   * 末轮指标也同步上升 ⇒ 确实是欠训，加轮数有真实收益；
    #   * 末轮指标反而下降、只有被挑出来的最优值上升 ⇒ 是过拟合 + 挑选虚高，
    #     这时该做的是**降 lr / 加正则 / 早停**，不是无脑加轮数。
    model.last_val = last
    return model


def get_oof_probs(X: torch.Tensor, y: torch.Tensor, num_classes: int,
                  folds: int, device, seed: int = C.SEED,
                  kind: str = "ce", **kw) -> np.ndarray:
    """分层 K 折 OOF 预测概率，供 Confident Learning 估 T。"""
    from sklearn.model_selection import StratifiedKFold

    y_np = y.cpu().numpy()
    oof = np.zeros((len(y_np), num_classes), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, va) in enumerate(skf.split(np.zeros(len(y_np)), y_np), 1):
        tr_t = torch.as_tensor(tr, device=device)
        va_t = torch.as_tensor(va, device=device)
        # OOF 折内不加 CL 权重（此时还没有 T），只用基础损失 + 类平衡
        w = torch.ones(len(tr), device=device)
        if C.USE_CLASS_BALANCE == "cb":
            w = torch.as_tensor(NZ.class_balanced_weights(y_np[tr], num_classes,
                                                          C.CB_BETA), device=device)
        elif C.USE_CLASS_BALANCE == "inv":
            w = torch.as_tensor(NZ.inverse_freq_weights(y_np[tr], num_classes),
                                device=device)
        m = _train_loop(X[tr_t], y[tr_t], w, None, None, None, kind, device,
                        epochs=kw.get("epochs", 20),
                        # ★ 用 OOF_HEAD_LR 而不是 HEAD_LR：最终头的 lr 被调到很小时
                        #   （现为 1e-4），折内只训 20 轮会训不起来 ⇒ OOF 概率接近均匀
                        #   ⇒ 估出的 T 退化。两者必须解耦，理由见 config 里那段注释。
                        lr=C.OOF_HEAD_LR, wd=C.HEAD_WD,
                        hidden=C.HEAD_HIDDEN, alpha=C.SCE_ALPHA, beta=C.SCE_BETA,
                        q=C.GCE_Q, label_smooth=C.LABEL_SMOOTH,
                        log_prefix=f"[oof f{f}]", seed=seed + f,
                        num_classes=num_classes)
        probs = accuracy(m, X[va_t])  # y=None 时只返回概率
        oof[va] = probs.cpu().numpy().astype(np.float64)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    np.save(C.CACHE_DIR / "oof_probs.npy", oof)
    print(f"[oof] 完成，已存 {C.CACHE_DIR / 'oof_probs.npy'}")
    return oof


def train_final(X: torch.Tensor, y: torch.Tensor, T: np.ndarray | None,
                num_classes: int, device, val_ratio: float = 0.05,
                kind: str = C.LOSS_KIND, epochs: int = C.HEAD_EPOCHS,
                seed: int = C.SEED, log_prefix: str = "[head]",
                save_name: str = "head.pt") -> RobustHead:
    """用全部数据训练最终头（含验证留出用于选 epoch）。

    ``save_name`` 决定权重落盘文件名。**消融实验务必传不同的名字**，
    否则会互相覆盖（曾因此把已提交版本的权重冲掉）。
    """
    y_np = y.cpu().numpy()
    n = len(y_np)
    rng = np.random.RandomState(seed)
    val_mask = rng.rand(n) < val_ratio

    w_np = NZ.combine_weights(y_np, T, num_classes,
                              use_cl=C.USE_CL_REWEIGHT,
                              cl_lo=C.CLIP_LO, cl_hi=C.CLIP_HI,
                              class_balance=C.USE_CLASS_BALANCE,
                              cb_beta=C.CB_BETA)
    w = torch.as_tensor(w_np, dtype=torch.float32, device=device)
    T_t = None if T is None else torch.as_tensor(T, dtype=torch.float32, device=device)

    tr = torch.as_tensor(~val_mask, device=device)
    va = torch.as_tensor(val_mask, device=device)
    model = _train_loop(X[tr], y[tr], w[tr], X[va], y[va], T_t, kind, device,
                        epochs=epochs, lr=C.HEAD_LR, wd=C.HEAD_WD,
                        hidden=C.HEAD_HIDDEN, alpha=C.SCE_ALPHA, beta=C.SCE_BETA,
                        q=C.GCE_Q, label_smooth=C.LABEL_SMOOTH,
                        log_prefix=log_prefix, seed=seed,
                        num_classes=num_classes)

    out = C.CACHE_DIR / save_name
    torch.save({"state": model.state_dict(),
                "num_classes": num_classes,
                "in_dim": int(X.shape[1]),
                "hidden": C.HEAD_HIDDEN,
                "kind": kind}, out)
    print(f"{log_prefix} 已保存 {out}")
    return model


def load_final(path: Path | None = None, device: str = "cpu") -> RobustHead:
    path = Path(path or (C.CACHE_DIR / "head.pt"))
    ck = torch.load(path, map_location=device)
    m = RobustHead(ck["in_dim"], ck["num_classes"], ck.get("hidden", 0)).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


# --------------------------------------------------- 推理期先验（长尾用）