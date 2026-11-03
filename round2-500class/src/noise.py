"""标签噪声处理核心方法（从二分类推广到多分类）。

本模块把原团队在表格欺诈检测上验证过的两条机制迁移到细粒度图像识别：

  1. **Confident Learning 噪声转移矩阵估计**
     用 K 折 Out-of-Fold 预测概率估计「真实标签 -> 观测标签」的转移矩阵 T。
     T[k, k] 即类别 k 的**标签保真度**（该类别标注可信程度）。

  2. **逐类保真度降权**（原式：``y=0`` 用 ``clip(T[0,0], .5, 1)``）
     多分类推广为：``w_i = clip(T[y_i, y_i], LO, HI)``
     直觉：保真度低的类别，其标签更可能是错的，降权以抑制错标污染。

  3. **前向校正损失（ARR）**
     ``L = -sum_j T[y, j] * log p_j``，把「观测标签」的监督信号按噪声通道
     摊回到真实标签的可能取值上，而非盲目相信观测标签。
     —— 原项目 ``run_arr_loss.py`` 的同一思想，从 2x2 推广到 KxK。

  4. **类平衡加权**（复赛/半决赛引入长尾分布）
     ``w_c = (1 - beta) / (1 - beta^{n_c})``（Cui et al. 2019）

上面 1+3 是本方法与「直接用交叉熵微调」的核心区别；2 与 4 分别针对
噪声与长尾两个数据缺陷。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------- CL 噪声估计
def estimate_transition_matrix(oof_probs: np.ndarray,
                               y_noisy: np.ndarray,
                               num_classes: int | None = None,
                               method: str = "confusion",
                               verbose: bool = True) -> np.ndarray:
    """用 Out-of-Fold 预测概率估计噪声转移矩阵 T。

    ``method`` 两种实现 —— 默认 **confusion**，原因见下：

    ``"confusion"``（默认，本场景更可靠）
        直接统计 OOF 混淆矩阵 ``A[k, j] = P(pred = j | obs = k)``。
        当模型对**真实**标签预测得较准时，``A[k, j]`` 就是 ``P(真 = j | 观 = k)``；
        再用贝叶斯关系换算到所需方向：

            T[j, k] = A[k, j] * pi_obs[k] / pi_true[j]

        其中 ``pi_obs`` 为观测标签边缘分布，``pi_true`` 取模型预测的边缘分布。
        （该式自动满足一致性 ``pi_obs = T^T pi_true``，无需迭代。）

    ``"confident_joint"``（Northcutt et al. 2021 原始做法，保留供对照）
        对每个观测类取该类「被判为自身标签」概率的均值为阈值，只保留超过阈值的
        「置信样本」再统计混淆。

        ⚠️ **本项目中该方法会严重低估噪声率**：其前提是模型会「记住」噪声标签，
        从而让错标样本仍以高置信度留在置信集里。而本工程用的是**强特征 + 线性探针**，
        模型是**泛化**而非记忆 —— 错标样本被 OOF 模型高置信度判成真实类，
        于是被阈值恰好筛掉，T 退化为单位阵，降权与前向校正同时失效。
        （实测：真实噪声 30%，该方法估出 0.0016。）

    Returns:
        T: (K, K)，``T[i, j] = P(观测 = j | 真实 = i)``，行归一化；对角线为保真度。
    """
    oof_probs = np.asarray(oof_probs, dtype=np.float64)
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    K = int(num_classes or oof_probs.shape[1])
    if oof_probs.shape[0] != y_noisy.shape[0]:
        raise ValueError("oof_probs 与 y_noisy 样本数不一致")

    pred = oof_probs.argmax(axis=1)

    if method == "confusion":
        C = np.zeros((K, K), dtype=np.float64)
        np.add.at(C, (y_noisy, pred), 1.0)
        row = C.sum(axis=1, keepdims=True)
        empty = (row.ravel() == 0)
        A = np.divide(C, np.maximum(row, 1.0), where=~empty,
                      out=np.tile((np.ones(K) / K), (K, 1)))
        A[empty, :] = 1.0 / K

        pi_obs = C.sum(axis=1) / max(C.sum(), 1.0)
        pi_true = np.bincount(pred, minlength=K).astype(np.float64)
        pi_true = pi_true / max(pi_true.sum(), 1.0)
        pi_true = np.maximum(pi_true, 1e-6)

        # T[j, k] = A[k, j] * pi_obs[k] / pi_true[j]
        T = (A.T * pi_obs[None, :]) / pi_true[:, None]
        rs = T.sum(axis=1, keepdims=True)
        T = np.divide(T, np.maximum(rs, 1e-12))
    elif method == "confident_joint":
        C = np.zeros((K, K), dtype=np.float64)
        for k in range(K):
            idx = np.where(y_noisy == k)[0]
            if idx.size == 0:
                continue
            p_own = oof_probs[idx, k]
            conf = idx[p_own >= p_own.mean()]
            if conf.size == 0:
                conf = idx
            C[k, :] = np.bincount(pred[conf], minlength=K).astype(np.float64)
        row = C.sum(axis=1, keepdims=True)
        empty = (row.ravel() == 0)
        if empty.any():
            C[empty, :] = 1.0
            row = C.sum(axis=1, keepdims=True)
        T = C / row
    else:
        raise ValueError(f"未知 method：{method}")

    if verbose:
        diag = np.diag(T)
        print(f"[CL:{method}] 转移矩阵 {K}x{K}")
        print(f"[CL:{method}] 整体标签噪声率 ≈ "
              f"{1.0 - diag.mean():.4f}（类宏平均）")
        worst = np.argsort(diag)[:5]
        print("[CL:%s] 保真度最低 5 类：%s" % (
            method, ", ".join(f"{i}:{diag[i]:.3f}" for i in worst)))
    return T


def fidelity_weights(y_noisy: np.ndarray, T: np.ndarray,
                     lo: float = 0.30, hi: float = 1.00) -> np.ndarray:
    """逐样本按类别保真度降权：``w_i = clip(T[y_i, y_i], lo, hi)``。

    这是原项目 ``minority_preserving_weights`` 的多分类推广：
    二分类时它退化为「正常类用 clip(T[0,0], .5, 1)、少数类保权」。
    """
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    T = np.asarray(T, dtype=np.float64)
    fid = np.clip(np.diag(T), lo, hi)
    return fid[y_noisy].astype(np.float64)


def temper_transition_matrix(T: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """把 T 向单位阵收缩：``T_use = (1-s)*I + s*T``。

    T 估计必然有偏差，而**一个错的 T 会主动伤害**性能（前向校正把监督信号
    按错误通道摊开）。用 strength < 1 收缩可以把「校正」限制在安全范围内：
    strength=0 即完全不做校正（退化为普通 CE），strength=1 即全量校正。
    """
    T = np.asarray(T, dtype=np.float64)
    s = float(np.clip(strength, 0.0, 1.0))
    if s >= 1.0:
        return T
    K = T.shape[0]
    return (1.0 - s) * np.eye(K) + s * T


def estimate_label_fidelity(oof_probs: np.ndarray,
                            y_noisy: np.ndarray,
                            num_classes: int | None = None) -> np.ndarray:
    """直接估计逐类别**标签保真度** rho_k = P(真 = k | 观 = k)。

    取 OOF 混淆矩阵的对角（soft 版本，用概率而非 argmax）：
    ``rho_k = mean(p_k | obs = k)``。

    比走「估 T 再取对角」更直接，也少一层换算误差；用途是逐类降权。
    """
    oof_probs = np.asarray(oof_probs, dtype=np.float64)
    y = np.asarray(y_noisy, dtype=np.int64)
    K = int(num_classes or oof_probs.shape[1])
    rho = np.ones(K, dtype=np.float64)
    for k in range(K):
        idx = np.where(y == k)[0]
        if idx.size:
            rho[k] = float(oof_probs[idx, k].mean())
    return rho


def fidelity_weights_from_rho(y_noisy: np.ndarray, rho: np.ndarray,
                              lo: float = 0.30, hi: float = 1.00) -> np.ndarray:
    """用直接估出的保真度向量做逐样本降权。"""
    y = np.asarray(y_noisy, dtype=np.int64)
    return np.clip(np.asarray(rho, dtype=np.float64), lo, hi)[y]


# ---------------------------------------------------------------- 类平衡
def class_balanced_weights(y_noisy: np.ndarray, num_classes: int,
                           beta: float = 0.999) -> np.ndarray:
    """类平衡逐样本权重：``w_c = (1-beta)/(1-beta^{n_c})``（Cui et al. 2019）。

    ★★ 关于 ``beta`` 的两条硬约束（都是实测踩出来的）：

    1. **``beta = 1.0` 会让全部权重变成 NaN`**，而且**不会报错**。
       原因：分子 `(1.0 - beta) = 0`，于是 `w_c` 全为 0，紧接着的
       `w_c / w_c.mean()` 就是 `0 / 0 = NaN`。实测 `beta=1.0` 时
       **148,695 个权重全是 NaN**，随后 loss 直接变 `nan`、`val_acc` 掉到
       0.0016（等价于全押一类）。⚠️ `np.maximum(eff, 1e-12)` 那道地板只护住了
       **分母**，护不住分子的 0 —— 这正是它没被拦住的原因。
       ⇒ 本函数现在会**主动拦截** `beta >= 1.0`，而不是静默产出 NaN。

    2. **``beta`` 越大，尾部类权重越极端。** 实测（本工程真实 750 类数据）：

       ==========  ==================  ==================
       ``beta``    类平衡权重跨度       乘 CL 降权后的跨度
       ==========  ==================  ==================
       0.9         2.4x                8.1x
       0.99        18.7x               62.3x
       **0.999**   **44.0x**           **145.7x**
       0.9999      49.0x               162.0x
       ==========  ==================  ==================

       但本项目真实长尾**很轻**（头尾比 49.6x、Gini 0.0956、有效类数
       736.1/750）。也就是说默认的 0.999 是**对着比实际严重得多的长尾**设的，
       把尾部类样本的梯度贡献压到了头部的约 1/146。扫描显示 0.99 与 0.999
       在覆盖度上**完全等价**（都是 748/750 类），而跨度小得多 ⇒ 见
       ``scripts/sweep_cb_beta.py``。
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(
            f"class_balanced_weights: beta 必须落在开区间 (0, 1)，收到 {beta!r}。\n"
            f"  ⚠️ beta=1.0 会让**全部**权重变成 NaN（分子 1-beta=0，随后 0/0），\n"
            f"     而 loss 会静默变成 nan、准确率掉到随机水平以下，极难排查。\n"
            f"     想要「接近均匀」，请用 0.999 或 0.9999。")
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    n_c = np.bincount(y_noisy, minlength=num_classes).astype(np.float64)
    eff = 1.0 - np.power(beta, np.maximum(n_c, 1.0))
    w_c = (1.0 - beta) / np.maximum(eff, 1e-12)
    w_c = w_c / w_c.mean()  # 归一化，避免整体学习率漂移
    if not np.isfinite(w_c).all():
        raise ValueError(
            f"class_balanced_weights: 权重出现非有限值（beta={beta}）。"
            f"请检查 num_classes 与 y_noisy 的取值范围是否一致。")
    return w_c[y_noisy].astype(np.float64)


def inverse_freq_weights(y_noisy: np.ndarray, num_classes: int) -> np.ndarray:
    y_noisy = np.asarray(y_noisy, dtype=np.int64)
    n_c = np.bincount(y_noisy, minlength=num_classes).astype(np.float64)
    w_c = 1.0 / np.maximum(n_c, 1.0)
    w_c = w_c / w_c.mean()
    return w_c[y_noisy].astype(np.float64)


def combine_weights(y_noisy: np.ndarray, T: np.ndarray | None,
                    num_classes: int,
                    use_cl: bool = True, cl_lo: float = 0.30, cl_hi: float = 1.00,
                    class_balance: str = "cb", cb_beta: float = 0.999) -> np.ndarray:
    """把「CL 保真度降权」与「类平衡加权」乘在一起，返回逐样本权重。"""
    w = np.ones(len(y_noisy), dtype=np.float64)
    if use_cl and T is not None:
        w *= fidelity_weights(y_noisy, T, cl_lo, cl_hi)
    if class_balance == "cb":
        w *= class_balanced_weights(y_noisy, num_classes, cb_beta)
    elif class_balance == "inv":
        w *= inverse_freq_weights(y_noisy, num_classes)
    w = w / max(w.mean(), 1e-12)
    return w


# ---------------------------------------------------------------- 鲁棒损失
def sce_loss_legacy(logits: torch.Tensor, y: torch.Tensor,
                    alpha: float = 0.1, beta: float = 1.0,
                    reduction: str = "none") -> torch.Tensor:
    """【历史实现 —— **不是标准 SCE**】仅用于复现已提交的包，**不要用于新实验**。

    ★★★ 2026-10-05 起，本函数被**隔离**出来，不再作为 `sce` 的默认实现——
      原因：它的 RCE 项写成了 **预测熵**，而不是标准 SCE 的「含标签」形式。

    这个实现里：
        ``rce = -sum_j p_j log p_j``  = **预测熵 H(p)**，**完全不含标签 y**
    标准 SCE（Wang et al. 2019）是：
        ``rce = -sum_j p_j log y_j``  （one-hot 下 = 9.21·(1 - p_c)）
    ⇒ 两者**对真实类 logit 的梯度符号相反**（实测：本实现 +0.0008、标准 −0.0041）
      ⇒ **不是同一个算法**，只是一直被叫作 SCE。

    ★ 保留本函数的唯一理由是**可复现性**：
      已提交并得分的包（如榜分 74.3 的 `d_r448mlp1024`）用的就是它
      ⇒ 对应命令行 `--kind sce_legacy`。
    ★ **本函数的数值行为被逐字冻结，任何情况下不得修改。**

    详见 `outputs/SCE实现错误_复核与后果_20261005.md`。
    """
    logp = F.log_softmax(logits, dim=1)
    ce = F.nll_loss(logp, y, reduction="none")
    rce = -(logp.exp() * logp).sum(dim=1)      # ← 历史写法：熵（非标准）
    out = alpha * ce + beta * rce
    return out if reduction == "none" else out.mean()


def sce_loss(logits: torch.Tensor, y: torch.Tensor,
             alpha: float = 0.1, beta: float = 1.0,
             reduction: str = "none") -> torch.Tensor:
    """**标准**对称交叉熵 SCE = alpha*CE + beta*RCE（Wang et al. 2019）。

    RCE := ``-sum_j p_j log y_j``（官方 y 为 one-hot，且 clip 到 [1e-4, 1] 以避开 log 0）
        在 one-hot 下 = ``-log(1e-4) * (1 - p_c) = 9.21 * (1 - p_c)``
    ⇒ **含标签**：奖励抬高真实类概率（对真实类 logit 的梯度为负）。

    官方参考实现：
        ``beta*reduce_mean(-reduce_sum(y_pred_2 * tf.log(y_true_2)))``
        https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels/blob/master/loss.py

    ★ 与 `sce_loss_legacy` 的差别不是"参数不同"，而是**目标函数不同**（见其说明）。
    ★ 2026-10-05 起本函数是 `--kind sce` 的实现（即默认）。
    """
    logp = F.log_softmax(logits, dim=1)
    ce = F.nll_loss(logp, y, reduction="none")
    y1h = F.one_hot(y, logits.shape[1]).to(logp.dtype).clamp(1e-4, 1.0)
    rce = -(logp.exp() * torch.log(y1h)).sum(dim=1)
    out = alpha * ce + beta * rce
    return out if reduction == "none" else out.mean()


def gce_loss(logits: torch.Tensor, y: torch.Tensor, q: float = 0.7,
             reduction: str = "none") -> torch.Tensor:
    """广义交叉熵 GCE = (1 - p_y^q)/q（Zhang & Sabuncu 2018），对噪声鲁棒。"""
    p = F.softmax(logits, dim=1)
    py = p.gather(1, y.view(-1, 1)).squeeze(1).clamp(1e-7, 1.0)
    out = (1.0 - py.pow(q)) / q
    return out if reduction == "none" else out.mean()


def forward_corrected_loss(logits: torch.Tensor, y: torch.Tensor, T: torch.Tensor,
                           kind: str = "ce",
                           alpha: float = 0.1, beta: float = 1.0,
                           q: float = 0.7,
                           reduction: str = "none") -> torch.Tensor:
    """前向校正损失（ARR）：把观测标签监督信号按噪声通道摊回真实标签。

    ``L_i = - sum_j T[y_i, j] * log p_j``

    实现上等价于把 one-hot 标签替换成噪声通道的软目标 ``soft = T[y_i, :]``，
    再用 soft-label 形式的 CE / SCE / GCE 计算。
    """
    soft = T[y]  # (B, K) 行归一化
    logp = F.log_softmax(logits, dim=1)

    ce = -(soft * logp).sum(dim=1)
    if kind == "ce":
        out = ce
    elif kind in ("sce", "sce_legacy"):
        p = logp.exp()
        if kind == "sce_legacy":
            # 历史行为（= 预测熵，非标准）：**逐字冻结**，仅供复现已提交的包
            rce = -(p * logp).sum(dim=1)
        else:
            # ★ 标准 RCE：用**软目标**（含标签），与上面 ce 一致地"看标签"。
            #   注意 `soft = T[y]` 手边就有，历史上却写成了 `log p`。
            rce = -(p * torch.log(soft.clamp(1e-4, 1.0))).sum(dim=1)
        out = alpha * ce + beta * rce
    elif kind == "gce":
        # soft-label GCE：对每个可能类按 soft 权重加权
        p = logp.exp().clamp(1e-7, 1.0)
        out = (soft * ((1.0 - p.pow(q)) / q)).sum(dim=1)
    else:
        raise ValueError(f"未知的校正损失类型：{kind}")
    return out if reduction == "none" else out.mean()


def compute_loss(logits: torch.Tensor, y: torch.Tensor,
                 sample_w: torch.Tensor | None = None,
                 T: torch.Tensor | None = None,
                 kind: str = "forward_sce",
                 alpha: float = 0.1, beta: float = 1.0, q: float = 0.7,
                 label_smooth: float = 0.0,
                 return_per: bool = False) -> torch.Tensor:
    """统一损失入口。
    ``kind`` 取 ce|gce|forward_ce|forward_gce，以及
    ``sce`` / ``forward_sce``（= **标准 SCE**，Wang et al. 2019）与
    ``sce_legacy`` / ``forward_sce_legacy``（= **历史实现**，其 RCE 是预测熵而非标准形式）。

    ★★ **命名约定（2026-10-05 起）**：
      · ``sce`` = **标准 SCE**（RCE = ``-Σ p log y``），**这是默认，也是新实验该用的**；
      · ``*_legacy`` = 历史实现，**数值逐字冻结**，**仅用于复现已提交并得分的包**
        （如榜分 74.3 的 `d_r448mlp1024` ⇒ `--kind sce_legacy`）。
      详见 `outputs/SCE实现错误_复核与后果_20261005.md`。

    ``return_per=True`` 时返回**逐样本**损失（不取均值），供噪声样本筛选累积；
    ⚠️ 此时返回的是**未乘 ``sample_w``** 的原始损失 —— 这是必须的，理由见下。
    """
    if kind.startswith("forward"):
        if T is None:
            raise ValueError("前向校正损失需要提供转移矩阵 T")
        sub = kind.split("_", 1)[1] if "_" in kind else "ce"
        per = forward_corrected_loss(logits, y, T, sub, alpha, beta, q)
    elif kind in ("sce", "sce_legacy"):
        per = (sce_loss_legacy if kind == "sce_legacy" else sce_loss)(
            logits, y, alpha, beta)
    elif kind == "gce":
        per = gce_loss(logits, y, q)
    elif kind == "ce":
        per = F.cross_entropy(logits, y, reduction="none",
                              label_smoothing=label_smooth)
    else:
        raise ValueError(f"未知损失类型：{kind}")

    if return_per:
        # ★ 在乘 sample_w **之前**返回。sample_w 里含 CL 保真度降权
        #   （w = clip(T[y,y], 0.30, 1.00)）与类平衡权重，而被降权的恰恰是
        #   "更可疑"的那批样本。若判据用乘过权重的损失，它们会显得 loss 更小、
        #   从而被**保留**下来 —— 判据被自己的动作改写，形成自证其说的闭环。
        #   要问的是"模型对这个样本有多不适应"，就该取原始损失。
        return per

    if sample_w is not None:
        per = per * sample_w
    return per.mean()


def classwise_small_loss_flag(loss_sum: np.ndarray, loss_cnt: np.ndarray,
                              y: np.ndarray, frac: float,
                              min_keep: int = 10) -> np.ndarray:
    """按**类内分位**标出平均损失最高的一批样本（疑似噪声），返回 bool 掩码。

    为什么必须"类内"而不是全局
    --------------------------
    复赛是 750 类长尾（头尾比 49.6×）。小类样本天然学得慢、平均损失系统性高于
    大类 —— 全局分位会把**整个小类**判成噪声清掉，既与赛题「避免模型过度偏向
    头部类别」的要求正好相反，也会先把尾部毁掉。类内分位问的是"**在这一类里**，
    哪些样本模型最不适应"，那才是噪声的合理代理。

    ``min_keep`` 是硬地板：每类至少留这么多张。一类被清空等于删掉一个类，而
    官方测试集是**类别分布均衡**的 ⇒ 删类直接扣分。

    只用 ``loss_cnt > 0``（本轮真正访问过）的样本参与分位；没访问到的样本既不算
    也**不会**因为"loss=0"被误判成干净样本。
    """
    y = np.asarray(y, dtype=np.int64)
    n = y.shape[0]
    flag = np.zeros(n, dtype=bool)
    if frac <= 0:
        return flag
    seen = loss_cnt > 0
    mean_loss = np.zeros(n, dtype=np.float64)
    np.divide(loss_sum, np.maximum(loss_cnt, 1), out=mean_loss,
              where=seen)   # 未访问的保持 0，但下面用 seen 过滤，不会被选中
    for c in np.unique(y):
        idx = np.where((y == c) & seen)[0]
        if idx.size <= min_keep:
            continue                      # 整类已在硬地板内 ⇒ 一个不筛
        k = int(round(idx.size * frac))
        k = min(k, idx.size - min_keep)   # 最多筛到只剩 min_keep 张
        if k <= 0:
            continue
        order = np.argsort(-mean_loss[idx])   # 损失从大到小
        flag[idx[order[:k]]] = True
    return flag


# ------------------------------------------------------- 推理期先验修正
def prior_correction(logits: torch.Tensor, target_prior: np.ndarray) -> torch.Tensor:
    """把预测分布对齐到目标先验：``p_j <- p_j / q_j`` 后重新归一化。

    长尾场景下直接 argmax 会系统性偏向头类；用目标先验做除法修正可显著
    改善宏平均与整体准确率。``q_j`` 为训练集观测先验（或给定先验）。
    """
    q = torch.as_tensor(target_prior, dtype=logits.dtype, device=logits.device)
    q = q.clamp_min(1e-12)
    p = F.softmax(logits, dim=1)
    return (p / q).clamp_min(1e-12)


# =============================================================================
# 软目标支持（2026-10-06 新增，**纯追加**：不改任何现有函数 ⇒ 75.0641 复现基准不受影响）
# =============================================================================
# 动机：数据集含「同一文件被复制到多个类目录」（1,877 组 / 3,884 张，文件 md5 判定）。
#   组内标签互斥 ⇒ 对**单标签**任务必有一错，但**无物种名 ⇒ 无法判断哪个标签正确**。
#   方案（用户 2026-10-06 定的规格）：**组总权重=1**（组内每张 1/m，权重在样本层处理）
#   + **软目标在组内不同标签间均分**（标签不确定性在损失层处理）。
#
# ★ 为什么"软标签"必须做成新函数而不是给 `sce_loss` 加参数：
#   现有 `sce_loss` 用 `F.nll_loss(logp, y)` 与 `F.one_hot(y)`，都只吃**硬标签**。
#   若改其签名/分支，会动到 75.0641（`--kind sce`）与 74.3（`--kind sce_legacy`）的复现路径。
#   ⇒ 新增独立函数，并**断言 u 为 one-hot 时与现有实现逐位一致**（见 `soft_selfcheck`）。


def sce_loss_soft(logits: torch.Tensor, u: torch.Tensor,
                  alpha: float = 0.1, beta: float = 0.05,
                  reduction: str = "none") -> torch.Tensor:
    """软目标版（标准 SCE 形式）：`u` 为 (B,K) 概率分布（每行非负且和为 1）。

    CE  = -Σ_j u_j·log p_j
    RCE = -Σ_j p_j·log clip(u_j, 1e-4, 1)

    ★ 与 one-hot 的关系（**必须成立，否则等于换了目标函数**）：
      u = one_hot(y) ⇒ CE = -log p_y（= nll_loss）；
             RCE = -(p_y·log1 + Σ_{j≠y} p_j·log 1e-4) = 9.21·(1 - p_y)
      ⇒ 退化为标准 SCE ✔（`soft_selfcheck()` 逐元素断言这一点）
    ★ 软目标下 RCE 的含义：不再"抬高某一个类"，而是**相对排斥候选集合之外**的类
      （候选集合内系数 log m，集合外系数 9.21；因 log m ≤ log 750 ≈ 6.62 < 9.21，
        集合内的概率被保留得相对更多 ⇒ 方向仍是"偏向候选集合"，符合设计意图）。
    """
    logp = F.log_softmax(logits, dim=1)
    ce = -(u * logp).sum(dim=1)
    rce = -(logp.exp() * torch.log(u.clamp(1e-4, 1.0))).sum(dim=1)
    out = alpha * ce + beta * rce
    return out if reduction == "none" else out.mean()


def gce_loss_soft(logits: torch.Tensor, u: torch.Tensor, q: float = 0.7,
                  reduction: str = "none") -> torch.Tensor:
    """软目标版 GCE：``(1 - Σ_j u_j·p_j^q) / q``（Zhang & Sabuncu 2018 的软目标推广）。

    ★ 与 one-hot 的关系（**必须成立，否则等于换了目标函数**）：
      u = one_hot(y) ⇒ Σ_j u_j·p_j^q = p_y^q ⇒ 退化为 ``(1-p_y^q)/q`` ＝ `gce_loss` ✔
      （`soft_selfcheck()` 逐元素断言这一点）
    ★ 软目标下的含义：不再只看真实类概率，而是对**候选集合整体**做 q 次幂的
      饱和惩罚（q<1 时对低概率样本梯度有界 ⇒ 抗噪）。
    """
    p = F.softmax(logits, dim=1).clamp(1e-7, 1.0)
    out = (1.0 - (u * p.pow(q)).sum(dim=1)) / q
    return out if reduction == "none" else out.mean()


def ce_loss_soft(logits: torch.Tensor, u: torch.Tensor,
                 label_smooth: float = 0.0,
                 reduction: str = "none") -> torch.Tensor:
    """软目标版 CE（`--kind ce` 用）。u 为 one-hot 时逐位等于 `F.nll_loss`。"""
    logp = F.log_softmax(logits, dim=1)
    if label_smooth > 0.0:
        K = logits.shape[1]
        u = (1.0 - label_smooth) * u + label_smooth / K
        u = u / u.sum(dim=1, keepdim=True)
    ce = -(u * logp).sum(dim=1)
    return ce if reduction == "none" else ce.mean()


def sce_loss_legacy_soft(logits: torch.Tensor, u: torch.Tensor,
                         alpha: float = 0.1, beta: float = 0.05,
                         reduction: str = "none") -> torch.Tensor:
    """历史实现（`--kind sce_legacy`）的软目标版；**仅用于复现口径对照**，非新增实验用。"""
    logp = F.log_softmax(logits, dim=1)
    ce = -(u * logp).sum(dim=1)
    rce = -(logp.exp() * logp).sum(dim=1)          # 历史：RCE = 预测熵 H(p)
    out = alpha * ce + beta * rce
    return out if reduction == "none" else out.mean()


def compute_loss_soft(logits: torch.Tensor, u: torch.Tensor,
                      y_hard: torch.Tensor,
                      sample_w: torch.Tensor | None = None,
                      T: torch.Tensor | None = None,
                      kind: str = "sce", alpha: float = 0.1, beta: float = 0.05,
                      q: float = 0.7, label_smooth: float = 0.0,
                      reduction: str = "mean") -> torch.Tensor:
    """软目标统一入口（**只支持**不依赖转移矩阵 T 的 kind）。

    ★ `T` 被**显式接受但不用**：训练侧 `_lkw` 统一带 `T=...`，
      不接就会 `TypeError`（10-06 实跑踩到：24 秒就崩，所幸没白跑 2 小时）。
      带 T 进来时 `kind` 多半是 `forward_*`，已在下面显式拒绝。
    ★ `q` 自 2026-10-09 起**被 gce 分支使用**（此前仅为兼容签名而接受）。
    """
    if kind.startswith("forward"):
        raise NotImplementedError(
            f"[soft] kind={kind}（前向校正）在软目标下未定义 ⇒ 拒绝静默回退")
    if kind == "ce":
        per = ce_loss_soft(logits, u, label_smooth=label_smooth, reduction="none")
    elif kind == "sce":
        per = sce_loss_soft(logits, u, alpha=alpha, beta=beta, reduction="none")
    elif kind == "sce_legacy":
        per = sce_loss_legacy_soft(logits, u, alpha=alpha, beta=beta, reduction="none")
    elif kind == "gce":
        per = gce_loss_soft(logits, u, q=q, reduction="none")
    else:
        raise NotImplementedError(f"[soft] 该 kind 不支持软目标：{kind}")
    if sample_w is not None:
        per = per * sample_w
    return per if reduction == "none" else per.mean()


def soft_selfcheck(K: int = 750, seed: int = 0) -> dict:
    """★ 自证：u 取 one-hot 时，软目标实现必须与现有硬标签实现**逐位一致**。

    这是「新增软目标没有偷偷换掉目标函数」的判据（铁律 86 的精神）。
    """
    torch.manual_seed(seed)
    lg = torch.randn(8, K)
    y = torch.arange(8) * 13 % K
    a, b = 0.1, 0.05
    u = torch.nn.functional.one_hot(y, K).to(lg.dtype)
    out = {}
    d1 = (sce_loss_soft(lg, u, a, b) - sce_loss(lg, y, a, b)).abs().max().item()
    d2 = (ce_loss_soft(lg, u) - torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(lg, 1), y, reduction="none")).abs().max().item()
    d3 = (sce_loss_legacy_soft(lg, u, a, b)
          - sce_loss_legacy(lg, y, a, b)).abs().max().item()
    d4 = (gce_loss_soft(lg, u, 0.7) - gce_loss(lg, y, 0.7)).abs().max().item()
    out["sce_soft_vs_sce"] = d1
    out["ce_soft_vs_ce"] = d2
    out["sce_legacy_soft_vs_legacy"] = d3
    out["gce_soft_vs_gce"] = d4
    # 软目标确实不同于硬目标（否则说明实现退化成 no-op）
    u2 = torch.zeros(8, K)
    u2[torch.arange(8), torch.tensor([(i * 7 + 3) % K for i in range(8)])] = 1.0
    u2[torch.arange(8), torch.tensor([(i * 11 + 5) % K for i in range(8)])] += 1.0
    u2 = u2 / u2.sum(dim=1, keepdim=True)
    out["soft_vs_hard_differs"] = float(
        (sce_loss_soft(lg, u2, a, b) - sce_loss(lg, y, a, b)).abs().max().item())
    return out
