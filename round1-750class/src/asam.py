# -*- coding: utf-8 -*-
"""ASAM（Adaptive Sharpness-Aware Minimization）—— SAM 的**尺度自适应**变体

--------------------------------------------------------------------
为什么加它（2026-10-09）
--------------------------------------------------------------------
SAM 已**榜上验证 +1.55pp**（76.8107）。ASAM 是它的直接变体：SAM 的扰动

    ε = ρ · g / ‖g‖

对**所有参数一视同仁**——但我们的参数组**尺度差异极大**：
  · LoRA A：kaiming 初始化 ~0.02 量级
  · LoRA B：**零初始化**，训练中才长大
  · MLP 头（hidden 1024）：~0.01–0.1 量级
  · logit_scale：~4.6（标量）
SAM 的一视同仁 ⇒ 对尺度大的参数，ρ 相对其权重比值极小（几乎无扰动）；
对尺度小的参数，ρ 相对极大（扰动过强）。ASAM 用**逐元素尺度归一**修正：

    ε* = ρ · w² ⊙ g / ‖ |w| ⊙ g ‖₂

（Kwon et al., ICML 2021, "ASAM: Adaptive Sharpness-Aware Minimization"）

--------------------------------------------------------------------
合规（与 sam.py 同判据）
--------------------------------------------------------------------
只改**优化过程**，不加参数、不改结构、不引外部数据/权重、不碰测试集
⇒ 纯训练侧，**完全合规**。

--------------------------------------------------------------------
实现要点
--------------------------------------------------------------------
- 与 sam.py 的**唯一差别**在 first_step 的 ε 公式；second_step **逐字相同**
  （先恢复、再 base.step()，顺序不能反）。
- 分母 ‖|w|⊙g‖ 用**全局 L2**（与 SAM 的 ‖g‖ 对应、同构），
  不做逐参数归一（那会退化成纯噪声方向，破坏"全局最大化"的语义）。
- ★ 零初始化参数（LoRA B）在早期 w≈0 ⇒ ε≈0（该参数此步几乎无扰动）。
  这是 ASAM 的**固有性质**，不是 bug；B 在第一个 epoch 内即长到非零。
- AMP：与 SAM 一样必须走 fp32（scaled-grad 与扰动步长会打架），
  由 finetune_lora 侧统一强制（复用 `_sam_rho > 0` 的那条分支）。
"""
from __future__ import annotations

import torch


class ASAM:
    """把任意 base optimizer 包装成 ASAM。

    用法与 `SAM` **逐字兼容**（`first_step(scale)` / `second_step(scale)`），
    因此 `finetune_lora.py` 的训练循环零改动，只换类。
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 eps: float = 1e-12):
        if rho <= 0:
            raise ValueError(f"rho 必须 > 0（收到 {rho}）；关闭 ASAM 请走默认路径")
        self.params = [p for p in params if p.requires_grad]
        self.base = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self._e: list[torch.Tensor | None] | None = None
        # ★ 调试/自证用：记录最近一次的加权范数
        self.last = {}

    # ---------------------------------------------------------------- 工具
    @torch.no_grad()
    def _weighted_grad_norm(self, scale: float) -> torch.Tensor:
        """全局 L2 加权范数 ‖ |w| ⊙ g ‖₂（g 已 unscale）。

        ★ 注意分母用 |w| 而分子用 w²——这正是官方 ASAM 的形式：
          max 的一阶解为 ε = ρ·T²g/‖Tg‖，其中 T = |w|（逐元素）。
        """
        terms = []
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach() / scale          # ★ AMP 的 grad 被放大过，除回去
            terms.append((p.detach().abs() * g).flatten())
        if not terms:
            return torch.zeros((), device=self.params[0].device)
        return torch.norm(torch.cat(terms), p=2)

    # ---------------------------------------------------------------- 两步
    @torch.no_grad()
    def first_step(self, scale: float = 1.0) -> float:
        """① 尺度自适应扰动 `w += ρ·w²⊙g/‖|w|⊙g‖`（g 已 unscale）。

        ⚠️ 调用方随后**必须**清空梯度并重算前向——本函数**不清零**，
           以免与调用侧的 AMP/clip 逻辑打架（清零点由调用侧显式控制）。
        """
        norm = self._weighted_grad_norm(scale)
        s = self.rho / (norm + self.eps)         # ρ / ‖|w|⊙g‖
        e_list: list[torch.Tensor | None] = []
        for p in self.params:
            if p.grad is None:
                e_list.append(None)
                continue
            g = p.grad.detach() / scale          # 已 unscale
            # ★ 与 SAM 的唯一差别：乘 w²（尺度自适应），而非乘 1
            e = (p.detach().pow(2) * g) * s.to(g.device)
            p.add_(e.to(p.dtype))                # ★ 原地扰动（不建图）
            e_list.append(e.to(p.dtype))
        self._e = e_list
        self.last["wg_norm"] = float(norm)
        self.last["perturb_rho"] = float(self.rho)
        return float(norm)

    @torch.no_grad()
    def second_step(self, scale: float = 1.0) -> None:
        """② 恢复权重 `w -= ε`，然后用**扰动后的梯度**做真正的更新。

        ★ 顺序不能反：必须先恢复、再 base.step()（与 sam.py 逐字相同）。
        """
        if self._e is None:
            raise RuntimeError("second_step 前必须调用 first_step")
        for p, e in zip(self.params, self._e):
            if e is not None:
                p.sub_(e)                        # 恢复原权重
        self._e = None
        self.base.step()
        self.last["step_scale"] = float(scale)

    # ---------------------------------------------------------------- 便捷
    def zero_grad(self, set_to_none: bool = True) -> None:
        """★ 转发给 base（调用侧习惯用 `opt.zero_grad`）。"""
        self.base.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base.param_groups

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, sd):
        return self.base.load_state_dict(sd)
