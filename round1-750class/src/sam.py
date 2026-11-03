# -*- coding: utf-8 -*-
# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""SAM（Sharpness-Aware Minimization）优化器包装 —— ★ 本项目首次引入

--------------------------------------------------------------------
为什么加它（2026-10-07）
--------------------------------------------------------------------
排查"与榜一差 21.24pp 能归因到什么"时，grep 全仓发现：
    SAM = 0 文件 ｜ SWA = 0 文件 ｜ sharpness = 0 文件 ｜ lookahead = 0 文件
⇒ ★ **优化层是彻底空白**。

而 SAM **正对本项目的核心问题**：
① ★★★ SAM 找「**平坦极小值**」⇒ 提升**对输入扰动的鲁棒性** ⇒ **正是跨域泛化**。
   我们的差距主体已定量确认是**训练-测试落差**
   （冠军 2.15pp；`region_attn` 5.27pp；且落差模型证明 89% 由 `sim_max` 解释）。
② ★ **不加任何结构性参数** ⇒ 按筛选器的经验，差异率应偏低 ⇒ **风险低**
   （与 `region_attn`「加 4.8M 模块 ⇒ −3.04pp」**不是同一类**）。
③ 属官方点名「鲁棒损失函数 / 表征约束」的同族。

--------------------------------------------------------------------
算法（Foret et al., 2021）
--------------------------------------------------------------------
要最小化的目标不是 `L(w)`，而是**邻域内的最大损失**：

    min_w  max_{‖ε‖≤ρ}  L(w + ε)

一阶近似下 ε* = ρ · g/‖g‖（g 为当前梯度）。故每个优化步：
    ① 用当前梯度扰动权重：  w ← w + ρ·g/‖g‖
    ② 在扰动后的权重上重新前向/反向，得到 g'
    ③ 恢复权重：            w ← w − ρ·g/‖g‖
    ④ 用 g' 做真正的更新：   w ← w − lr·g'
⇒ ★ **代价：每步两次前向+反向**（本臂约 3.2h，普通臂 1.9h）。

--------------------------------------------------------------------
★★★ 与本项目训练循环的三个交互（**这是本文件存在的理由**）
--------------------------------------------------------------------
`finetune_lora.py` 的训练循环有 **AMP** 和 **梯度累积**，两者都与 SAM 冲突：

1. **AMP**：`scaler.scale(loss).backward()` 得到的 `p.grad` 是**放大过**的
   （scale 通常 65536）。直接拿它算 `ρ·g/‖g‖`，扰动步长会被放大 scale 倍 ⇒ 权重直接炸。
   ⇒ ★ 本实现**显式 unscale**：`p.grad / scaler.get_scale()`，
     且**不调用** `scaler.unscale_()`（那个会把 grad 原地改掉，破坏后续 `scaler.step`）。

2. **梯度累积**：`loss = loss_raw / accum` ⇒ `p.grad` 是 accum 个 batch 的平均。
   而 SAM 的第二次前向**只能用当前这一个 batch**（历史 batch 的数据已释放）。
   ⇒ 这是**已知的近似**：扰动方向来自"累积梯度"，第二次梯度来自"当前 batch"。
     ★ 当 `accum=1`（本项目冠军配方）时**二者一致，是严格的 SAM**。

3. **扰动对象**：只扰动 `requires_grad` 的参数；被冻结的骨干不动 ⇒ 与 LoRA 语义自洽。

--------------------------------------------------------------------
合规
--------------------------------------------------------------------
只改**优化过程**，不加参数、不改结构、不引入外部数据/权重、
不碰测试集 ⇒ 纯训练侧，**完全合规**。
"""
from __future__ import annotations

import torch


class SAM:
    """把任意 base optimizer（如 AdamW）包装成 SAM。

    用法（见 `finetune_lora.py` 的 `is_step` 分支）::

        sam = SAM(params, opt, rho=0.05)
        ...
        loss.backward()
        sam.first_step(scale=scaler.get_scale() if scaler.is_enabled() else 1.0)
        # 用同一批数据重算 forward/backward
        loss2.backward()
        sam.second_step(scale=...)

    ★ `first_step` 后**必须**重算前向；否则 `second_step` 用的是同一个梯度，
      等价于普通更新（白加一次开销）。
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 eps: float = 1e-12):
        if rho <= 0:
            raise ValueError(f"rho 必须 > 0（收到 {rho}）；关闭 SAM 请走默认路径")
        self.params = [p for p in params if p.requires_grad]
        self.base = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self._e: list[torch.Tensor | None] | None = None
        self._gnorm: float | None = None
        # ★ 调试/自证用：记录最近一次的扰动范数与梯度范数
        self.last = {}

    # ---------------------------------------------------------------- 工具
    def _grad_norm(self, scale: float) -> torch.Tensor:
        """全局 L2 梯度范数（**已 unscale**）。

        ★ 用「所有参数的 grad 拼成一个向量」的标准做法；
          若某参数无梯度则跳过（例如该步未被使用）。
        """
        grads = []
        for p in self.params:
            if p.grad is None:
                continue
            # ★★ 关键：AMP 的 grad 被放大了 scale 倍 ⇒ 必须除回去
            grads.append((p.grad.detach() / scale).flatten())
        if not grads:
            return torch.zeros((), device=self.params[0].device)
        return torch.norm(torch.cat(grads), p=2)

    # ---------------------------------------------------------------- 两步
    @torch.no_grad()
    def first_step(self, scale: float = 1.0) -> float:
        """① 扰动权重 `w += ρ·g/‖g‖`（g 已 unscale）；返回梯度范数（供自证）。

        ⚠️ 调用方随后**必须**清空梯度并重算前向 —— 本函数**不清零**，
           以免与调用侧的 AMP/clip 逻辑打架（清零点由调用侧显式控制）。
        """
        gnorm = self._grad_norm(scale)
        self._gnorm = float(gnorm)
        s = self.rho / (gnorm + self.eps)          # ρ/‖g‖
        e_list: list[torch.Tensor | None] = []
        for p in self.params:
            if p.grad is None:
                e_list.append(None)
                continue
            e = (p.grad.detach() / scale) * s.to(p.grad.device)   # 已 unscale
            p.add_(e.to(p.dtype))                  # ★ 原地扰动（不建图）
            e_list.append(e.to(p.dtype))
        self._e = e_list
        self.last["gnorm"] = self._gnorm
        self.last["perturb_l2"] = float(self.rho)
        return self._gnorm

    @torch.no_grad()
    def second_step(self, scale: float = 1.0) -> None:
        """② 恢复权重 `w −= ρ·g/‖g‖`，然后用**扰动后的梯度**做真正的更新。

        ★ 顺序不能反：必须先恢复、再 base.step()。
          若先 step() 再恢复，等于"在扰动权重上更新、再减去扰动"，
          **不是 SAM**（本项目对"顺序反了"这类静默错误尤其警惕）。
        """
        if self._e is None:
            raise RuntimeError("second_step 前必须调用 first_step")
        for p, e in zip(self.params, self._e):
            if e is not None:
                p.sub_(e)                          # 恢复原权重
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
