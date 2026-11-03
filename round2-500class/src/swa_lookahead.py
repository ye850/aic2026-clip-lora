# -*- coding: utf-8 -*-
"""SWA 与 Lookahead —— 优化器层扩展（2026-10-08）

--------------------------------------------------------------------
为什么加它们（2026-10-08）
--------------------------------------------------------------------
SAM 已上榜验证 +1.55pp（76.8107），证了**「优化器层」能动**这个方向。
grep 全仓当初发现：SAM = 0 ｜ SWA = 0 ｜ sharpness = 0 ｜ lookahead = 0
⇒ 优化层是彻底空白。SAM 只是第一块砖，同族的 SWA / Lookahead 还没碰。

两者都**正对「训练-测试落差」**（本项目差距主体的已定量来源）：
① SWA 对训练末期的权重轨迹做平均 ⇒ 落在「平坦极小值」附近 ⇒ 跨域更稳。
② Lookahead 让快权重探索、慢权重收敛 ⇒ 优化轨迹更平滑 ⇒ 同理抗输入扰动。

--------------------------------------------------------------------
合规（与 sam.py 同判据）
--------------------------------------------------------------------
只改**优化过程**，不加参数、不改结构、不引入外部数据/权重、
不碰测试集 ⇒ 纯训练侧，**完全合规**。

--------------------------------------------------------------------
与本项目训练循环的交互（这是本文件存在的理由）
--------------------------------------------------------------------
`finetune_lora.py` 里：
- 优化器建在 `opt = torch.optim.AdamW(...)`。
- Lookahead 包在 AdamW 外：`opt = Lookahead(opt, k, alpha)`。
- SAM 再包在 Lookahead 外：`SAM(params, opt, rho)` ⇒ SAM 的
  `second_step` 调 `base.step()` = Lookahead.step()，天然兼容。
- SWA 与优化器正交：每轮结束 `swa.update(ep)` 累积权重，
  训练完 `swa.apply_to(model)` 把平均权重灌回模型当提交权重。
- 三者都可独立开关，默认全关 ⇒ 冠军复现路径零影响。

--------------------------------------------------------------------
★★ 冒烟实录（2026-10-08 08:5x，铁律 239 模式重演）
--------------------------------------------------------------------
首版 Lookahead 是普通包装类 ⇒ 云端冒烟 2 分钟即炸：
    TypeError: Lookahead is not an Optimizer（OneCycleLR 的 isinstance 检查）
⇒ 本类**必须真继承 `torch.optim.Optimizer`**，且构造时把 inner 的
param_groups **同一批 dict 对象**（不是拷贝）登记进 self.param_groups：
  - isinstance 检查（OneCycleLR / GradScaler）通过；
  - scheduler 写 `group['lr']` 落在共享 dict ⇒ inner AdamW 立即可见；
  - `state_dict / load_state_dict` 仍转发 inner（否则 ckpt 丢动量态）；
  - `add_param_group` 构造期**不得**转发 inner（组已在 inner 里，
    重复添加会触发 "parameters appear in more than one group"）。
"""
from __future__ import annotations

import torch


class SWA:
    """随机权重平均（Stochastic Weight Averaging, Izmailov et al., 2018）。

    在 `start_epoch` 之后的每个 epoch 结束时，把模型**当前权重**做滑动平均：
        avg = avg * (n-1)/n + w / n
    训练结束后用 `apply_to(model)` 把平均权重灌回模型，作为最终提交权重。

    ★ 只平均可训练参数（`named_parameters`），不动 buffer（如 BN 的 running stat）。
    ★ 与 EMA 正交：EMA 是训练期影子、用于「选优」；SWA 是训练末期对 raw 权重的
      轨迹平均，用于「最终泛化」。两者可同时开，互不干扰。
    """

    def __init__(self, model, start_epoch: int = 5):
        self.model = model
        self.start = int(start_epoch)
        self.n = 0
        self.avg = None
        # 只记可训练参数键；buffer（非参数）不参与平均
        self._pkeys = {k for k, _ in model.named_parameters()}

    @torch.no_grad()
    def update(self, epoch: int) -> None:
        """每轮结束调用一次：epoch >= start 时把当前权重并入滑动平均。"""
        if epoch < self.start:
            return
        sd = self.model.state_dict()
        if self.avg is None:
            self.avg = {k: sd[k].detach().float().clone() for k in self._pkeys}
            self.n = 1
        else:
            self.n += 1
            inv = 1.0 / self.n
            for k in self._pkeys:
                # avg = avg*(1-1/n) + w*(1/n)
                self.avg[k].mul_(1.0 - inv).add_(sd[k].detach().float(), alpha=inv)

    @torch.no_grad()
    def apply_to(self, model) -> None:
        """把平均权重灌回模型（替换当前权重），用作提交/推理权重。"""
        if self.avg is None:
            return
        sd = model.state_dict()
        new = {k: (self.avg[k].to(v.dtype) if k in self.avg else v)
               for k, v in sd.items()}
        model.load_state_dict(new)

    @property
    def count(self) -> int:
        return self.n


class Lookahead(torch.optim.Optimizer):
    """Lookahead 优化器（Zhang et al., 2019）—— 真 Optimizer 子类。

    把任意 base optimizer 包成 Lookahead：每 `k` 个 inner step 后，
    把 slow 权重向 fast 权重插值 `alpha`，再把 fast 拉回 slow：
        slow <- slow + alpha * (fast - slow)
        fast <- slow
    ⇒ 快权重负责探索、慢权重负责收敛 ⇒ 轨迹更平滑、更易泛化。

    ★ 只改优化过程，不加参数 ⇒ 合规。
    ★ 与 SAM 兼容：`SAM(params, Lookahead(opt), rho)`，
      SAM 的 `second_step` 调 `base.step()` = Lookahead.step()。
    ★★★ 必须真继承 Optimizer（见文件头「冒烟实录」），且 param_groups
        用 inner 的**同一批 dict 对象**——调度器写 lr 才能落到 inner。
    """

    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        # ★ 自有字段先建好；`_constructing` 必须在 super().__init__ 之前，
        #   因为基类 __init__ 会逐组回调 self.add_param_group（见下）
        self.optimizer = optimizer
        self.k = int(k)
        self.alpha = float(alpha)
        self.count = 0
        self._constructing = True
        # 采集 base optimizer 管理的全部参数（同对象，sync 时原地改）
        self.params = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                self.params.append(p)
        # slow 副本（独立张量，不参与梯度）
        self.slow = [p.detach().clone() for p in self.params]
        # ★ 传 inner 的 param_groups（同一批 dict 对象）：
        #   - isinstance(optimizer, Optimizer) 通过（OneCycleLR / GradScaler 都查）；
        #   - scheduler 写 group['lr'] 落在共享 dict ⇒ inner AdamW 立即可见。
        # ★ defaults 也用 inner 的拷贝：OneCycleLR(cycle_momentum=True) 要求
        #   optimizer.defaults 含 'betas'（冒烟第 2 炸：ValueError "optimizer
        #   must support momentum or beta1"）；动量循环同样经共享 group dict
        #   写 group['betas'] ⇒ 与裸 AdamW 行为逐字一致，不引入第二变量。
        super().__init__(optimizer.param_groups, dict(optimizer.defaults))
        self._constructing = False

    def _resync(self) -> None:
        """重建参数镜像（仅在训练中 add_param_group 时才需要；本项目用不到）。"""
        self.params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.params.append(p)
        self.slow = [p.detach().clone() for p in self.params]

    def add_param_group(self, param_group) -> None:
        """同时登记到包装器与 inner。

        ⚠️ 构造期（super().__init__ 逐组回调到这里）**不得**转发 inner：
           那些组本来就在 inner 里，重复添加会直接 ValueError
           （"some parameters appear in more than one parameter group"）。
        """
        super().add_param_group(param_group)
        if not self._constructing:
            self.optimizer.add_param_group(param_group)
            self._resync()

    @torch.no_grad()
    def step(self, closure=None):
        loss = (self.optimizer.step(closure) if closure is not None
                else self.optimizer.step())
        self.count += 1
        if self.count % self.k == 0:
            for p, q in zip(self.params, self.slow):
                # slow 向 fast 插值，再把 fast 拉回 slow
                q.add_(p.detach() - q, alpha=self.alpha)
                p.copy_(q)
        return loss

    # ---------------------------------------------------------------- 交接
    # ★ state_dict/load_state_dict 必须转发 inner：
    #   基类实现序列化的是 self.state（本类恒空），会把 AdamW 的动量态丢掉，
    #   ckpt 续训就废了。zero_grad 用基类的即可（param_groups 是共享 dict，
    #   清的就是同一批参数的梯度，与转发等价）。
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        return self.optimizer.load_state_dict(sd)
