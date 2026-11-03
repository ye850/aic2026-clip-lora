# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""Visual Prompt Tuning（VPT）—— 给 CLIP 视觉塔注入可学习的 prompt token。

背景与动机
----------
本项目此前所有"视觉侧"的改动都是**旋钮级**（解冻层数、LoRA 秩、目标模块、
分辨率），而 已确认旋钮级的 flip 精度全在 46~52%（等于抛硬币）。
VPT 是**机制级**改动：它往视觉塔的输入序列里插入若干**从零开始训练**的
向量，让冻结的预训练权重在不被修改的前提下获得新的"上下文条件"。

为什么注入点选在 ``vision_model.embeddings`` 的输出上
------------------------------------------------------
transformers 5.x 的 ``CLIPVisionTransformer.forward`` 是：

    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=...)
    hidden_states = self.pre_layrnorm(hidden_states)
    encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
    pooled_output = self.post_layernorm(encoder_outputs.last_hidden_state[:, 0, :])

即在 ``embeddings`` 之后依次做 pre_layrnorm → 12 层 encoder → 取 **index 0** 做 pool。
把 prompt 拼在 embeddings 输出的 **CLS 之后、patch 之前**，则：

* CLS 仍稳居 index 0 ⇒ ``pooled_output`` 的语义完全不变，下游分类头无需改动；
* prompt 与 patch 一样走 pre_layrnorm 与全部 12 层注意力 ⇒ 是真正的 VPT；
* 序列长度从 ``1+P`` 变 ``1+N+P``，注意力是全序列双向的、天然支持变长，
  **不需要改任何 attention mask**（这是选它而不选"每层插一次"的 VPT-Deep 的
  主要原因 —— Deep 需要逐层对齐 mask，改错会静默出错）。

★ 位置编码的处理
    位置编码在 ``embeddings.forward`` **内部**就已加到 patch 上，我们拼接发生在
    它**之后** ⇒ prompt 天然不带位置编码（与 VPT 原论文一致，prompt 是自由向量）。
    同时 ``interpolate_pos_encoding`` 的网格推断发生在原件内部，看到的是**原始**
    的 1+P 序列 ⇒ 448 分辨率下（14×14=196 patch）插值逻辑完全不受影响。

★ 与 LoRA 的关系
    VPT 与 LoRA 是**正交**的：LoRA 改的是权重（低秩旁路），VPT 改的是输入。
    二者可单独开、也可同时开，便于做「VPT-only / LoRA-only / 两者」三臂对照。

★ 没有"零初始化等价性"（与 LoRA 不同，别照抄那条自检）
    LoRA 靠 B=0 做到"训练起点逐位等于原模型"。VPT **做不到**：多出来的 token
    会真实参与注意力 softmax（分母变大），且 prompt 经过 ``pre_layrnorm`` 后
    **模长被归一化**、只有方向有意义 ⇒ 把 init 缩小并不能让扰动趋于 0。
    因此本模块的自检不是"δ≈0"，而是"**δ 非零且有限**"——
    一个恒等映射的 wrapper 才是失败态（见 ``report``）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# 参数名里带这个串才被认作 VPT 参数。刻意避开 "lora_"，
# 因为 ``lora.py::lora_parameters(strict=True)`` 按 `"lora_" in name` 判定、
# ``mark_only_lora_trainable`` 更会把**不含 lora_ 的参数一律冻结** ——
# 名字若撞上 "lora_"，两组参数会互相干扰。
VPT_TAG = "vpt_prompt"


class VPTShallowEmbeddings(nn.Module):
    """包装 ``CLIPVisionEmbeddings``，在其输出序列的 CLS 之后插入 N 个 prompt token。

    输出序列：``[CLS] + [P1..PN] + [patch1..patchP]``
    """

    def __init__(self, orig: nn.Module, n_prompt: int, dim: int,
                 init_std: float = 0.02):
        super().__init__()
        if n_prompt <= 0:
            raise ValueError(f"n_prompt 必须为正，收到 {n_prompt}")
        self.orig = orig
        self.n_prompt = int(n_prompt)
        self.prompt_dim = int(dim)
        # 参数名带 VPT_TAG，便于按名精确分组（不要用 requires_grad 反查）
        self.vpt_prompt = nn.Parameter(torch.empty(1, self.n_prompt, dim))
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.vpt_prompt, mean=0.0, std=float(init_std))

    def forward(self, pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        x = self.orig(pixel_values, interpolate_pos_encoding)  # (B, 1+P, D)
        if x.dim() != 3:
            raise RuntimeError(f"[vpt] embeddings 输出维度异常：{tuple(x.shape)}")
        # dtype 对齐：AMP autocast 下 x 是 fp16、参数是 fp32。
        # 直接 cat 会触发隐式类型提升（且反向时 fp32 副本白白占显存）。
        p = self.vpt_prompt.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
        return torch.cat([x[:, :1], p, x[:, 1:]], dim=1)

    # 让 .image_size / .position_embedding / .interpolate_pos_encoding 等
    # 外部可能访问的属性继续可见 —— 否则「包装」会悄悄削掉原模块的接口。
    #
    # ★★ 实现要点（两次踩坑的教训）：
    #   1. 必须**先链到 nn.Module.__getattr__**：Parameter / Buffer / 子模块
    #      （含 `orig` 本身）都不在实例 __dict__ 里，它们存在 `_parameters` /
    #      `_buffers` / `_modules` 中，只能由 `nn.Module.__getattr__` 取到。
    #      若跳过父类直接 `getattr(self.orig, name)`，会废掉属性查找 ⇒
    #      连 `self.vpt_prompt` 都读不到（__init__ 里就炸，闸门抓到过）。
    #   2. **不要把 `orig` 写进拒绝名单**：`orig` 是子模块，父类的
    #      `__getattr__` 能正确返回它。早先版本把 `"orig"` 列入拒绝名单，
    #      导致 `forward` 里的 `self.orig(...)` 直接抛 AttributeError ——
    #      这是"为了防递归而误伤子模块"的典型反例。
    #   3. 只有当父类**确实没有**这个属性时，才回退到 `orig`（代理语义）。
    def __getattr__(self, name: str):
        if name == "orig":
            # 子模块应由父类查找返回；仅在极端情况下兜底
            try:
                return super().__getattr__(name)
            except AttributeError:
                return self.__dict__["_modules"]["orig"]
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        mods = self.__dict__.get("_modules")
        if mods is not None and "orig" in mods:
            try:
                return getattr(mods["orig"], name)
            except AttributeError:
                pass
        raise AttributeError(name)

    def extra_repr(self) -> str:
        return f"n_prompt={self.n_prompt}, dim={self.prompt_dim}"


def _vision_tower(model: nn.Module) -> nn.Module:
    vt = getattr(model, "vision_model", None) or getattr(model, "visual", None)
    if vt is None:
        raise RuntimeError("[vpt] 找不到视觉塔（vision_model / visual）")
    return vt


def _resolve_by_path(model: nn.Module, path: str) -> nn.Module:
    """按 ``named_modules()`` 给出的**点分路径**取回子模块。

    ★ 为什么不直接用 ``getattr(vt, "<leaf>")``：
      transformers 5.x 里 ``CLIPModel.vision_model`` 是 ``CLIPVisionModel``，
      它把 ``embeddings`` / ``encoder`` / ``pre_layrnorm`` 这类属性**代理**到
      内层的 ``CLIPVisionTransformer``。于是
      ``setattr(vt, "embeddings", wrapper)`` 会**注册到代理层**，
      而真正跑前向的是内层那个 ``embeddings`` ⇒ **注入被静默绕开**：
      参数在、梯度为零、图像嵌入一字不变、loss 照降。
      实测判据就是把 δ 打出来（见 ``report`` 的第 3 项）。
      ⇒ 只认 ``named_modules()`` 的真实树路径，它是 ``_modules`` 递归出来的，
        不存在代理歧义。
    """
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _find_embedding_candidates(model: nn.Module) -> list:
    """列出所有 CLIPVisionEmbeddings 的真实路径（按类名匹配，兼容不同版本）。"""
    out = []
    for n, m in model.named_modules():
        if type(m).__name__ == "CLIPVisionEmbeddings" or isinstance(
                m, VPTShallowEmbeddings):
            out.append((n, m))
    return out


def inject_vpt(model: nn.Module, n_prompt: int, init_std: float = 0.02,
               verbose: bool = True) -> dict:
    """在 ``model`` 的视觉塔里注入 VPT-Shallow。

    幂等：若已注入则先解包再重注（避免套娃）。
    返回统计信息 dict，调用方应当落盘/打印。
    """
    if n_prompt <= 0:
        return {"n_prompt": 0, "params": 0, "dim": 0, "injected": False}

    _vision_tower(model)   # 仅做存在性校验，真正的定位走 named_modules

    # ★ 幂等：先把可能存在的旧包装拆掉，避免"套娃"（长度会变成 1+2N+P）
    for nm, m in _find_embedding_candidates(model):
        if isinstance(m, VPTShallowEmbeddings):
            parent = _resolve_by_path(model, nm.rpartition(".")[0])
            setattr(parent, nm.rpartition(".")[2], m.orig)
            if verbose:
                print(f"[vpt] 检测到已注入的 prompt（{m.n_prompt} 个）@ {nm}，"
                      f"已解包后再重注")

    cands = _find_embedding_candidates(model)
    if not cands:
        raise RuntimeError(
            "[vpt] 找不到 CLIPVisionEmbeddings。视觉塔内一个都没匹配到 —— "
            "说明本版本的类名不同，请先跑 scripts/probe_vpt_modules.py 核对。")
    if len(cands) > 1:
        names = [n for n, _ in cands]
        raise RuntimeError(
            f"[vpt] 匹配到 {len(cands)} 个 embeddings，无法确定该注入哪一个："
            f"{names}。拒绝猜测 —— 注错了就是静默无效。")
    attr_path, orig = cands[0]
    if verbose:
        print(f"[vpt] 注入目标（真实路径）: {attr_path}  "
              f"[{type(orig).__name__}]")

    # dim 从位置编码/类别 token 推出，别写死 768
    dim = None
    pe = getattr(orig, "position_embedding", None)
    if pe is not None and hasattr(pe, "weight"):
        dim = int(pe.weight.shape[-1])
    if dim is None:
        ce = getattr(orig, "class_embedding", None)
        if ce is not None:
            dim = int(ce.numel())
    if dim is None:
        pw = getattr(getattr(orig, "patch_embedding", None), "weight", None)
        if pw is not None:
            dim = int(pw.shape[0])
    if dim is None:
        raise RuntimeError("[vpt] 无法推断视觉塔隐藏维度")

    # ★ 设备/ dtype 对齐（与 lora.py 同一个坑）：
    #   新模块默认在 CPU，而模型可能已在 GPU 上，不对齐会让前向直接抛
    #   RuntimeError: Expected all tensors to be on the same device。
    base_device = next(model.parameters()).device
    base_dtype = next(model.parameters()).dtype

    new_mod = VPTShallowEmbeddings(orig, n_prompt, dim, init_std=init_std)
    new_mod = new_mod.to(device=base_device, dtype=base_dtype)
    # 显式置为可训练：inject_lora 会整体 requires_grad=False，
    # 若 VPT 在那之前注入就会被静默冻住（参数在、梯度为零、训练毫无变化）。
    new_mod.vpt_prompt.requires_grad = True
    _parent_path, _, _leaf = attr_path.rpartition(".")
    _parent = _resolve_by_path(model, _parent_path) if _parent_path else model
    setattr(_parent, _leaf, new_mod)

    # ★★ 结构性验证：确认"注入真的生效"，而不是被属性代理绕开。
    #   判据 —— 包装后该模块自身的输出序列长度必须变成 1+N+P。
    #   只看"参数在 state_dict 里"是不够的，那样注入到代理层也照样成立。
    seq_len = None
    try:
        with torch.no_grad():
            xs = _placeholder_input(model, base_device)
            seq_len = int(new_mod(xs, interpolate_pos_encoding=False).shape[1])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"[vpt] 注入后自测前向失败：{type(e).__name__}: {e}") from e
    want = 1 + int(n_prompt) + 49   # 224 网格；仅用于"有没有变长"的量级判断
    if seq_len <= 50:
        raise SystemExit(
            f"[vpt] 🛑 注入未生效：包装模块输出序列长度 {seq_len}，"
            f"未随 prompt 增长 ⇒ 很可能挂到了属性代理层，真实前向仍走原件。")

    info = {
        "n_prompt": int(n_prompt),
        "params": int(new_mod.vpt_prompt.numel()),
        "dim": int(dim),
        "init_std": float(init_std),
        "device": str(base_device),
        "dtype": str(base_dtype),
        "attr_path": f"{attr_path}.vpt_prompt",
        "self_seq_len": seq_len,
        "injected": True,
    }
    if verbose:
        print(f"[vpt] ★ 注入 VPT-Shallow：{n_prompt} 个 prompt token，"
              f"dim={dim}，新增参数 {info['params']:,}"
              f"（{base_device}/{base_dtype}，init_std={init_std}）")
        print(f"[vpt]   生效复验：包装模块输出 (B,{seq_len},D) "
              f"= 1+{n_prompt}+49  ✔")
    return info


def _placeholder_input(model: nn.Module, device) -> torch.Tensor:
    """造一份与视觉塔配置匹配的占位输入（仅用于自测，不进训练）。"""
    pe = None
    for _n, m in model.named_modules():
        if type(m).__name__ == "CLIPVisionEmbeddings":
            pe = getattr(m, "position_embedding", None)
            break
    side = 224
    if pe is not None and getattr(pe, "weight", None) is not None:
        n_pos = int(pe.weight.shape[0]) - 1
        g = int(round(n_pos ** 0.5))
        if g * g == n_pos:
            side = g * 32
    return torch.zeros(1, 3, side, side, device=device)


def find_vpt_module(model: nn.Module) -> VPTShallowEmbeddings | None:
    """返回已注入的 VPT 包装模块，未注入时返回 None。

    走 ``named_modules()`` 的真实路径，避免属性代理把查找引到别处。
    """
    for _n, m in model.named_modules():
        if isinstance(m, VPTShallowEmbeddings):
            return m
    return None


def vpt_attr_path(model: nn.Module) -> str | None:
    """返回 VPT 包装模块在模型里的真实点分路径（便于核对与排障）。"""
    for n, m in model.named_modules():
        if isinstance(m, VPTShallowEmbeddings):
            return n
    return None


def vpt_parameters(model: nn.Module, only_trainable: bool = True) -> list:
    """按**参数名**精确收集 VPT 参数。

    ⚠️ 刻意不用 ``[p for p in model.parameters() if p.requires_grad]``：
    那会把分类头、logit_scale、解冻的骨干一起收进来，导致它们在优化器分组里
    被 VPT 组抢走、再用各自的组去重剔掉 ⇒ **静默用了错的学习率**。
    （lora.py 里同一个坑已经踩过一次。）
    """
    out = []
    for name, p in model.named_parameters():
        if VPT_TAG not in name:
            continue
        if only_trainable and not p.requires_grad:
            continue
        out.append(p)
    return out


def vpt_state_dict(model: nn.Module) -> dict:
    """只抽 VPT 权重（几 KB），便于快速比对/单独保存。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if VPT_TAG in k}


def report(model: nn.Module, img_size: int, batch: int = 2,
           device: str | None = None, verbose: bool = True) -> dict:
    """注入后的自检（在真机上跑，判据见各字段说明）。

    返回 dict，其中 ``ok`` 为总判定。任何一项不通过都应当**拒绝开跑** ——
    VPT 的失败态几乎都是静默的：程序照跑、loss 照降，但学的是别的东西。

    检查项：
      1. seq_len == 1 + N + (img_size/32)^2
      2. δ 非零且有限  —— 恒等映射的 wrapper 是失败态（VPT 没有零初始化等价性）
      3. 梯度流向      —— prompt 有非零梯度；未解冻的骨干 ``.grad is None``
      4. 参数账目      —— 可训练参数增量 == N×dim，且不与其它组重叠
    """
    mod = find_vpt_module(model)
    if mod is None:
        raise RuntimeError("[vpt] report() 必须在 inject_vpt() 之后调用")

    dev = torch.device(device or next(model.parameters()).device)
    grid = img_size // 32
    n_patch = grid * grid
    want_len = 1 + mod.n_prompt + n_patch

    model.eval()
    x = torch.zeros(batch, 3, img_size, img_size, device=dev)
    interp = img_size != 224
    res = {"n_prompt": mod.n_prompt, "dim": mod.prompt_dim,
           "attr_path": vpt_attr_path(model),
           "grid": f"{grid}x{grid}", "n_patch": n_patch,
           "expect_seq_len": want_len}

    # --- 1. 序列长度 + 2. δ ---
    with torch.no_grad():
        seq_new = mod(x, interpolate_pos_encoding=interp)
        res["seq_len"] = int(seq_new.shape[1])
        seq_len_ok = res["seq_len"] == want_len
        # 剥掉 prompt 后应当与"原件输出"逐位一致 —— 证明我们只是插入、
        # 没有改动 patch/CLS 的任何数值（这是比"δ>0"更强的结构性证据）。
        stripped = torch.cat([seq_new[:, :1], seq_new[:, 1 + mod.n_prompt:]], dim=1)
        seq_old = mod.orig(x, interpolate_pos_encoding=interp)
        res["strip_max_abs_diff"] = float((stripped - seq_old).abs().max().item())
        strip_ok = res["strip_max_abs_diff"] < 1e-5

        # 端到端：图像嵌入是否真的变了
        f_new = _embed(model, x, interp)
        bak = mod.vpt_prompt.data.clone()
        mod.vpt_prompt.data.zero_()
        f_zero = _embed(model, x, interp)
        mod.vpt_prompt.data.copy_(bak)
        res["emb_delta_vs_zero_prompt"] = float(
            (f_new - f_zero).abs().max().item())
        delta_finite = bool(torch.isfinite(f_new).all().item())
        delta_nonzero = res["emb_delta_vs_zero_prompt"] > 1e-6

    # --- 3. 梯度流向 ---
    model.train()
    if dev.type == "cuda":
        model.to(dev)
    xg = torch.zeros(batch, 3, img_size, img_size, device=dev)
    out = _embed(model, xg, interp)
    loss = out.float().pow(2).sum()
    loss.backward()
    g = mod.vpt_prompt.grad
    res["vpt_grad_absmax"] = float(g.abs().max().item()) if g is not None else 0.0
    res["vpt_grad_is_none"] = g is None
    grad_ok = (g is not None) and bool(torch.isfinite(g).all().item()) \
        and res["vpt_grad_absmax"] > 0.0

    # 冻结骨干必须没有梯度（名字里不含 vpt_prompt / lora_）
    leaked = [n for n, p in model.named_parameters()
              if p.requires_grad is False and p.grad is not None
              and not torch.all(p.grad == 0)]
    res["frozen_params_with_grad"] = len(leaked)
    res["frozen_examples"] = leaked[:3]
    frozen_ok = len(leaked) == 0

    # --- 4. 参数账目 ---
    n_vpt_params = sum(p.numel() for p in vpt_parameters(model, False))
    res["vpt_params_found"] = n_vpt_params
    res["vpt_params_expect"] = mod.n_prompt * mod.prompt_dim
    # 重叠检查：VPT 参数不得出现在 LoRA 名集合里
    vpt_names = {n for n, _ in model.named_parameters() if VPT_TAG in n}
    lora_names = {n for n, _ in model.named_parameters() if "lora_" in n}
    res["name_overlap_with_lora"] = sorted(vpt_names & lora_names)
    acct_ok = (n_vpt_params == res["vpt_params_expect"]
               and not res["name_overlap_with_lora"])

    res["check_seq_len"] = bool(seq_len_ok)
    res["check_strip_identity"] = bool(strip_ok)
    res["check_delta_nonzero"] = bool(delta_nonzero and delta_finite)
    res["check_grad_flow"] = bool(grad_ok)
    res["check_frozen_clean"] = bool(frozen_ok)
    res["check_param_accounting"] = bool(acct_ok)
    res["ok"] = all([seq_len_ok, strip_ok, delta_nonzero, delta_finite,
                     grad_ok, frozen_ok, acct_ok])

    if verbose:
        def _m(b):
            return "PASS" if b else "**FAIL**"
        print("[vpt-report] " + "-" * 56)
        print(f"[vpt-report] n_prompt={mod.n_prompt} dim={mod.prompt_dim} "
              f"grid={res['grid']}")
        print(f"[vpt-report] 真实路径   {res['attr_path']}")
        print(f"[vpt-report]  1 序列长度  {res['seq_len']} (期望 {want_len})  "
              f"{_m(seq_len_ok)}")
        print(f"[vpt-report]  2 剥离恒等  max|Δ|={res['strip_max_abs_diff']:.3e}  "
              f"{_m(strip_ok)}   ← 证明只插入、未改动 CLS/patch 数值")
        print(f"[vpt-report]  3 端到端 δ  {res['emb_delta_vs_zero_prompt']:.6f}"
              f"（非零且有限）  {_m(delta_nonzero and delta_finite)}")
        print(f"[vpt-report]  4 梯度流向  prompt grad max={res['vpt_grad_absmax']:.3e}"
              f"  {_m(grad_ok)}")
        print(f"[vpt-report]  5 冻结干净  带梯度的冻结参数 {len(leaked)} 个  "
              f"{_m(frozen_ok)}")
        print(f"[vpt-report]  6 参数账目  {n_vpt_params} / "
              f"{res['vpt_params_expect']}，与 LoRA 重名 "
              f"{len(res['name_overlap_with_lora'])}  {_m(acct_ok)}")
        print(f"[vpt-report] 总判定: {'OK' if res['ok'] else 'FAILED'}")
        print("[vpt-report] " + "-" * 56)

    # 收尾：清掉自检留下的梯度，避免污染真实训练的第一次 backward
    model.zero_grad(set_to_none=True)
    model.eval()
    return res


def _embed(model: nn.Module, x: torch.Tensor, interp: bool) -> torch.Tensor:
    """取"图像嵌入"（投射后 512 维），兼容 transformers 新旧返回类型。"""
    from extract import take_embeds
    return take_embeds(model.get_image_features(
        pixel_values=x, interpolate_pos_encoding=interp))


def param_level_count(model: nn.Module) -> int:
    """可训练参数数（用于与 lora.py 的统计口径对齐）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log2_ratio(n_prompt: int, n_patch: int) -> float:
    """prompt 占（prompt+patch）的比例 —— 用于判断是否"喧宾夺主"。"""
    if n_prompt + n_patch <= 0:
        return float("nan")
    return math.log2((n_prompt + n_patch) / max(n_patch, 1))
