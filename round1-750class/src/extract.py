# ========================================================================
# 出处声明 / Provenance
#   作者：本队（AIC 2026 算法精英赛 · 赛题 4）
#   首版：2026-09（项目初期建立，此后持续演进）
#   末次修订：2026-10-11
#   说明：本文件不含任何赛事数据；算法动机、消融与证据见随附《技术方案》，
#         被实验证伪的方向见《技术方案》第五节。
# ========================================================================
"""CLIP ViT-B/32 特征提取（冻结骨干）。

为什么先提特征、再在特征上训分类头？
--------------------------------------------------
本机 GPU 为 RTX 4050 Laptop / **6GB 显存**，而本阶段训练集有 14.9 万张。
端到端微调 ViT-B/32 在这个显存下只能压到极小 batch，一轮成本高、不便快速迭代。

冻结骨干 + 缓存特征的路线：
  * 14.9 万张图走一遍 ViT-B/32 实测约 10.5 分钟（推理无梯度，显存占用小）；
  * 特征只有 512 维，(14.9万 x 512) float16 约 146MB，可常驻内存；
  * 之后分类头训练是「小矩阵乘法」，**每轮秒级**，可自由做消融、交叉验证、调参。

这条路线是本次时间约束下的正确取舍：把算力花在方法迭代而非重复前向。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import config as C  # ★ 必须早于 numpy/torch（config 里设 BLAS 线程限制）
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import load_image


# --------------------------------------------------------------- 模型加载
def _enable_hf_mirror() -> None:
    """国内网络环境下备用镜像（不覆盖用户已设置的 HF_ENDPOINT）。"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class _ProcShim:
    """当 CLIPProcessor 不可用时的兜底封装。

    不同 transformers 版本里 CLIPProcessor 存在与否有变动，而本工程只需要
    「图像预处理」与「文本分词」两件事。此 shim 统一接口，避免版本差异直接把
    流程打断。
    """

    def __init__(self, img_proc, tok):
        self._img, self._tok = img_proc, tok

    def __call__(self, images=None, text=None, **kw):
        if images is not None:
            return {"pixel_values":
                    self._img(images=images, return_tensors="pt")["pixel_values"]}
        return self._tok(text=text, return_tensors="pt", padding=True,
                         truncation=True, **kw)


def resolve_model_ref() -> str:
    """决定用本地权重目录还是 HF repo id。

    本地目录优先：既避开 HF 缓存布局的坑，也让总决赛复核时能完全离线复现。
    """
    d = C.CLIP_LOCAL_DIR
    if (d / "config.json").exists() and (
            (d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()):
        print(f"[clip] 使用本地权重目录 {d}")
        return str(d)
    print(f"[clip] 本地权重目录不完整，回退在线拉取 {C.CLIP_HF_ID}")
    return C.CLIP_HF_ID


def load_clip(device: str = "cuda", prefer: str = "transformers",
              local_only: bool = False):
    """加载 CLIP ViT-B/32（仅使用 OpenAI 官方公开权重，符合赛题约束）。

    ``local_only=True`` 时只用本地文件，用于断网/离线复现。
    """
    _enable_hf_mirror()
    ref = resolve_model_ref()
    is_local = Path(ref).exists()
    if prefer == "transformers":
        try:
            from transformers import CLIPModel
            kw = {"local_files_only": True} if (local_only or is_local) else {}
            model = CLIPModel.from_pretrained(ref, **kw)

            try:
                from transformers import CLIPProcessor
                proc = CLIPProcessor.from_pretrained(ref, **kw)
            except Exception:  # noqa: BLE001
                from transformers import CLIPImageProcessor, CLIPTokenizer
                proc = _ProcShim(
                    CLIPImageProcessor.from_pretrained(ref, **kw),
                    CLIPTokenizer.from_pretrained(ref, **kw))
                print("[clip] 使用 CLIPImageProcessor + CLIPTokenizer 兜底封装")

            model.eval().to(device)
            print(f"[clip] transformers 载入 {ref} "
                  f"(投影维度 {model.config.projection_dim})")
            return ("transformers", model, proc)
        except Exception as e:  # noqa: BLE001
            if local_only:
                raise
            print(f"[clip] transformers 载入失败（{type(e).__name__}: {e}），"
                  f"尝试 open_clip")

    import open_clip
    name, pretrained = C.CLIP_OPENCLIP_ID
    model, _, proc = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=device)
    model.eval()
    print(f"[clip] open_clip 载入 {name}/{pretrained}")
    return ("open_clip", model, proc)


def _preprocess_batch(kind: str, model, proc, images):
    if kind == "transformers":
        return proc(images=images, return_tensors="pt")["pixel_values"]
    return torch.stack([proc(im) for im in images])


def take_embeds(out):
    """从 get_image_features / get_text_features 的返回里取出嵌入向量。

    ⚠️ **transformers 5.x 的破坏性变更**：这两个方法过去直接返回 ``Tensor``，
    5.x 起改为返回 ``BaseModelOutputWithPooling`` 对象（实测 5.17.0）。
    该项目投影后的 512 维向量在 ``pooler_output`` 里（``last_hidden_state``
    是投影前的特征，图像侧为 768 维）。

    本函数同时兼容新旧两种返回，避免换 transformers 版本就把流程打断。
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                return v
    raise TypeError(f"无法从 {type(out).__name__} 取出嵌入向量；"
                    f"可用属性：{getattr(out, '__dict__', {}).keys()}")


def _image_features(kind: str, model, pixel_values):
    if kind == "transformers":
        # ★ IMG_SIZE≠224 时必须插值位置编码（7×7 → 9×9 / 11×11），否则报 shape 错。
        return take_embeds(model.get_image_features(
            pixel_values=pixel_values, interpolate_pos_encoding=(C.IMG_SIZE != 224)))
    return model.encode_image(pixel_values)


# ------------------------------------------------- CLIP 图像预处理（可并行）
# CLIP 官方预处理参数（与 CLIPProcessor 默认值一致）：
#   resize 短边到 224（双三次）→ 中心裁剪 224 → /255 → 按下面均值方差归一化
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _FlipLR:
    """水平镜像（可 pickle）。

    ⚠️ 必须写成**模块级类**，不能用 ``T.Lambda(lambda im: ...)``。
    Windows 上 DataLoader 用 spawn，dataset（连同它的 transform）要被 pickle
    后送进 worker；而 Lambda 是 ``build_transform`` 的局部对象，pickle 会报
        AttributeError: Can't get local object 'build_transform.<locals>.<lambda>'
    （实测踩到，且只有当 ``num_workers>0`` 时才暴露。）
    """

    def __call__(self, im):
        from PIL import Image as _I
        return im.transpose(_I.Transpose.FLIP_LEFT_RIGHT)


def build_transform(flip: bool = False):
    """构造自写的 CLIP 预处理管线（``PREPROC_MODE="fast"``）。

    为什么自己实现：官方 CLIPProcessor 即使在 worker 里也要走它自己的
    PIL 路径，实测单进程约 123 img/s；本管线 201 img/s，快约 1.6 倍。

    与官方处理器的数值差异：换算回像素空间最大约 2/255 灰阶（舍入级），
    不是算法差异 —— 两边都是「短边 224 双三次 → 中心裁剪 → /255 → 归一化」，
    参数逐项相同（resample=3、shortest_edge=224、同一组 mean/std），
    只是双三次插值的实现细节不同。

    ``flip=True`` 用于 TTA：在**预处理之前**水平镜像。因为中心裁剪是水平
    对称的，"先翻转再裁剪" 与 "先裁剪再翻转" 等价，所以放在最前面最省事。
    """
    from torchvision import transforms as T

    ops = []
    if flip:
        ops.append(_FlipLR())
    ops += [
        T.Resize(C.IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(C.IMG_SIZE),
        T.ToTensor(),                       # PIL -> float32 [0,1]
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ]
    return T.Compose(ops)


class _OfficialPreproc:
    """``PREPROC_MODE="official"``：用官方 CLIPImageProcessor 做预处理。

    刻意做成**惰性创建**，有两个原因：
      1. Windows 上 DataLoader 用 spawn，dataset 必须可 pickle —— 存路径字符串
         能安全跨进程，存处理器对象则可能出问题；
      2. 每个 worker 进程各持一份，互不干扰。
    """

    def __init__(self, size: int = C.IMG_SIZE):
        self.size = size
        self._proc = None

    def _get(self):
        if self._proc is None:
            from transformers import CLIPImageProcessor
            self._proc = CLIPImageProcessor.from_pretrained(
                resolve_model_ref(), local_files_only=Path(resolve_model_ref()).exists())
        return self._proc

    def __call__(self, im):
        return self._get()(images=[im], return_tensors="pt")["pixel_values"][0]


# --------------------------------------------------------------- Dataset
class ImageOnlyDataset(Dataset):
    """产出「预处理好的张量 + 索引」。

    预处理放在 ``__getitem__`` 里 → 跑在 DataLoader 工作进程中并行完成。
    读取失败时用灰度占位图兜底（赛题说明部分图片被截断，Pillow 一般可读，
    但个别文件仍可能异常，不能让整批中断）。
    """

    def __init__(self, paths: list[str], transform=None, mode: str | None = None,
                 out_dtype=None):
        self.paths = paths
        self.mode = mode or C.PREPROC_MODE
        if transform is not None:
            self.tf = transform
            self.mode = "custom"
        elif self.mode == "official":
            self.tf = _OfficialPreproc()
        else:
            self.tf = build_transform()
        # ★ 在 worker 里就把张量降到 fp16。
        # 动机：批大小 96 时，float32 张量一批 96×3×224×224×4 ≈ 58MB，
        # 要经共享内存从 worker 传到主进程；转 fp16 直接减半。
        # 归一化已在 float32 下完成再降精度，数值上没有损失（取值范围约 ±2.5，
        # fp16 在该区间有约 10 位有效位）。
        self.out_dtype = out_dtype
        self.n_failed = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = load_image(self.paths[i])
        if im is None:
            self.n_failed += 1
            from PIL import Image
            im = Image.new("RGB", (C.IMG_SIZE, C.IMG_SIZE), (127, 127, 127))
        t = self.tf(im)
        if self.out_dtype is not None and t.dtype != self.out_dtype:
            t = t.to(self.out_dtype)
        return t, i


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, idx


# ------------------------------------------------------------ 特征提取主流程
@torch.no_grad()
def extract_features(paths: list[str], device: str = "cuda",
                     batch: int = C.EXTRACT_BATCH,
                     tag: str = "feat",
                     mode: str | None = None,
                     workers: int | None = None,
                     profile: bool = False,
                     transform=None) -> np.ndarray:
    """返回 (N, 512) float32 特征（已 L2 归一化）。

    ``mode``      预处理实现："fast"（默认）| "official"（官方处理器逐位口径）
    ``workers``   覆盖 DataLoader 进程数（默认取 C.EXTRACT_WORKERS）
    ``profile``   打印「等数据 / GPU 前向 / 后处理」的耗时拆分，便于定位瓶颈
    ``transform`` 自定义预处理（TTA 用，如 ``build_transform(flip=True)``）。
                  ⚠️ 它必须同时传给**降级分支**里重建的 dataset —— 否则 pin
                  失败时那一次会静默提成"未翻转"的特征，TTA 就白做了。
    """
    kind, model, proc = load_clip(device)
    use_amp = (device == "cuda")
    n_workers = C.EXTRACT_WORKERS if workers is None else workers

    # ★ 模型也要转 fp16，不能只转输入。
    # 实测：fp32 模型 + fp16 输入 = 561 img/s；fp16 模型 + fp16 输入 = 1486 img/s
    # （2.7 倍）。只转输入会让每层做隐式类型转换。
    if use_amp:
        model.half()

    ds = ImageOnlyDataset(paths, transform=transform, mode=mode,
                          out_dtype=torch.float16 if use_amp else torch.float32)

    # ★ pin_memory 只在**单进程**（num_workers==0）时开启。
    #
    # 原因（实测踩坑）：num_workers>0 且 pin_memory=True 时，torch 会让 worker
    # 通过共享内存把张量送给主进程里一条专门的「pin memory 线程」，
    # 跨进程同步依赖命名事件。在本机（Windows）上这条路径不可靠，worker 一多就崩：
    #     RuntimeError: Couldn't open shared event: <torch_..._event>, error code: <2>
    #     RuntimeError: Pin memory thread exited unexpectedly
    # 单进程时 pin 是在本进程内完成的，不涉及共享内存，稳定可用。
    #
    # 代价很小：96 张 fp16 批约 29MB，可分页拷贝与固定内存拷贝的差别在
    # 整个流程里占不到 5%（真正的耗时在 CPU 预处理与 GPU 前向）。
    use_pin = use_amp and n_workers == 0

    def _make_loader(pin: bool):
        # 注意：这里**故意不用** persistent_workers。
        # 本函数只遍历一次数据集，保活 worker 没有收益；而 worker 持有的
        # pinned 内存是 cudaHostRegister 注册的，进程内后续再建 DataLoader
        # 会对同一块内存重复注册，触发
        #   RuntimeError: CUDA error: resource already mapped
        # （实测于「同一进程里先提一次特征、再提一次」的场景）。
        return DataLoader(ds, batch_size=batch, shuffle=False,
                          num_workers=n_workers, collate_fn=_collate,
                          pin_memory=pin,
                          prefetch_factor=4 if n_workers > 0 else None)

    out = None
    t0 = time.time()
    t_wait = t_prep = t_gpu = t_post = 0.0
    dl = _make_loader(use_pin)
    it = iter(dl)
    step = 0
    while True:
        _t0 = time.time()
        try:
            x, idx = next(it)
        except StopIteration:
            break
        except RuntimeError as e:
            msg = str(e)
            # 两类都可恢复的「环境性」故障，统一降级为不带 pin 的加载器：
            #   * resource already mapped  —— pinned 内存重复注册（CUDA）
            #   * shared event / Pin memory thread —— Windows 跨进程 pin 失败
            if not any(k in msg for k in
                       ("resource already mapped", "shared event",
                        "Pin memory thread")):
                raise
            print(f"[{tag}] ⚠️ pinned 传输不可用（{msg.splitlines()[0][:70]}），"
                  f"降级为非 pinned", flush=True)
            del it, dl
            ds = ImageOnlyDataset(paths, transform=transform, mode=mode,  # 重建（n_failed 归零重计）
                                  out_dtype=torch.float16 if use_amp else torch.float32)
            dl = _make_loader(False)
            it = iter(dl)
            out = None
            t0 = time.time()
            t_wait = t_prep = t_gpu = t_post = 0.0
            step = 0
            continue

        step += 1
        # non_blocking 只有在源内存是 pinned 时才真正异步；否则 torch 会退化为
        # 同步拷贝。这里跟随 use_pin，避免给出"看似异步"的错误印象。
        _t = time.time() if profile else 0.0
        x = x.to(device, non_blocking=use_pin)
        if use_amp:
            x = x.half()
        if profile:
            t_wait += _t - _t0
            t_prep += time.time() - _t
            _t = time.time()
        f = _image_features(kind, model, x)
        if isinstance(f, torch.Tensor) is False:
            f = take_embeds(f)
        if profile:
            torch.cuda.synchronize() if use_amp else None
            t_gpu += time.time() - _t
            _t = time.time()
        f = f.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(paths), f.shape[1]), dtype=np.float32)
        out[idx.numpy()] = f
        if profile:
            t_post += time.time() - _t
        if step % 50 == 0:
            el = time.time() - t0
            done = min(step * batch, len(paths))
            print(f"[{tag}] {done}/{len(paths)}  {done/max(el,1e-6):.1f} img/s  "
                  f"已用 {el/60:.1f} min", flush=True)
    del it, dl

    if ds.n_failed:
        print(f"[{tag}] ⚠️ {ds.n_failed} 张图读取失败，已用灰度占位")
    total = time.time() - t0
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    print(f"[{tag}] 完成：{out.shape} 用时 {total/60:.1f} min，"
          f"吞吐 {len(paths)/total:.1f} img/s")
    if profile:
        print(f"[{tag}] ── 耗时拆分（workers={n_workers}, pin={use_pin}）──")
        for _n, _v in (("等数据（worker 解码+预处理经共享内存送达）", t_wait),
                       ("H2D 拷贝", t_prep),
                       ("GPU 前向（含同步）", t_gpu),
                       ("取回 CPU + 归位", t_post)):
            print(f"[{tag}]   {_n:<44} {_v:7.2f}s  {_v/total*100:5.1f}%")
        print(f"[{tag}]   {'其它（模型加载、worker 启停等）':<44} "
              f"{total-t_wait-t_prep-t_gpu-t_post:7.2f}s")
    del model
    if use_amp:
        torch.cuda.empty_cache()
    return out.astype(np.float32)


def cache_path(name: str) -> Path:
    return C.CACHE_DIR / f"{name}.npy"


def get_or_build_features(paths: list[str], name: str,
                          device: str = "cuda",
                          transform=None) -> np.ndarray:
    """带磁盘缓存的提特征：同名已存在则直接复用。

    ``transform`` 仅供 TTA 之类需要换预处理的场景使用 —— 换了它就必须换
    ``name``（缓存只认名字），否则会静默复用另一套预处理的特征。
    """
    p = cache_path(name)
    if p.exists():
        arr = np.load(p)
        if arr.shape[0] == len(paths):
            print(f"[cache] 复用 {p.name} {arr.shape}")
            return arr
        print(f"[cache] {p.name} 行数不符（{arr.shape[0]} != {len(paths)}），重建")
    arr = extract_features(paths, device=device, tag=name, transform=transform)
    np.save(p, arr)
    print(f"[cache] 写入 {p}")
    return arr
