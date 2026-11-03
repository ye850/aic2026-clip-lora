# ============================================================================
# AIC 2026 算法挑战赛道 · 半决赛 —— 复现环境镜像
# ============================================================================
# 依据组委会要求：
#   · 镜像**仅包含复现环境**，尽量精简
#   · 数据**不放**镜像
#   · 模型、代码**不放**镜像，单独提交；复现时从宿主机映射进容器
#
# 环境取值以「实际产出提交包（榜分 76.6137）的那台机」为准：
#   Python 3.12.3 / torch 2.6.0+cu124 / transformers 5.20.2 / numpy 2.1.3
#   ★ 不用 pytorch 官方镜像作基座，因为其 Python 版本为 3.11；本方案保证 3.12 精确一致。
#   ★ torch 的 cu124 构建由 PyPI 默认 wheel 提供（自带 nvidia-* CUDA 运行时库），
#     容器只需宿主机有 NVIDIA 驱动 + nvidia-container-toolkit，无需 CUDA toolkit。
#
# 构建：  docker build -t aic-ntl-semi:760137 .
# 运行：  docker run --gpus all --rm -it \
#           -v /宿主机/代码:/workspace/aic/code:ro \
#           -v /宿主机/模型:/workspace/aic/models:ro \
#           -v /宿主机/数据:/workspace/aic/data:ro \
#           -v /宿主机/输出:/workspace/out \
#           aic-ntl-semi:760137 bash
# ============================================================================
FROM python:3.12-slim

LABEL org.opencontainers.image.title="AIC2026 半决赛复现环境" \
      org.opencontainers.image.description="CLIP-LoRA 500类图像分类 · 仅环境（不含数据/模型/代码）" \
      org.opencontainers.image.version="760137"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# torch 运行时需要 libgomp1；pillow 的 wheel 自带图像编解码库
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.image.txt /tmp/requirements.image.txt
RUN pip install --no-cache-dir -r /tmp/requirements.image.txt \
 && rm -f /tmp/requirements.image.txt \
 && python -c "import torch, torchvision, transformers, numpy, pandas, PIL; \
print('torch', torch.__version__, '| torchvision', torchvision.__version__, \
'| transformers', transformers.__version__, '| numpy', numpy.__version__, \
'| pandas', pandas.__version__, '| pillow', PIL.__version__)"

# 以下三处刻意留空，运行时由宿主机挂载：
#   /workspace/aic/code     代码包（单独提交）
#   /workspace/aic/models   CLIP 官方权重 + fd560v2_best.pt（单独提交）
#   /workspace/aic/data     测试集（组委会提供）
WORKDIR /workspace/aic

CMD ["/bin/bash"]
