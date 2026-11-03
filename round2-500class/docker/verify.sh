#!/usr/bin/env bash
# ============================================================================
# 环境自检（可选，约 30 秒）
# ============================================================================
# 用途：在跑完整复现之前，先确认镜像本身是好的（包版本对不对、CUDA 通不通）。
#      完整复现请直接用同目录的 reproduce_submit.sh。
#
# 用法：
#   1) 只验镜像环境（不需要任何挂载）：
#        bash verify.sh
#   2) 连挂载一起验（顺带确认代码/模型/数据能被看到）：
#        CODE_DIR=/path/to/代码与数据 MODELS_DIR=/path/to/模型文件 DATA_DIR=/path/to/data bash verify.sh
#
# Windows Git Bash 用户请在命令前加 MSYS_NO_PATHCONV=1
# ============================================================================
set -euo pipefail

IMG="${AIC_IMAGE:-aic-ntl-semi:760137}"
CODE_DIR="${CODE_DIR:-}"
MODELS_DIR="${MODELS_DIR:-}"
DATA_DIR="${DATA_DIR:-}"
WORKERS_ENV="${AIC_WORKERS:-8}"

MOUNTS=()
[ -n "$CODE_DIR" ]   && MOUNTS+=(-v "$CODE_DIR:/workspace/aic/code:ro")
[ -n "$MODELS_DIR" ] && MOUNTS+=(-v "$MODELS_DIR:/workspace/aic/models:ro")
[ -n "$DATA_DIR" ]   && MOUNTS+=(-v "$DATA_DIR:/workspace/aic/data:ro")

echo "镜像: $IMG"
echo

docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" \
  -e AIC_ROOT=/workspace/aic \
  "$IMG" \
  python -c "
import sys, torch, torchvision, transformers, numpy, pandas, PIL
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__, '| CUDA build', torch.version.cuda)
print('torchvision ', torchvision.__version__)
print('transformers', transformers.__version__)
print('numpy       ', numpy.__version__, '| pandas', pandas.__version__, '| pillow', PIL.__version__)
print()
print('CUDA 可用   ', torch.cuda.is_available(), '（False 时请确认 docker run 带了 --gpus all）')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU         ', p.name, round(p.total_memory / 2**30, 1), 'GB')
"

if [ -n "$CODE_DIR" ]; then
  echo
  echo "--- 代码可导入性 ---"
  docker run --rm --gpus all --ipc=host "${MOUNTS[@]}" \
    -e AIC_ROOT=/workspace/aic \
    -e PYTHONPATH=/workspace/aic/code/src:/workspace/aic/code/scripts \
    -e AIC_CLIP_DIR=/workspace/aic/models/clip-vit-base-patch32 \
    -e AIC_IMG_SIZE=560 \
    "$IMG" \
    python -c "
import config, tta_lora_eval, predict_tta, calib_sinkhorn_apply, check_submission
print('模块导入成功 | STAGE_NAME =', config.STAGE_NAME, '| 视图数 =', len(tta_lora_eval.ALL_VIEWS))
"
fi

if [ -n "$DATA_DIR" ]; then
  echo
  echo "--- 测试集张数 ---"
  echo "  $(ls "$DATA_DIR/test" 2>/dev/null | wc -l) 张（应为 49857）"
fi

echo
echo "自检结束。完整复现请运行：bash reproduce_submit.sh"
echo "（DataLoader 进程数默认 $WORKERS_ENV，可用 -e AIC_WORKERS=N 调整）"
