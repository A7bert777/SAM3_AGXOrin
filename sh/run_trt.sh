#!/usr/bin/env bash
# SAM3_AGXOrin 用 TensorRT engine 推理的运行入口
#
# 用法（与 run.sh 参数完全一致）：
#   ./run_trt.sh --image inputimage/1.jpg --text "dog" --out outputimage/1_dog.png
#   ./run_trt.sh --image inputimage/1.jpg --box 480 290 110 360 --out outputimage/1_box.png
#   ./run_trt.sh --image inputimage/1.jpg --text "dog" --bench
#
# 额外开关（用于 A/B 精度对比）：
#   --vision-backend pytorch    关闭 vision engine，回退 PyTorch
#   --text-backend   pytorch    关闭 text engine，回退 PyTorch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./setup.sh 初始化环境"
  exit 1
fi

VB="$ROOT/models/sam3_vision_encoder.engine"
TB="$ROOT/models/sam3_text_encoder.engine"
if [[ ! -f "$VB" || ! -f "$TB" ]]; then
  echo "[错误] 未找到 engine 文件："
  [[ -f "$VB" ]] || echo "       缺失 $VB"
  [[ -f "$TB" ]] || echo "       缺失 $TB"
  echo "       请先执行 ./build_engine_sam3.sh"
  exit 1
fi

# TensorRT 运行库（Python 绑定从 deb 手动装入 venv，不需要额外路径）
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
# 关闭 SAM3 的 perflib（依赖 triton，Jetson 无 triton）
export USE_PERFLIB="${USE_PERFLIB:-0}"
# Jetson 上避免碎片化
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

exec "$VENV/bin/python" "$ROOT/py/sam3_trt_infer.py" "$@"