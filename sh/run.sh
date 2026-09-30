#!/usr/bin/env bash
# SAM3_AGXOrin 运行入口：设置环境后调用客户端 sam3_client.py
#
# 用法：
#   ./sh/run.sh --image inputimage/a.jpg --text "dog"  --out outputimage/a_dog.png
#   ./sh/run.sh --image inputimage/a.jpg --point 114 285 --out outputimage/a_point.png
#
# 行为：
#   若已执行 ./sh/serve.sh start（常驻服务在线）-> 请求转发给服务，约 1~2s 返回
#   否则                                      -> 本地加载模型单次运行（约 16s）
#   两种方式输出完全一致。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./setup.sh 初始化环境"
  exit 1
fi

# cuSPARSELt 等 NVIDIA 运行库（Jetson 上 torch 依赖）
export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
# 关闭 SAM3 的 perflib（依赖 triton，Jetson 无 triton）
export USE_PERFLIB="${USE_PERFLIB:-0}"
# Jetson 上避免碎片化；如显存紧张可改为 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

exec "$VENV/bin/python" "$ROOT/py/sam3_client.py" "$@"
