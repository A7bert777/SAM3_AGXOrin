#!/usr/bin/env bash
# SAM3_AGXOrin 一键环境初始化（Jetson AGX Orin / JetPack, aarch64, Python 3.10）
#
# 步骤：
#   1. 创建 venv310（若已存在则复用）
#   2. 检查/安装 NVIDIA 定制 PyTorch（Jetson 专用 wheel）
#   3. 安装 SAM3 依赖
#   4. 安装 sam3 本体（--no-deps，避免 pip 改动 torch）
#   5. 打「无 Triton」兼容补丁（Jetson 无 triton，图像分割不需要）
#   6. 下载权重与 BPE 词表
#   7. 自检 import sam3
#
# 用法：./setup.sh
# 说明：若已从 SAM_AGXOrin 复制可用的 venv310（内含 torch 2.8.0），
#       本脚本会直接复用，不会重新下载 PyTorch。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.10/site-packages"
LOG="$ROOT/logs"

mkdir -p "$LOG" "$ROOT/models" "$ROOT/assets" \
         "$ROOT/inputimage" "$ROOT/outputimage" "$ROOT/outputs"

echo "=============================================="
echo " SAM3_AGXOrin 环境初始化"
echo " 项目根目录: $ROOT"
echo "=============================================="

# ---------- 1. venv ----------
if [ -x "$PY" ]; then
  echo "[1/7] 复用已有 venv: $VENV"
else
  echo "[1/7] 创建 venv310 ..."
  python3 -m venv "$VENV"
fi

export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export USE_PERFLIB=0

# ---------- 2. torch ----------
if "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "[2/7] torch 已就绪: $("$PY" -c 'import torch; print(torch.__version__)')"
else
  echo "[2/7] 安装 NVIDIA 定制 PyTorch（Jetson aarch64, JetPack 6.x）..."
  "$PY" -m pip install --no-cache-dir \
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl" \
    || { echo "[提示] 自动安装失败，请参考 NVIDIA Jetson PyTorch 论坛贴手动安装，"
         echo "       或复制 SAM_AGXOrin/venv310 到本目录。"; exit 1; }
fi

# ---------- 3. 依赖 ----------
echo "[3/7] 安装 SAM3 运行依赖 ..."
"$PY" -m pip install --no-cache-dir \
  "numpy<2" "timm>=1.0.17" "ftfy>=6.1.1" "regex" \
  "iopath>=0.1.10,<0.2.0" "huggingface-hub>=0.30.0,<2.0" "einops>=0.8.0,<0.9.0" \
  "psutil" "hydra-core" "omegaconf" \
  "opencv-python-headless" "pillow" "pycocotools" "tqdm" \
  > "$LOG/deps.log" 2>&1 || { tail -20 "$LOG/deps.log"; exit 1; }
echo "      依赖安装完成（日志：logs/deps.log）"

# ---------- 4. sam3 本体 ----------
echo "[4/7] 安装 sam3 本体（--no-deps，防止 pip 改动 torch）..."
"$PY" -m pip install --no-cache-dir --no-deps "sam3==0.1.4" 2>&1 | tail -3

# ---------- 5. triton 补丁 ----------
echo "[5/7] 应用 SAM3「无 Triton」兼容补丁 ..."
"$PY" "$ROOT/py/patch_sam3_triton.py"

# ---------- 6. 资源 ----------
echo "[6/7] 下载权重与词表 ..."
bash "$ROOT/sh/download.sh"

# ---------- 7. 自检 ----------
echo "[7/7] 自检 import sam3 ..."
"$PY" "$ROOT/py/selfcheck.py"

echo
echo "=============================================="
echo " 完成！示例："
echo "   ./run.sh --image inputimage/001.jpg --text dog --out outputimage/001_dog.png"
echo "   ./run.sh --image inputimage/001.jpg --text dog --bench"
echo "=============================================="