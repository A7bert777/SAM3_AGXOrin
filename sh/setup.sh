#!/usr/bin/env bash
# SAM3_AGXOrin 一键环境初始化（Jetson AGX Orin / JetPack, aarch64, Python 3.10）
#
# 步骤：
#   1. 创建 venv310（若已存在且版本正确则复用）
#   2. 检查/安装 NVIDIA 定制 PyTorch（Jetson 专用 wheel）
#   3. 安装 SAM3 依赖
#   4. 安装 sam3 本体（--no-deps，避免 pip 改动 torch）
#   5. 打「无 Triton」兼容补丁（Jetson 无 triton，图像分割不需要）
#   6. 下载权重与 BPE 词表
#   7. 自检 import sam3
#
# 用法：./sh/setup.sh
#
# ── 为什么必须用 Python 3.10 ─────────────────────────────────────────────
# NVIDIA 为 Jetson 发布的 PyTorch wheel 只有 cp310 一个 ABI 版本。
# 若用系统默认的 python3（可能是 3.12）建 venv，pip 会拒绝安装并报：
#   ERROR: torch-...-cp310-cp310-linux_aarch64.whl is not a supported wheel on this platform.
# 所以这里显式使用 python3.10 建环境，并在发现已有 venv 版本不对时自动重建。
#
# ── 为什么可能要 get-pip.py ──────────────────────────────────────────────
# 部分 Jetson/L4T 环境里的 python3.10 缺 ensurepip（apt 源没有 python3.10-venv），
# 此时 `python3.10 -m venv` 会失败。脚本会退回 `--without-pip` 建环境，
# 再用 get-pip.py 补上 pip。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.10/site-packages"
LOG="$ROOT/logs"

# Jetson 定制 PyTorch 源（JetPack 6.x / CUDA 12.6，aarch64 + cp310）
# venv310 目录名约定：3 = py3.10，10 = 3.10
#
# 注意：这里必须用「单一 --index-url」，不能只把该源追加为 --extra-index-url。
# 因为 PyPI 上也有同名的 torch-2.8.0-cp310-cp310-manylinux_2_28_aarch64.whl（CPU 版），
# 两源同时可见时 pip 会装上 CPU 版，导致 torch.cuda.is_available() == False。
TORCH_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/"
TORCH_VER="2.8.0"
TV_VER="0.23.0"

mkdir -p "$LOG" "$ROOT/models" "$ROOT/assets" \
         "$ROOT/inputimage" "$ROOT/outputimage" "$ROOT/outputs"

echo "=============================================="
echo " SAM3_AGXOrin 环境初始化"
echo " 项目根目录: $ROOT"
echo "=============================================="

# ---------- 0. 找 python3.10 ----------
PY310=""
for c in python3.10 /usr/bin/python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PY310="$(command -v "$c")"; break; fi
done
if [ -z "$PY310" ]; then
  echo "[错误] 未找到 python3.10。"
  echo "       Jetson 版 PyTorch wheel 只有 cp310 版本，必须使用 Python 3.10。"
  echo "       请先安装 Python 3.10（例如：sudo apt install python3.10）后重试。"
  exit 1
fi

venv_ver() {  # 打印现有 venv 的 "X.Y"，无则空
  [ -x "$PY" ] && "$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true
}

bootstrap_pip() {  # 给 $VENV 补上 pip（用于 python3.10 缺 ensurepip 的情况）
  echo "      为 venv 引导安装 pip（get-pip.py）..."
  local GETPIP="$ROOT/get-pip.py"
  curl -fsSL --retry 3 --retry-delay 2 --max-time 180 -o "$GETPIP" \
    https://bootstrap.pypa.io/get-pip.py \
  || curl -fsSL --retry 3 --retry-delay 2 --max-time 180 -o "$GETPIP" \
       https://mirrors.aliyun.com/pypi/get-pip.py \
  || { echo "[错误] 无法下载 get-pip.py，请检查网络。"; exit 1; }
  "$PY" "$GETPIP" --no-cache-dir >"$LOG/getpip.log" 2>&1 \
    || { echo "[错误] pip 引导失败："; tail -20 "$LOG/getpip.log"; exit 1; }
  rm -f "$GETPIP"
}

# ---------- 1. venv ----------
CUR="$(venv_ver)"
if [ "$CUR" = "3.10" ]; then
  echo "[1/7] 复用已有 venv: $VENV ($("$PY" -V 2>&1))"
  # 复用前确认 pip 真的可用：venv 可能是上次中断留下的半成品
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    bootstrap_pip
  fi
else
  if [ -n "$CUR" ]; then
    echo "[1/7] 现有 venv 是 Python $CUR（wheel 需要 cp310），删除重建 ..."
    rm -rf "$VENV"
  else
    echo "[1/7] 创建 venv310 ..."
  fi

  if ! "$PY310" -m venv "$VENV" >/dev/null 2>&1; then
    echo "      python3.10 缺少 ensurepip，改用 --without-pip 建环境"
    rm -rf "$VENV"
    "$PY310" -m venv --without-pip "$VENV"
    bootstrap_pip
  fi
  echo "      已创建: $("$PY" -V 2>&1)"
fi

export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export USE_PERFLIB=0

# ---------- 2. torch ----------
if "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "[2/7] torch 已就绪: $("$PY" -c 'import torch; print(torch.__version__)')"
else
  echo "[2/7] 安装 NVIDIA 定制 PyTorch（Jetson aarch64, JetPack 6.x）..."
  # 第 1 步：torch/torchvision 只从 Jetson 源取，--no-deps 避免它顺手把
  #         CPU 版依赖拉进来；版本号固定，确保拿到的是 CUDA 版。
  if ! "$PY" -m pip install --no-cache-dir --no-deps \
        --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VER" "torchvision==$TV_VER" \
        >"$LOG/torch.log" 2>&1; then
    echo "[错误] PyTorch 安装失败，日志尾部："
    tail -20 "$LOG/torch.log"
    echo
    echo "[提示] 可参考 NVIDIA Jetson PyTorch 论坛贴手动安装后重跑本脚本。"
    exit 1
  fi
  # 第 2 步：torch/torchvision 的运行期依赖走 PyPI 正常安装
  "$PY" -m pip install --no-cache-dir \
        "filelock" "typing-extensions>=4.10.0" "sympy>=1.13.3" \
        "networkx" "jinja2" "fsspec" "pillow" \
        >>"$LOG/torch.log" 2>&1 \
    || { echo "[错误] PyTorch 依赖安装失败，日志尾部："; tail -20 "$LOG/torch.log"; exit 1; }
  # 第 3 步：确认装到的确实是能在 Orin 上用 GPU 的版本
  "$PY" - <<'PYCHK' >"$LOG/torchchk.log" 2>&1 || {
import sys, torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
PYCHK
    echo "[错误] torch 装上了但 CUDA 不可用（很可能拿到了 CPU 版）："
    cat "$LOG/torchchk.log"
    echo
    echo "[提示] 请确认 Jetson 源可访问：$TORCH_INDEX"
    exit 1
  }
  echo "      已安装: $("$PY" -c 'import torch; print(torch.__version__)')（CUDA 可用）"
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

# ---------- 3b. ONNX 导出 / 校验依赖 ----------
# onnx 用于把 sam3.pt 导出为 ONNX；onnxruntime 用于导出后的数值校验。
# ⚠️ onnx 依赖 ml_dtypes，而新版 ml_dtypes 会把 numpy 强升到 2.x；
#    Jetson 版 torch 2.8 是按 numpy 1.x 编译的，numpy>=2 会导致 import torch 失败。
#    因此这里安装 onnx 后，显式把 numpy 拉回 <2，并把 ml_dtypes 固定到兼容 numpy1.x 的 0.5.4。
echo "[3/7] 安装 ONNX 导出/校验依赖（onnx / onnxruntime）..."
"$PY" -m pip install --no-cache-dir "onnx" "onnxruntime" \
  > "$LOG/onnx.log" 2>&1 || { tail -20 "$LOG/onnx.log"; exit 1; }
# 回退 numpy 到 <2，并固定 ml_dtypes（--no-deps 防止 pip 又把 numpy 升上去）
"$PY" -m pip install --no-cache-dir "numpy<2" >> "$LOG/onnx.log" 2>&1 || true
"$PY" -m pip install --no-cache-dir --no-deps "ml_dtypes==0.5.4" >> "$LOG/onnx.log" 2>&1 || true
# 校验：numpy<2 且 onnx 可用
"$PY" - <<'ONNXCHK' || { echo "[错误] onnx/numpy 环境异常，日志尾部："; tail -20 "$LOG/onnx.log"; exit 1; }
import numpy, onnx
assert numpy.__version__.split(".")[0] == "1", f"numpy 必须为 1.x，当前 {numpy.__version__}"
print(f"      numpy {numpy.__version__} | onnx {onnx.__version__} | onnxruntime 就绪")
ONNXCHK
echo "      已安装（日志：logs/onnx.log）"

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
echo "   ./sh/run.sh --image inputimage/1.jpg --text dog --out outputimage/1_dog.png"
echo "   ./sh/run.sh --image inputimage/1.jpg --text dog --bench"
echo "=============================================="