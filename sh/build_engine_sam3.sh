#!/usr/bin/env bash
# 用 trtexec 把 SAM3 的 ONNX 构建为 **FP32** TensorRT engine
#
# 用法:
#   bash build_engine_sam3.sh <onnx> [precision] [workspace_MB]
# 例:
#   bash build_engine_sam3.sh models/sam3_vision_encoder.onnx fp32 4096
#
# 说明:
#   - 精度固定 FP32：不加 --fp16 / --int8，且显式填入 --precisionConstraints=obey
#     （SAM3 的 ViT 对数值较敏感，FP16 会明显掉精度）
#   - 输出的 engine 与 ONNX 同目录，同名 .engine
set -euo pipefail
cd "$(dirname "$0")/.."

ONNX="${1:-models/sam3_vision_encoder.onnx}"
PREC="${2:-fp32}"
WS_MB="${3:-4096}"

[ -f "$ONNX" ] || { echo "[错误] 找不到 $ONNX"; exit 1; }
[ "$PREC" = "fp32" ] || { echo "[错误] 本脚本只构建 FP32（当前传入 $PREC）"; exit 1; }

TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "$TRTEXEC" ] || TRTEXEC="$(command -v trtexec)"
[ -n "${TRTEXEC:-}" ] || { echo "[错误] 找不到 trtexec"; exit 1; }

ENGINE="${ONNX%.onnx}.engine"
LOG="logs/engine_$(basename "${ONNX%.onnx}")_${PREC}.log"

echo "======================================================================"
echo " 构建 TensorRT engine (FP32)"
echo "   ONNX      : $ONNX"
echo "   ENGINE    : $ENGINE"
echo "   workspace : ${WS_MB} MB"
echo "   trtexec   : $TRTEXEC"
echo "   日志      : $LOG"
echo "======================================================================"

START=$(date +%s)

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --memPoolSize=workspace:${WS_MB} \
  --precisionConstraints=obey \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --verbose \
  > "$LOG" 2>&1 || {
    echo "[失败] trtexec 返回非 0，末尾日志："
    tail -40 "$LOG"
    exit 1
  }

END=$(date +%s)
DUR=$((END - START))

echo
echo "----------------------------------------------------------------------"
echo " 构建完成，耗时 ${DUR}s ($(echo "scale=1; $DUR/60" | bc) 分钟)"
ls -lh "$ENGINE" 2>/dev/null || true
echo "----------------------------------------------------------------------"
grep -iE "^\[[0-9/]+.*\] (Engine built|Total time|Throughput)" "$LOG" | tail -5 || true
echo
echo "关键指标（从日志提取）:"
grep -iE "Engine built in|Serialized engine|Total Host Walltime|Total GPU Compute Time" "$LOG" | tail -6 || true
echo
echo "耗时: ${DUR}s" 
echo "$DUR" > "${ENGINE}.build_seconds"
echo "已写入 ${ENGINE}.build_seconds"