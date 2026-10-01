#!/usr/bin/env bash
# 用 trtexec 把 SAM3 的 ONNX 构建为**指定精度**的 TensorRT engine
#
# 用法:
#   bash build_engine_sam3.sh <onnx> [precision] [workspace_MB]
#
# precision（默认 tf32）:
#   tf32   Tensor Core TF32（FP32 权重近似）—— 与 PyTorch 默认一致，精度几乎无损【推荐】
#   fp32   纯 IEEE FP32（--noTF32）—— 最精确，但不用 Tensor Core
#   fp16   半精度混合（--fp16）—— 更快，但 SAM3 的 ViT 对数值敏感，可能明显掉精度
#   bf16   bfloat16 混合（--bf16）
#
# 例:
#   bash sh/build_engine_sam3.sh models/sam3_vision_encoder.onnx tf32
#   bash sh/build_engine_sam3.sh models/sam3_text_encoder.onnx   fp16
#
# 说明:
#   - TensorRT 10.3 **没有 --tf32 选项**，TF32 是默认行为；--noTF32 才回到纯 FP32。
#     故：tf32=不加精度 flag，fp32=加 --noTF32，fp16=加 --fp16，bf16=加 --bf16。
#   - 输出文件名按精度区分，便于多种精度共存与 A/B 对比：
#         models/<name>.<precision>.engine
#   - 推理时用环境变量选择精度：
#         SAM3_PREC=tf32 ./sh/run_trt.sh --image ... --text ...
set -euo pipefail
cd "$(dirname "$0")/.."

ONNX="${1:-models/sam3_vision_encoder.onnx}"
PREC="${2:-tf32}"
WS_MB="${3:-4096}"

[ -f "$ONNX" ] || { echo "[错误] 找不到 $ONNX"; exit 1; }

# ---- 精度 -> trtexec 参数 ----
case "$PREC" in
  tf32) PREC_FLAGS=""          ; PREC_DESC="TF32（Tensor Core，默认）" ;;
  fp32) PREC_FLAGS="--noTF32"  ; PREC_DESC="FP32（IEEE，关闭 TF32）" ;;
  fp16) PREC_FLAGS="--fp16"    ; PREC_DESC="FP16（混合精度）" ;;
  bf16) PREC_FLAGS="--bf16"    ; PREC_DESC="BF16（混合精度）" ;;
  *) echo "[错误] 不支持的精度 '$PREC'（可选：tf32 / fp32 / fp16 / bf16）"; exit 1 ;;
esac

TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "$TRTEXEC" ] || TRTEXEC="$(command -v trtexec)"
[ -n "${TRTEXEC:-}" ] || { echo "[错误] 找不到 trtexec"; exit 1; }

ENGINE="${ONNX%.onnx}.${PREC}.engine"
LOG="logs/engine_$(basename "${ONNX%.onnx}")_${PREC}.log"
mkdir -p logs

echo "======================================================================"
echo " 构建 TensorRT engine"
echo "   ONNX      : $ONNX"
echo "   precision : $PREC_DESC ${PREC_FLAGS:+（flag: $PREC_FLAGS）}"
echo "   ENGINE    : $ENGINE"
echo "   workspace : ${WS_MB} MB"
echo "   trtexec   : $TRTEXEC"
echo "   日志      : $LOG"
echo "======================================================================"

START=$(date +%s)

# shellcheck disable=SC2086  # 需要把 $PREC_FLAGS 按空格拆成独立参数
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  $PREC_FLAGS \
  --memPoolSize=workspace:${WS_MB} \
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
echo " 构建完成（${PREC_DESC}），耗时 ${DUR}s ($(echo "scale=1; $DUR/60" | bc) 分钟)"
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
echo
echo "已生成: $ENGINE"
echo "推理时用该精度（本脚本会把精度写进文件名，run_trt.sh 靠 SAM3_PREC 选择）："
echo "   SAM3_PREC=$PREC ./sh/run_trt.sh --image inputimage/6.jpg --text \"shoe\" --out outputimage/6_trt.png"