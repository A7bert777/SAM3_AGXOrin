#!/usr/bin/env bash
# 下载 SAM3 所需的本地资源（支持断点续传，重复执行即可）：
#   1) models/sam3.pt                        权重（约 3.45 GB）
#   2) assets/bpe_simple_vocab_16e6.txt.gz   文本 tokenizer 词表
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/models" "$ROOT/assets"

CKPT="$ROOT/models/sam3.pt"
BPE="$ROOT/assets/bpe_simple_vocab_16e6.txt.gz"

CKPT_URLS=(
  "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
)
BPE_URLS=(
  "https://huggingface.co/facebook/sam3/resolve/main/bpe_simple_vocab_16e6.txt.gz"
  "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
)

fetch() {  # fetch <目标文件> <最小字节> <候选URL...>
  local out="$1" min="$2"; shift 2
  if [[ -f "$out" ]] && (( $(stat -c%s "$out") >= min )); then
    echo "[跳过] 已存在：$out ($(du -h "$out" | cut -f1))"
    return 0
  fi
  for url in "$@"; do
    echo "[下载] $url"
    # -#      : 强制显示实时进度条（断点续传时同样有效）。
    #           注意：curl 默认的进度表在「输出不是终端」时会被自动隐藏，
    #           所以这里必须用 -#，才能在任何情况下都看到进度。
    # --retry : 断线自动重试；-C - : 断点续传
    if curl -# -L --fail --retry 5 --retry-delay 3 -C - -o "$out" "$url"; then
      echo
      if (( $(stat -c%s "$out") >= min )); then
        echo "[完成] $out ($(du -h "$out" | cut -f1))"
        return 0
      fi
      echo "[警告] 文件偏小，尝试下一个源"
    fi
  done
  echo "[错误] 下载失败：$out"
  return 1
}

fetch "$CKPT" 3000000000 "${CKPT_URLS[@]}"
fetch "$BPE" 1000000 "${BPE_URLS[@]}"

echo
echo "资源就绪："
ls -la "$CKPT" "$BPE"