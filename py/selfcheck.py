#!/usr/bin/env python3
"""环境自检：确认 SAM3 在 Jetson 上可用。

检查项：
  1. 关键模块可导入（torch / sam3 / Sam3Processor）
  2. CUDA 可用、设备名称、显存
  3. torch 的 Triton 检测结果为 False（Jetson 正常现象）
  4. 本地权重与 BPE 词表存在且大小合理
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(desc: str, passed: bool, detail: str = "") -> None:
    global ok
    mark = "✅" if passed else "❌"
    print(f"  {mark} {desc}" + (f"  —— {detail}" if detail else ""))
    if not passed:
        ok = False


print("=" * 46)
print(" SAM3_AGXOrin 环境自检")
print("=" * 46)

# 1. 模块导入
try:
    import torch
    import sam3
    from sam3 import build_sam3_image_model  # noqa: F401
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    check("导入 sam3 / Sam3Processor", True, f"sam3={sam3.__version__}")
except Exception as e:  # noqa: BLE001
    check("导入 sam3 / Sam3Processor", False, repr(e))
    print("\n[提示] 若报 triton 相关错误，请执行： python patch_sam3_triton.py")
    sys.exit(1)

# 2. CUDA
try:
    cuda = torch.cuda.is_available()
    detail = torch.cuda.get_device_name(0) if cuda else "不可用"
    if cuda:
        free, total = torch.cuda.mem_get_info()
        detail += f" | 显存 {total / 2**30:.1f} GiB（空闲 {free / 2**30:.1f} GiB）"
    check("CUDA 可用", cuda, detail)
except Exception as e:  # noqa: BLE001
    check("CUDA 可用", False, repr(e))

# 3. Triton 检测（应为 False）
try:
    from torch.utils._triton import has_triton_package

    has_tri = has_triton_package()
    check("torch 的 Triton 检测（期望 False）", not has_tri,
          f"has_triton_package()={has_tri}；图像分割不依赖 Triton")
except Exception as e:  # noqa: BLE001
    check("torch 的 Triton 检测", False, repr(e))

# 4. 本地资源
ckpt = ROOT / "models" / "sam3.pt"
bpe = ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
check("权重 models/sam3.pt 存在且 >3GB", ckpt.is_file() and ckpt.stat().st_size > 3_000_000_000,
      f"{ckpt.stat().st_size / 2**30:.2f} GiB" if ckpt.is_file() else "缺失，请执行 ./download.sh")
check("词表 assets/bpe_...txt.gz 存在", bpe.is_file(),
      f"{bpe.stat().st_size} B" if bpe.is_file() else "缺失，请执行 ./download.sh")

print("=" * 46)
if ok:
    print(" 结论：环境就绪，可以开始推理。")
else:
    print(" 结论：存在未通过项，请按上述提示修复。")
sys.exit(0 if ok else 1)