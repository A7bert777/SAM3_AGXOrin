#!/usr/bin/env python3
"""SAM3 结构探针：打印子模块参数量与各阶段前向耗时，用于设计 ONNX 导出方案。

只做 introspection，不修改任何文件。建议重定向输出到日志：
    ./run.sh 之外单独调用：
    ./venv310/bin/python probe_sam3.py > logs/probe.log 2>&1
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")

import numpy as np
import torch
from PIL import Image


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def nparams(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


def main() -> int:
    t_all = time.time()
    bpe = ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    ckpt = ROOT / "models" / "sam3.pt"

    from sam3 import build_sam3_image_model

    print("=" * 76)
    print(" SAM3 结构探针")
    print("=" * 76)

    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=str(bpe),
        checkpoint_path=str(ckpt),
        load_from_HF=False,
        enable_segmentation=True,
        device="cuda",
    )
    model.eval()
    print(f"[加载] build_sam3_image_model: {time.time() - t0:.2f}s")

    total = nparams(model)
    print("\n--- 顶层子模块参数量 ---")
    for name, child in model.named_children():
        n = nparams(child)
        print(f"  {name:<32}{n:>14,d}  {100.0*n/total:5.1f}%  {human(n*4)}")
    print(f"  {'TOTAL':<32}{total:>14,d}  100.0%  {human(total*4)}")

    bb = model.backbone
    print("\n--- backbone (SAM3VLBackbone) ---")
    for name, child in bb.named_children():
        print(f"  {name:<32}{nparams(child):>14,d}  {human(nparams(child)*4)}")

    vb = bb.vision_backbone
    print("\n--- vision_backbone (Sam3DualViTDetNeck) ---")
    for name, child in vb.named_children():
        print(f"  {name:<32}{nparams(child):>14,d}  {human(nparams(child)*4)}")
    trunk = vb.trunk
    print(f"\n--- trunk = {type(trunk).__name__} ---")
    for name, child in trunk.named_children():
        print(f"  {name:<32}{nparams(child):>14,d}  {human(nparams(child)*4)}")

    lb = bb.language_backbone
    print(f"\n--- language_backbone = {type(lb).__name__} ---")
    for name, child in lb.named_children():
        print(f"  {name:<32}{nparams(child):>14,d}  {human(nparams(child)*4)}")
    enc = getattr(lb, "encoder", None)
    if enc is not None and hasattr(enc, "transformer"):
        print(f"  text encoder 层数 = {len(enc.transformer.resblocks)}"
              f" | width = {getattr(enc, 'width', '?')}"
              f" | context_length = {getattr(enc, 'context_length', '?')}")

    print("\n--- 其他关键子模块 ---")
    for attr in ("transformer", "geometry_encoder", "segmentation_head",
                 "dot_prod_scoring", "class_embed", "inst_interactive_predictor"):
        m = getattr(model, attr, None)
        if m is None:
            print(f"  {attr:<28}= None")
            continue
        print(f"  {attr:<28}{type(m).__name__:<26}{nparams(m):>14,d}"
              f"  {human(nparams(m)*4)}")

    tr = model.transformer
    dec = getattr(tr, "decoder", None)
    print(f"\n  d_model            = {getattr(tr, 'd_model', '?')}")
    print(f"  num_queries        = {getattr(dec, 'num_queries', '?')}")
    print(f"  NumDecoderLayers   = {len(getattr(dec, 'layers', []) or [])}")
    print(f"  num_feature_levels = {model.num_feature_levels}")
    print(f"  use_dot_prod_scoring = {model.use_dot_prod_scoring}")
    print(f"  multimask_output     = {model.multimask_output}")

    # ---------------- 前向拆解 ----------------
    print("\n" + "=" * 76)
    print(" 前向拆解实测 (1008x1008 FP32, TF32 开启)")
    print("=" * 76)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    img = Image.open(ROOT / "inputimage" / "1.jpg").convert("RGB")
    from torchvision.transforms import v2

    tf = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(1008, 1008)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = tf(v2.functional.to_image(img)).unsqueeze(0).to("cuda")
    print(f"  输入: {tuple(x.shape)} {x.dtype}")

    def bench(fn, n=3, warm=1, tag=""):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.time()
            fn()
            torch.cuda.synchronize()
            ts.append((time.time() - t) * 1000)
        arr = np.array(ts)
        print(f"  {tag:<36} 均值 {arr.mean():8.1f}ms | 最小 {arr.min():8.1f}ms")
        return float(arr.mean())

    t_img = bench(lambda: bb.forward_image(x), 3, 1, "forward_image (视觉编码)")

    with torch.no_grad():
        st = {}
        bo = bb.forward_image(x)
        st["backbone_out"] = bo
        st["original_height"], st["original_width"] = img.height, img.width
        t_txt = bench(lambda: bb.forward_text(["dog"], device="cuda"), 3, 1,
                      "forward_text (文本编码)")

        from sam3.model.sam3_image_processor import Sam3Processor
        proc = Sam3Processor(model, confidence_threshold=0.5, device="cuda")
        st_full = proc.set_image(img)

        def only_ground():
            s = dict(st_full)
            s["backbone_out"] = dict(s["backbone_out"])
            return proc.set_text_prompt(prompt="dog", state=s)

        t_g = bench(only_ground, 3, 1, "set_text_prompt (文本+grounding)")

        st_full2 = proc.set_image(img)
        t_e2e = bench(lambda: proc.set_text_prompt(prompt="dog", state=dict(st_full2)),
                      3, 1, "端到端 (视觉+文本+grounding)")

    print("\n  ---- 汇总 ----")
    print(f"  视觉编码器 : {t_img:8.1f}ms  ({100*t_img/t_e2e:.1f}% of e2e)")
    print(f"  文本+grounding: {t_g:8.1f}ms  ({100*t_g/t_e2e:.1f}% of e2e)")
    print(f"  端到端     : {t_e2e:8.1f}ms")

    print(f"\n[显存] 分配 {torch.cuda.memory_allocated()/2**30:.2f}GiB | "
          f"峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
    print(f"[探针总耗时] {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())