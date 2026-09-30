#!/usr/bin/env python3
"""探针：打印 backbone.forward_image / forward_text / forward_grounding 的输入输出结构。"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")
import torch
from PIL import Image

def describe(name, obj, depth=0, max_depth=4):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict[{len(obj)}]")
        if depth < max_depth:
            for k, v in obj.items():
                describe(str(k), v, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}[{len(obj)}]")
        if depth < max_depth:
            for i, v in enumerate(obj):
                describe(f"[{i}]", v, depth + 1, max_depth)
    elif isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: Tensor{tuple(obj.shape)} {obj.dtype} dev={obj.device}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {repr(obj)[:80]}")

def main():
    from sam3 import build_sam3_image_model
    bpe = ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    ckpt = ROOT / "models" / "sam3.pt"
    model = build_sam3_image_model(bpe_path=str(bpe), checkpoint_path=str(ckpt),
                                   load_from_HF=False, enable_segmentation=True, device="cuda")
    model.eval()
    bb = model.backbone

    from torchvision.transforms import v2
    img = Image.open(ROOT / "inputimage" / "1.jpg").convert("RGB")
    tf = v2.Compose([v2.ToDtype(torch.uint8, scale=True), v2.Resize(size=(1008, 1008)),
                     v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    x = tf(v2.functional.to_image(img)).unsqueeze(0).to("cuda")

    print("=" * 78)
    print(" backbone.forward_image 输出结构")
    print("=" * 78)
    with torch.no_grad():
        bo = bb.forward_image(x)
    describe("out", bo)

    print("\n" + "=" * 78)
    print(" backbone.forward_text 输出结构")
    print("=" * 78)
    with torch.no_grad():
        to = bb.forward_text(["dog"], device="cuda")
    describe("out", to)

    print("\n" + "=" * 78)
    print(" 文本 tokenizer 结构")
    print("=" * 78)
    lb = bb.language_backbone
    tok = lb.tokenizer(["dog"], context_length=lb.context_length)
    print("  tokenized:", tuple(tok.shape), tok.dtype)
    print("  tokens:", tok.tolist())

    print("\n" + "=" * 78)
    print(" vision_backbone.forward (Sam3DualViTDetNeck) 输出结构")
    print("=" * 78)
    with torch.no_grad():
        vb_out = bb.vision_backbone.forward(x)
    describe("neck_out", vb_out)

    print("\n" + "=" * 78)
    print(" forward_grounding 输出结构")
    print("=" * 78)
    from sam3.model.sam3_image_processor import Sam3Processor
    proc = Sam3Processor(model, confidence_threshold=0.0, device="cuda")
    st = proc.set_image(img)
    st = proc.set_text_prompt("dog", st)
    print("  masks:", st["masks"].shape, " boxes:", st["boxes"].shape, " scores:", st["scores"].shape)

    # encoder_out 结构
    with torch.no_grad():
        out = model.forward_grounding(
            backbone_out=st["backbone_out"], find_input=proc.find_stage,
            geometric_prompt=st["geometric_prompt"], find_target=None)
    describe("grounding_out", out, max_depth=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())