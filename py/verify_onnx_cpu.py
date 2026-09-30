#!/usr/bin/env python3
"""公平精度校验：PyTorch(CPU) vs ONNX Runtime(CPU)。

之前 export_onnx_sam3.py 里的校验是「CUDA 参考 vs CPU-ORT」，
PyTorch 在 CUDA 上走的是 cuDNN/TF32 等不同 kernel，与 CPU 结果本就不同，
因此 fpn 的 ~7% 相对误差主要来自设备/后端差异，而非导出错误。

本脚本同设备对比，得到的才是导出本身的精度损失。
"""
import os
os.environ.setdefault("USE_PERFLIB", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEV = "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_model():
    from sam3 import build_sam3_image_model
    from patch_sam3_rope import apply as apply_rope
    m = build_sam3_image_model(
        bpe_path=str(ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        checkpoint_path=str(ROOT / "models" / "sam3.pt"),
        load_from_HF=False, enable_segmentation=True, device=DEV)
    m.eval()
    apply_rope(m)
    return m


def main():
    import onnxruntime as ort
    from export_onnx_sam3 import VisionEncoderWrapper, TextEncoderWrapper

    print("=" * 70)
    print(" 公平精度校验：PyTorch(CPU)  vs  ONNX Runtime(CPU)")
    print("=" * 70)
    m = load_model()

    # ---------- vision ----------
    print("\n[1/2] Vision Encoder (1x3x1008x1008)")
    wrap = VisionEncoderWrapper(m.backbone.vision_backbone,
                                int(getattr(m.backbone, "scalp", 0))).eval()
    x = torch.randn(1, 3, 1008, 1008)
    t0 = time.time()
    with torch.no_grad():
        ref = wrap(x)
    print(f"  PyTorch(CPU) 前向: {time.time()-t0:.1f}s")

    sess = ort.InferenceSession(str(ROOT / "models" / "sam3_vision_encoder.onnx"),
                                providers=["CPUExecutionProvider"])
    t0 = time.time()
    got = sess.run(None, {"image": x.numpy()})
    print(f"  ONNX(CPU) 前向  : {time.time()-t0:.1f}s")

    names = ["fpn_0", "fpn_1", "fpn_2", "pos_0", "pos_1", "pos_2"]
    worst = 0.0
    for n, r, g in zip(names, ref, got):
        rn = r.detach().numpy()
        d = float(np.abs(rn - g).max())
        rel = d / (float(np.abs(rn).max()) + 1e-12)
        worst = max(worst, rel)
        flag = "✅" if rel < 1e-3 else ("⚠️" if rel < 1e-2 else "❌")
        print(f"    {n:8s} shape={tuple(g.shape)} max|Δ|={d:.3e} rel={rel:.3e} {flag}")

    # ---------- text ----------
    print("\n[2/2] Text Encoder (1x32 tokens)")
    lb = m.backbone.language_backbone
    twrap = TextEncoderWrapper(lb).eval()
    tok = lb.tokenizer(["dog"], context_length=int(lb.context_length))
    with torch.no_grad():
        tref = twrap(tok)
    tsess = ort.InferenceSession(str(ROOT / "models" / "sam3_text_encoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    tgot = tsess.run(None, {"tokens": tok.numpy()})
    tn = ["text_mask", "text_memory", "text_embeds"]
    for n, r, g in zip(tn, tref, tgot):
        rn = r.detach().numpy()
        if rn.dtype == bool or n == "text_mask":
            print(f"    {n:12s} 完全一致={bool((rn == g).all())} ✅")
        else:
            d = float(np.abs(rn - g).max())
            rel = d / (float(np.abs(rn).max()) + 1e-12)
            worst = max(worst, rel)
            flag = "✅" if rel < 1e-3 else ("⚠️" if rel < 1e-2 else "❌")
            print(f"    {n:12s} max|Δ|={d:.3e} rel={rel:.3e} {flag}")

    print("\n" + "=" * 70)
    print(f" 综合最大相对误差 = {worst:.3e}")
    print(" 结论：ONNX 导出精度损失可忽略 ✅" if worst < 1e-3
          else f" 结论：相对误差 {worst:.3e}，需检查 ⚠️")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())