#!/usr/bin/env python3
"""验证 TensorRT engine 与 PyTorch 的数值一致性（精度对比）。

输出：
  1. vision encoder 张量级对比（fpn / pos）
  2. text  encoder  张量级对比（text_memory / text_embeds）
  3. 端到端 masks 对比（box 提示，比较 IoU）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from trt_runner import TRTEngine  # noqa: E402


def stats(a, b, name):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() != b.numel():
        print(f"  {name:18s} 元素数不等 {a.numel()} vs {b.numel()}")
        return None
    diff = (a - b).abs()
    denom = a.abs().max().item() or 1.0
    mx = diff.max().item()
    print(f"  {name:18s} max|D|={mx:.3e}  mean|D|={diff.mean().item():.3e}  "
          f"rel={mx/denom:.3e}  (|a|max={a.abs().max().item():.4f})")
    return mx


def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from sam3 import build_sam3_image_model
    from patch_sam3_rope import apply as apply_rope

    print("=" * 76)
    print(" TensorRT vs PyTorch 精度对比")
    print("=" * 76)

    print("\n[1] 加载模型 ...")
    model = build_sam3_image_model(
        bpe_path=str(ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        checkpoint_path=str(ROOT / "models" / "sam3.pt"),
        load_from_HF=False, enable_segmentation=True, device="cuda")
    model.eval()
    apply_rope(model)
    print("    完成")

    from sam3.model.sam3_image_processor import Sam3Processor
    trt_v = TRTEngine(ROOT / "models" / "sam3_vision_encoder.engine")
    trt_t = TRTEngine(ROOT / "models" / "sam3_text_encoder.engine")

    img_path = ROOT / "inputimage" / "1.jpg"
    image = Image.open(img_path).convert("RGB")
    iw, ih = image.size

    # ---- 捕获预处理后的图像张量（经 numpy 转出 inference_mode）----
    cap = {}
    saved_fi = model.backbone.forward_image
    saved_ft = model.backbone.forward_text

    def spy(samples):
        if "x" not in cap:
            cap["x"] = samples.detach().cpu().numpy().copy()
        return saved_fi(samples)

    model.backbone.forward_image = spy
    proc0 = Sam3Processor(model, confidence_threshold=0.5, device="cuda")
    proc0.set_image(image)
    model.backbone.forward_image = saved_fi

    x = torch.from_numpy(cap["x"]).to("cuda")
    print(f"\n[2] 预处理输入张量 {tuple(x.shape)} {x.dtype}")

    # ---- vision 张量级对比 ----
    print("\n[3] vision encoder 对比 (PyTorch vs TRT)")
    with torch.no_grad():
        ref = saved_fi(x)
    eng = trt_v({"image": x})
    vm = []
    for i in range(3):
        vm.append(stats(ref["backbone_fpn"][i], eng[f"fpn_{i}"], f"fpn_{i}"))
    for i in range(3):
        vm.append(stats(ref["vision_pos_enc"][i], eng[f"pos_{i}"], f"pos_{i}"))

    # ---- text 张量级对比 ----
    print("\n[4] text encoder 对比 (PyTorch vs TRT)")
    lb = model.backbone.language_backbone
    toks = lb.tokenizer(["dog"], context_length=int(lb.context_length)).to("cuda")
    with torch.no_grad():
        tro = saved_ft(["dog"], device="cuda")
    eng_t = trt_t({"tokens": toks.long()})
    tm_stat = stats(tro["language_features"], eng_t["text_memory"], "language_features")
    stats(tro["language_embeds"], eng_t["text_embeds"], "language_embeds")
    m1 = tro["language_mask"].detach().bool().flatten()
    m2 = eng_t["text_mask"].detach().bool().flatten()
    same = int((m1 == m2).sum().item())
    print(f"  {'language_mask':18s} 一致元素 {same}/{m1.numel()}")

    # ---- 端到端 masks 对比（box 提示）----
    print("\n[5] 端到端对比（box 提示，比较 mask IoU）")
    bx, by = iw * 0.35, ih * 0.30
    bw, bh = iw * 0.30, ih * 0.40
    norm_box = [(bx + bw / 2) / iw, (by + bh / 2) / ih, bw / iw, bh / ih]

    def run(use_trt):
        if use_trt:
            from sam3_trt_infer import TRTEngines, install_trt_backbone
            install_trt_backbone(model, TRTEngines(), True, True, verbose=False)
        else:
            model.backbone.forward_image = saved_fi
            model.backbone.forward_text = saved_ft
        p = Sam3Processor(model, confidence_threshold=0.5, device="cuda")
        st = p.set_image(image)
        st = p.add_geometric_prompt(state=st, box=norm_box, label=True)
        return st

    st_t = run(False)
    m_t, sc_t = st_t["masks"].cpu().numpy(), st_t["scores"].cpu().numpy()
    st_r = run(True)
    m_r, sc_r = st_r["masks"].cpu().numpy(), st_r["scores"].cpu().numpy()

    print(f"    PyTorch: 目标数 {len(sc_t)}  scores={np.round(sc_t, 4).tolist()}")
    print(f"    TRT    : 目标数 {len(sc_r)}  scores={np.round(sc_r, 4).tolist()}")

    n = min(len(m_t), len(m_r))
    ious = []
    for i in range(n):
        a = m_t[i].squeeze().astype(bool)
        b = m_r[i].squeeze().astype(bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        iou = inter / union if union else 1.0
        ious.append(iou)
        print(f"    mask[{i}] IoU={iou:.6f}  差异像素={int(np.logical_xor(a,b).sum())}"
              f"  (面积 {int(a.sum())} vs {int(b.sum())})")

    if n and sc_t.shape == sc_r.shape:
        d = np.abs(sc_t - sc_r)
        print(f"    score 最大差 {d.max():.3e}")

    # ---- 汇总 ----
    print("\n" + "=" * 76)
    ok_v = all(v is not None and v < 1e-2 for v in vm)
    ok_t = tm_stat is not None and tm_stat < 1e-2
    ok_m = bool(ious) and min(ious) > 0.99
    print(f" vision 数值一致(<1e-2): {'是' if ok_v else '否'}")
    print(f" text   数值一致(<1e-2): {'是' if ok_t else '否'}")
    if ious:
        print(f" mask  IoU 最低值      : {min(ious):.6f}  {'通过(>0.99)' if ok_m else '需检查'}")
    print("\n注：TRT 与 PyTorch 均默认启用 TF32，二者应高度一致；")
    print("    与 ONNX Runtime 的纯 IEEE FP32 结果相比则可能有 ~1e-3 差异。")
    print("=" * 76)


if __name__ == "__main__":
    main()