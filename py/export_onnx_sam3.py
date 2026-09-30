#!/usr/bin/env python3
"""把 SAM3 (sam3.pt) 导出为 ONNX，**FP32 精度**，输出到 models/ 目录。

为什么这样切分？
    SAM3 图像推理 = 视觉编码器 + 文本编码器 + grounding(编码器/解码器/分割头)
    实测(1008x1008, FP32, TF32)：
        vision_backbone.forward_image   790 ms   (~75% 端到端)
        language_backbone.forward_text   27 ms
        grounding (encoder+decoder+head) 239 ms
    → 视觉编码器是绝对瓶颈，且输入固定尺寸、无动态 shape，最适合 TensorRT。
    → 文本编码器干净、可独立导出，收益虽小但确定性高。
    → grounding 结构复杂（decoder 迭代细化、动态后处理），保留 PyTorch。

本脚本导出两个 ONNX（均为 FP32）：
    models/sam3_vision_encoder.onnx
        输入  : image             (1, 3, 1008, 1008)  float32
        输出  : fpn_0 (1,256,288,288) / fpn_1 (1,256,144,144) / fpn_2 (1,256,72,72)
                pos_0 / pos_1 / pos_2   （与 fpn_* 同形状）
        —— 对应 backbone.forward_image() 的 backbone_fpn / vision_pos_enc
    models/sam3_text_encoder.onnx
        输入  : tokens            (1, 32)  int64
        输出  : text_mask        (1, 32)  bool
                text_memory       (32, 1, 256) float32
                text_embeds       (32, 1, 1024) float32
        —— 对应 backbone.forward_text([prompt]) 的 language_mask/features/embeds

用法：
    ./venv310/bin/python export_onnx_sam3.py                  # 两个都导出
    ./venv310/bin/python export_onnx_sam3.py --part vision    # 只导视觉编码器
    ./venv310/bin/python export_onnx_sam3.py --part text
    ./venv310/bin/python export_onnx_sam3.py --no-verify      # 跳过数值校验

导出后构建 engine（FP32）：
    bash build_engine_sam3.sh models/sam3_vision_encoder.onnx fp32
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 包装模块（把 dict/list 输出展平成 ONNX 友好的多输出）
# ----------------------------------------------------------------------------
class VisionEncoderWrapper(nn.Module):
    """包装 Sam3DualViTDetNeck，固定输出 3 个 FPN 层 + 3 个位置编码。

    与 SAM3VLBackbone._forward_image_no_act_ckpt 的语义一致：
        sam3_features, sam3_pos, _, _ = vision_backbone.forward(images)
        再按 scalp 丢弃最低分辨率的层。
    实测 scalp=1（neck 出 4 层 → 保留 3 层），此处按实际值裁剪。
    """

    def __init__(self, vision_backbone: nn.Module, scalp: int):
        super().__init__()
        self.vb = vision_backbone
        self.scalp = scalp

    def forward(self, image: torch.Tensor):
        feats, pos, _, _ = self.vb(image)
        if self.scalp > 0:
            feats = feats[: -self.scalp]
            pos = pos[: -self.scalp]
        assert len(feats) == 3, f"期望 3 个 FPN 层，实际 {len(feats)}"
        return (
            feats[0], feats[1], feats[2],
            pos[0], pos[1], pos[2],
        )


class TextEncoderWrapper(nn.Module):
    """包装 VETextEncoder：输入已 tokenize 的 id 序列，输出文本特征。

    复刻 VETextEncoder.forward 中 text 为 str 的分支（去掉 tokenizer，
    以便 ONNX 只吃 int64 张量）：
        text_attention_mask = (tokenized != 0).bool().ne(1)
        _, text_memory      = encoder(tokenized)      # output_tokens=True
        text_memory_resized = resizer(text_memory.transpose(0, 1))
        text_embeds         = token_embedding(tokenized).transpose(0, 1)
    """

    def __init__(self, language_backbone: nn.Module):
        super().__init__()
        self.lb = language_backbone

    def forward(self, tokens: torch.Tensor):
        enc = self.lb.encoder
        text_attention_mask = (tokens != 0).bool().ne(1)
        _, text_memory = enc(tokens)                      # (B, L, 1024)
        text_memory = self.lb.resizer(text_memory.transpose(0, 1))   # (L, B, 256)
        text_embeds = enc.token_embedding(tokens).transpose(0, 1)    # (L, B, 1024)
        return text_attention_mask, text_memory, text_embeds


# ----------------------------------------------------------------------------
def build_model(ckpt: Path, bpe: Path, device: str):
    from sam3 import build_sam3_image_model

    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=str(bpe),
        checkpoint_path=str(ckpt),
        load_from_HF=False,
        enable_segmentation=True,
        device=device,
    )
    model.eval()

    # ------------------------------------------------------------------
    # ONNX 不支持复数类型（torch.polar / view_as_complex）。
    # SAM3 的 2D-RoPE 用复数实现，必须实数化后才能导出。
    # 该补丁数学等价，精度影响 <1 ulp。
    # ------------------------------------------------------------------
    from patch_sam3_rope import apply as apply_rope_patch

    n_patched = apply_rope_patch(model)
    print(f"  [RoPE] 已实数化 {n_patched} 个 Attention（ONNX 无复数类型）")

    print(f"  [模型] 加载完成 {time.time() - t0:.1f}s | device={device}")
    return model


def export_one(wrapper: nn.Module, args_io, out_path: Path, opset: int,
               device: str, input_names, output_names, dynamic_axes=None):
    """执行一次 torch.onnx.export 并打印结果。args_io 为 dummy 输入 tuple/list。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [导出] {out_path.name} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args_io,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,          # 用传统 tracer：对自定义模块最稳
        )
    dt = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"  [导出] 完成 {dt:.1f}s | 大小 {size_mb:.1f} MB")
    return dt


def check_onnx(out_path: Path):
    try:
        import onnx
    except ImportError:
        print("     (未安装 onnx，跳过结构校验)")
        return
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
           for i in m.graph.input]
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
            for o in m.graph.output]
    print(f"     ✅ ONNX 校验通过 | 节点数 {len(m.graph.node)}")
    print(f"        输入: {ins}")
    print(f"        输出: {outs}")


# ----------------------------------------------------------------------------
def export_vision(model, outdir: Path, size: int, opset: int, device: str):
    print("\n" + "=" * 72)
    print(" ① 视觉编码器 (Sam3DualViTDetNeck → 3×FPN + 3×pos)")
    print("=" * 72)
    bb = model.backbone
    scalp = int(getattr(bb, "scalp", 0))
    print(f"  scalp = {scalp}（丢弃最低分辨率层数）")

    wrap = VisionEncoderWrapper(bb.vision_backbone, scalp).to(device).eval()
    dummy = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)

    # 先跑一次 PyTorch 参考输出
    with torch.no_grad():
        ref = wrap(dummy)
    print("  参考输出形状: " + ", ".join(str(tuple(t.shape)) for t in ref))

    out = outdir / "sam3_vision_encoder.onnx"
    names = ["fpn_0", "fpn_1", "fpn_2", "pos_0", "pos_1", "pos_2"]
    export_one(wrap, (dummy,), out, opset, device,
               input_names=["image"], output_names=names)
    check_onnx(out)
    return out, ref, dummy


def export_text(model, outdir: Path, ckpt_dir_bpe, opset: int, device: str):
    print("\n" + "=" * 72)
    print(" ② 文本编码器 (VETextEncoder)")
    print("=" * 72)
    lb = model.backbone.language_backbone
    ctx = int(lb.context_length)
    tokens = lb.tokenizer(["dog"], context_length=ctx).to(device)
    print(f"  context_length={ctx} | 示例 tokens={tokens.tolist()}")

    wrap = TextEncoderWrapper(lb).to(device).eval()
    with torch.no_grad():
        ref = wrap(tokens)
    print("  参考输出形状: " + ", ".join(str(tuple(t.shape)) for t in ref))

    out = outdir / "sam3_text_encoder.onnx"
    export_one(wrap, (tokens,), out, opset, device,
               input_names=["tokens"],
               output_names=["text_mask", "text_memory", "text_embeds"])
    check_onnx(out)
    return out, ref, tokens


# ----------------------------------------------------------------------------
def verify(out_path: Path, wrapper, ref, dummy, out_names, device):
    """用 onnxruntime（若可用）对比 PyTorch 与 ONNX 的数值差异。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("     (未安装 onnxruntime，跳过数值校验；"
              "建议 pip install onnxruntime)")
        return

    print(f"\n  [校验] {out_path.name}")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: dummy.detach().cpu().numpy()}
    got = sess.run(None, feed)

    worst = 0.0
    for name, r, g in zip(out_names, ref, got):
        if isinstance(r, torch.Tensor) and r.dtype.is_floating_point:
            rn = r.detach().cpu().numpy()
            diff = float(abs(rn - g).max())
            rel = diff / (float(abs(rn).max()) + 1e-12)
            worst = max(worst, diff)
            print(f"     {name:<14} max|Δ|={diff:.3e}  rel={rel:.3e}")
        else:
            rn = r.detach().cpu().numpy()
            same = bool((rn == g).all())
            print(f"     {name:<14} 完全一致={same}")
    print(f"     最大绝对误差 = {worst:.3e}"
          f"  → {'✅ 可忽略' if worst < 1e-3 else '⚠️ 请检查'}")


# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="SAM3 → ONNX (FP32)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "models" / "sam3.pt"))
    p.add_argument("--bpe", default=str(ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"))
    p.add_argument("--outdir", default=str(ROOT / "models"), help="ONNX 输出目录")
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--part", choices=["vision", "text", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-verify", action="store_true", help="跳过数值校验")
    args = p.parse_args(argv)

    ckpt = Path(args.ckpt)
    bpe = Path(args.bpe)
    outdir = Path(args.outdir)
    for f, how in ((ckpt, "./download.sh"), (bpe, "./download.sh")):
        if not f.is_file():
            sys.exit(f"[错误] 缺少文件 {f}，请先执行 {how}")

    print("=" * 72)
    print(" SAM3  →  ONNX   (FP32)")
    print("=" * 72)
    print(f"  权重     : {ckpt}  ({ckpt.stat().st_size / 2**30:.2f} GiB)")
    print(f"  输出目录 : {outdir}")
    print(f"  输入尺寸 : 1 x 3 x {args.image_size} x {args.image_size}")
    print(f"  opset    : {args.opset}")
    print(f"  设备     : {args.device}")
    print("=" * 72)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [警告] CUDA 不可用，回退 CPU（导出会很慢）")
        args.device = "cpu"

    # FP32：明确关闭 TF32，避免导出时把数值"提前降精度"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    t_start = time.time()
    model = build_model(ckpt, bpe, args.device)

    results = []
    if args.part in ("both", "vision"):
        t0 = time.time()
        out, ref, dummy = export_vision(model, outdir, args.image_size,
                                       args.opset, args.device)
        results.append(("vision", out, ref, dummy, time.time() - t0))

    if args.part in ("both", "text"):
        t0 = time.time()
        out, ref, dummy = export_text(model, outdir, bpe, args.opset, args.device)
        results.append(("text", out, ref, dummy, time.time() - t0))

    if not args.no_verify:
        for kind, out, ref, dummy, _ in results:
            if kind == "vision":
                wrap = VisionEncoderWrapper(
                    model.backbone.vision_backbone,
                    int(getattr(model.backbone, "scalp", 0))).to(args.device).eval()
            else:
                wrap = TextEncoderWrapper(model.backbone.language_backbone).to(args.device).eval()
            verify(out, wrap, ref, dummy,
                   ["fpn_0", "fpn_1", "fpn_2", "pos_0", "pos_1", "pos_2"]
                   if kind == "vision" else
                   ["text_mask", "text_memory", "text_embeds"],
                   args.device)

    print("\n" + "=" * 72)
    print(" 完成")
    print("=" * 72)
    total = time.time() - t_start
    for kind, out, _, _, dt in results:
        print(f"  {kind:<8} {out}  ({out.stat().st_size / 1024**2:.1f} MB)"
              f"  用时 {dt:.1f}s")
    print(f"  总用时 {total:.1f}s")
    print("\n  下一步：构建 TensorRT engine（FP32）")
    print(f"    bash build_engine_sam3.sh {outdir}/sam3_vision_encoder.onnx fp32")
    return 0


if __name__ == "__main__":
    sys.exit(main())