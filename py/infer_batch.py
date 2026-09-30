#!/usr/bin/env python3
"""SAM3 批量分割：对目录下所有图像用同一个文本提示做分割。

特点：
  * 模型只加载一次，多张图复用（避免重复加载 3.4GB 权重）
  * 输出可视化图 + 可选掩码 + CSV 结果

用法::

    python infer_batch.py --input inputimage --text "dog" --outdir outputimage
    python infer_batch.py --input inputimage --text "car" --csv out.csv --save-mask
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")

import numpy as np
import torch
from PIL import Image

from sam3_infer import build_model, normalize_mask, overlay  # 复用单图脚本逻辑


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM3 批量文本提示分割",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="输入目录或单张图像")
    p.add_argument("--text", required=True, help="文本提示，如 'dog'")
    p.add_argument("--outdir", default=None, help="可视化输出目录")
    p.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png", "bmp", "webp"],
                   help="扫描的扩展名")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-masks", type=int, default=20)
    p.add_argument("--save-mask", action="store_true", help="额外导出二值掩码 PNG")
    p.add_argument("--csv", default=None, help="把每个目标写入 CSV")
    p.add_argument("--bpe", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tf32", action="store_true")
    return p.parse_args(argv)


def collect_images(inp: Path, exts) -> list[Path]:
    if inp.is_file():
        return [inp]
    if not inp.is_dir():
        sys.exit(f"[错误] 路径不存在: {inp}")
    exts = {e.lower().lstrip(".") for e in exts}
    return sorted(f for f in inp.rglob("*")
                  if f.is_file() and f.suffix.lower().lstrip(".") in exts)


def main(argv=None) -> int:
    args = parse_args(argv)
    images = collect_images(Path(args.input), args.ext)
    if not images:
        sys.exit(f"[错误] 在 {args.input} 未找到匹配图像（扩展名 {args.ext}）")

    print(f"[信息] 共 {len(images)} 张图像，提示词 = {args.text!r}")
    model = build_model(args)

    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model, confidence_threshold=args.threshold, device=args.device)

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    maskdir = (outdir / "masks") if (outdir and args.save_mask) else None
    if maskdir:
        maskdir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.time()
    for i, path in enumerate(images, 1):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(images)}] {path.name} 跳过（无法读取: {e}）")
            continue

        t0 = time.time()
        state = processor.set_image(image)
        t_enc = time.time() - t0

        t0 = time.time()
        state = processor.set_text_prompt(prompt=args.text, state=state)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t_prm = time.time() - t0

        masks = state["masks"].cpu().numpy()
        boxes = state["boxes"].cpu().numpy()
        scores = state["scores"].cpu().numpy()

        print(f"  [{i}/{len(images)}] {path.name}: 目标 {len(scores)} 个 | "
              f"编码 {t_enc*1000:.0f}ms 提示 {t_prm*1000:.0f}ms")

        if outdir:
            overlay(image, masks, boxes, scores, args.alpha, args.max_masks) \
                .save(outdir / f"{path.stem}_seg.png")

        if maskdir:
            for k in range(len(masks)):
                m = normalize_mask(masks[k], image.size)
                Image.fromarray((m * 255).astype(np.uint8)) \
                    .save(maskdir / f"{path.stem}_mask{k}.png")

        for k in range(len(scores)):
            x0, y0, x1, y1 = boxes[k]
            rows.append({
                "image": path.name, "index": k, "score": round(float(scores[k]), 4),
                "x0": round(float(x0), 1), "y0": round(float(y0), 1),
                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                "area_px": int(masks[k].sum()),
            })

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[输出] CSV -> {args.csv}（{len(rows)} 行）")

    print(f"[汇总] {len(images)} 张图像，共 {len(rows)} 个目标，耗时 {time.time() - t_all:.1f}s")
    if outdir:
        print(f"[输出] 可视化 -> {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())