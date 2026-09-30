#!/usr/bin/env python3
"""SAM3 图像分割推理（AGX Orin）：文本提示 / 框提示 / 点提示。

官方用法（examples/sam3_image_predictor_example.ipynb）::

    model = build_sam3_image_model(bpe_path=..., checkpoint_path=..., load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    state = processor.set_image(image)                # 视觉编码（只做一次）
    state = processor.set_text_prompt("dog", state)   # 文本提示
    # state["masks"] (N,H,W) bool / state["boxes"] (N,4) xyxy / state["scores"] (N,)

用法::

    ./sh/run.sh --image a.jpg --text "dog"   --out out.png
    ./sh/run.sh --image a.jpg --box 480 290 110 360 --out out.png
    ./sh/run.sh --image a.jpg --point 490 167 --out out.png
    ./sh/run.sh --image a.jpg --point 490 167 384 90 --point-label 1 1 --out out.png

本模块同时被 :mod:`sam3_server`（常驻服务）复用：服务的每次请求都会调用
:func:`prepare_args` + :func:`run_image`，因此「命令行单次运行」与「常驻服务推理」
走的是**同一套代码**，输出完全一致。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")  # Jetson 无 triton

_T_LAUNCH = time.time()          # 进程启动时刻（初始化耗时对比基准）

import numpy as np
import torch
from PIL import Image

_T_IMPORT = time.time() - _T_LAUNCH   # numpy/torch/PIL 导入完成耗时

PALETTE = [(255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 200, 0),
           (255, 0, 255), (0, 220, 220), (160, 80, 255), (255, 128, 0)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM3 图像分割（文本/框/点提示）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image", required=True, help="输入图像路径")
    p.add_argument("--text", default=None, help="文本提示，如 'dog'")
    p.add_argument("--box", nargs=4, type=float, default=None, metavar=("X", "Y", "W", "H"),
                   help="框提示：左上角 x y + 宽 高（像素）")
    p.add_argument("--label", type=int, default=1, help="框提示标签：1 正样本 / 0 负样本")
    p.add_argument("--point", nargs="+", type=float, default=None, metavar="X Y",
                   help="点提示：像素坐标，成对给出，可多对，如 --point 490 167 384 90")
    p.add_argument("--point-label", nargs="+", type=int, default=None,
                   help="每个点的标签：1 正样本 / 0 负样本，数量需与点一致（默认全 1）")
    p.add_argument("--no-box", action="store_true", help="只绘制掩码，不绘制目标框与分数")
    p.add_argument("--keep-all", action="store_true",
                   help="点提示时保留全部候选目标（默认仅保留包含正样本点的掩码）")
    p.add_argument("--out", default=None, help="输出可视化图像")
    p.add_argument("--mask-dir", default=None, help="导出每个掩码 PNG 到此目录")
    p.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    p.add_argument("--max-masks", type=int, default=20, help="最多可视化前 N 个")
    p.add_argument("--alpha", type=float, default=0.5, help="掩码透明度")
    p.add_argument("--bench", action="store_true", help="打印耗时基准")
    p.add_argument("--repeat", type=int, default=5, help="基准重复次数")
    p.add_argument("--bpe", default=None, help="BPE 词表路径")
    p.add_argument("--ckpt", default=None, help="权重路径")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    return p.parse_args(argv)


def prepare_args(argv=None):
    """解析命令行并做基本校验（本地运行与常驻服务共用）。

    失败时抛 ``SystemExit``（携带中文错误信息），由调用方决定是退出进程还是回给客户端。
    """
    args = parse_args(argv)
    if not Path(args.image).is_file():
        sys.exit(f"[错误] 找不到图像: {args.image}")
    if not args.text and args.box is None and args.point is None:
        sys.exit("[错误] 必须提供 --text、--box 或 --point")
    args.point_pairs, args.point_labels_p = parse_points(args)
    return args


def build_model(args):
    from sam3 import build_sam3_image_model

    bpe = Path(args.bpe) if args.bpe else ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / "models" / "sam3.pt"
    for f, how in ((bpe, "./sh/download.sh"), (ckpt, "./sh/download.sh")):
        if not f.is_file():
            sys.exit(f"[错误] 缺少文件 {f}，请先执行 {how}")

    if not args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=str(bpe), checkpoint_path=str(ckpt),
        load_from_HF=False, enable_segmentation=True, device=args.device,
    )
    model.eval()
    print(f"[信息] 模型加载 {time.time() - t0:.1f}s | 设备 {args.device}")
    return model


def parse_points(args):
    """解析 --point 的扁平坐标列表 -> ([(x, y), ...], [label, ...])。"""
    if args.point is None:
        return [], []
    coords = list(args.point)
    if len(coords) < 2 or len(coords) % 2 != 0:
        sys.exit("[错误] --point 需要成对的 x y 坐标，如 --point 490 167 384 90")
    pairs = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    labels = list(args.point_label) if args.point_label else [1] * len(pairs)
    if len(labels) != len(pairs):
        sys.exit(f"[错误] --point-label 数量({len(labels)})与点数({len(pairs)})不一致")
    return pairs, labels


def add_point_prompt(processor, image, state, points_px, labels):
    """点提示（像素坐标）——复刻 Sam3Processor.add_geometric_prompt，
    只是把 append_boxes 换成 append_points。

    SAM3 几何编码器要求点坐标为**归一化 [0,1]**，形状 (N_points, B, 2)，
    标签形状 (N_points, B)。纯几何提示（无文本）时先注入 "visual" 哑文本，
    模型才会只依赖几何信息。
    """
    if "backbone_out" not in state:
        raise ValueError("必须先调用 set_image()")
    if "language_features" not in state["backbone_out"]:
        state["backbone_out"].update(
            processor.model.backbone.forward_text(["visual"], device=processor.device))
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    iw, ih = image.size
    pts = torch.tensor([[[x / iw, y / ih]] for (x, y) in points_px],
                       device=processor.device, dtype=torch.float32)
    lbls = torch.tensor([[int(v)] for v in labels],
                        device=processor.device, dtype=torch.bool)
    state["geometric_prompt"].append_points(pts, lbls)
    return processor._forward_grounding(state)


def filter_by_points(state, image, points_px, labels):
    """只保留掩码中包含任一「正样本」提示点的目标。

    SAM3 的 grounding 头会一次性输出所有候选物体（点只是"几何提示"之一，
    并不会把结果限制在点所在的物体上）。而点分割的常见语义是"点谁分谁"，
    因此这里按点做一次筛选：掩码在正样本点像素处为 True 的才保留。
    """
    masks = state.get("masks")
    if masks is None or masks.numel() == 0:
        return state
    iw, ih = image.size
    pos = [(x, y) for (x, y), lb in zip(points_px, labels) if lb]
    if not pos:
        return state
    keep = []
    for i in range(masks.shape[0]):
        m = masks[i]                      # (1, H, W)
        for (x, y) in pos:
            xi = min(max(int(round(x)), 0), iw - 1)
            yi = min(max(int(round(y)), 0), ih - 1)
            if bool(m[..., yi, xi].any()):
                keep.append(i)
                break
    if not keep:
        return state

    # 去嵌套：含点的候选中常同时存在「物体本身」和「把物体包住的大区域」
    # （如桌面/地面）。按面积升序，若某个大掩码几乎完整包含了已保留的小掩码
    # （包含率 > 阈值），则丢弃它，只留下点所在的**最具体**物体。
    cand = sorted(keep, key=lambda i: int(masks[i].sum()))
    sel: list[int] = []
    np_masks = masks.detach().cpu().numpy().astype(bool)
    for i in cand:
        mi = np_masks[i].reshape(-1)
        ai = int(mi.sum())
        redundant = False
        for j in sel:
            mj = np_masks[j].reshape(-1)
            aj = int(mj.sum())
            if aj and np.logical_and(mi, mj).sum() / min(ai, aj) > 0.9:
                redundant = True      # 与已保留掩码高度重叠（互为父子）
                break
        if not redundant:
            sel.append(i)

    idx = torch.tensor(sel, device=masks.device, dtype=torch.long)
    out = dict(state)
    out["masks"] = masks[idx]
    if "boxes" in state:
        out["boxes"] = state["boxes"][idx]
    if "scores" in state:
        out["scores"] = state["scores"][idx]
    return out


def normalize_mask(mask, size):
    """把单个掩码规整为与图像同尺寸的 (H, W) bool 数组。

    SAM3 的 ``state["masks"]`` 形状为 ``(N, 1, H, W)``（比常见的 (N,H,W)
    多一个通道维）。若直接用 ``masks[i]`` 做布尔索引，numpy 会用第 0 维
    （长度 1）去索引图像的第 0 维（高度），从而报错：
        IndexError: boolean index did not match indexed array along dimension 0
    这里统一 squeeze 成 2D，尺寸不符时用最近邻缩放回原图大小。

    size: PIL 图像尺寸，即 (W, H)
    """
    w, h = size
    a = np.asarray(mask).astype(np.float32).squeeze()
    if a.ndim != 2:                       # (1, H, W) -> (H, W)
        a = a.reshape(a.shape[-2], a.shape[-1])
    if a.max() > 1.0:                     # 0-255 的掩码归一到 0-1
        a = a / 255.0
    if a.shape != (h, w):                 # 尺寸不符则最近邻缩放
        a = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    return a > 0.5


def overlay(image, masks, boxes, scores, alpha, max_masks, points=None, point_labels=None,
            draw_box=True):
    import cv2
    vis = np.array(image.convert("RGB"), dtype=np.float32)
    n = min(len(masks), max_masks)
    for i in range(n):
        m = normalize_mask(masks[i], image.size)
        vis[m] = vis[m] * (1 - alpha) + np.array(PALETTE[i % len(PALETTE)], dtype=np.float32) * alpha
    vis = vis.astype(np.uint8)
    for i in range(n if draw_box else 0):
        x0, y0, x1, y1 = boxes[i].astype(int)
        c = PALETTE[i % len(PALETTE)]
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2)
        cv2.putText(vis, f"{i}:{scores[i]:.2f}", (x0, max(0, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    # 画出提示点：正样本绿十字，负样本红十字（便于确认点位置）
    if points:
        for (px, py), lb in zip(points, point_labels or [1] * len(points)):
            c = (0, 255, 0) if lb else (255, 0, 0)
            cv2.drawMarker(vis, (int(px), int(py)), c, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(vis, (int(px), int(py)), 6, c, 1)
    return Image.fromarray(vis)


def run_once(processor, image, args):
    """一次「图像编码 + 提示前向」，返回 (state, 编码秒, 提示秒)。"""
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)
    t0 = time.time()
    state = processor.set_image(image)
    sync()
    t_enc = time.time() - t0

    t0 = time.time()
    if args.text:
        state = processor.set_text_prompt(prompt=args.text, state=state)
    elif args.point is not None:
        state = add_point_prompt(processor, image, state,
                                 args.point_pairs, args.point_labels_p)
    else:
        x, y, w, h = args.box
        iw, ih = image.size
        state = processor.add_geometric_prompt(
            state=state, box=[(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih],
            label=bool(args.label))
    sync()
    return state, t_enc, time.time() - t0


def report(image, state, args):
    """点筛选 -> 打印结果 -> 保存掩码/可视化，返回（可能已筛选的）state。"""
    if args.point is not None and not args.keep_all:
        n_before = int(state["masks"].shape[0])
        state = filter_by_points(state, image, args.point_pairs, args.point_labels_p)
        n_after = int(state["masks"].shape[0])
        if n_after < n_before:
            print(f"[过滤] 点提示：候选 {n_before} 个 -> 保留含正样本点的 {n_after} 个"
                  f"（如要保留全部请加 --keep-all）")

    masks = state["masks"].cpu().numpy()
    boxes = state["boxes"].cpu().numpy()
    scores = state["scores"].cpu().numpy()

    if args.text:
        desc = f'text("{args.text}")'
    elif args.point is not None:
        desc = f"point{args.point_pairs}"
    else:
        desc = f"box{list(args.box)}"
    print(f"\n[结果] 提示={desc} | 目标数 {len(scores)}（阈值 {args.threshold}）")
    print(f"       masks 形状 {tuple(np.shape(masks))} | boxes 形状 {tuple(np.shape(boxes))}")
    for i in range(min(len(scores), args.max_masks)):
        x0, y0, x1, y1 = boxes[i]
        print(f"       #{i}: score={scores[i]:.3f} box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) "
              f"面积={int(masks[i].sum())}px")

    if args.mask_dir:
        d = Path(args.mask_dir); d.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        for i in range(len(masks)):
            m = normalize_mask(masks[i], image.size)
            Image.fromarray((m * 255).astype(np.uint8)).save(d / f"{stem}_mask{i}.png")
        print(f"[输出] 掩码 -> {d}/")

    if args.out:
        out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay(image, masks, boxes, scores, args.alpha, args.max_masks,
                args.point_pairs, args.point_labels_p,
                draw_box=not args.no_box).save(out_path)
        print(f"[输出] 可视化 -> {out_path}")
    return state


def run_image(processor, args, warmup=False, announce=True):
    """读图 -> 推理 -> 打印/保存（本地运行与常驻服务共用的入口）。

    warmup=True: 跑两次并分别打印「首次 / 热身后」耗时（本地单次运行用）。
    warmup=False: 只跑一次，打印「推理」耗时（常驻服务用，模型已预热）。
    """
    image = Image.open(args.image).convert("RGB")
    if announce:
        print(f"[信息] 图像 {args.image}  {image.size[0]}x{image.size[1]}")

    if warmup:
        _, e1, p1 = run_once(processor, image, args)
        print(f"[计时] 首次推理:   图像编码 {e1*1000:7.1f}ms | 提示前向 {p1*1000:7.1f}ms "
              f"| 合计 {(e1 + p1)*1000:7.1f}ms")
        state, e2, p2 = run_once(processor, image, args)
        print(f"[计时] 热身后推理: 图像编码 {e2*1000:7.1f}ms | 提示前向 {p2*1000:7.1f}ms "
              f"| 合计 {(e2 + p2)*1000:7.1f}ms")
    else:
        state, e, p = run_once(processor, image, args)
        print(f"[计时] 推理:       图像编码 {e*1000:7.1f}ms | 提示前向 {p*1000:7.1f}ms "
              f"| 合计 {(e + p)*1000:7.1f}ms")
    return report(image, state, args)


def main(argv=None):
    args = prepare_args(argv)

    t_build = time.time()
    model = build_model(args)
    t_model = time.time() - t_build

    from sam3.model.sam3_image_processor import Sam3Processor
    t_p = time.time()
    processor = Sam3Processor(model, confidence_threshold=args.threshold, device=args.device)
    t_proc = time.time() - t_p

    print(f"[计时] 初始化: 导入三方库 {_T_IMPORT:.1f}s | 模型加载 {t_model:.1f}s | "
          f"Processor {t_proc:.2f}s | 初始化总计 {time.time() - _T_LAUNCH:.1f}s")

    if args.bench:
        print(f"\n[基准] 提示={args.text or args.box} 重复{args.repeat}次")
        image = Image.open(args.image).convert("RGB")
        run_once(processor, image, args)  # 预热
        enc, prm, tot = [], [], []
        for i in range(args.repeat):
            t0 = time.time()
            st, e, pr = run_once(processor, image, args)
            tot.append((time.time() - t0) * 1000); enc.append(e * 1000); prm.append(pr * 1000)
            print(f"  #{i+1}: 编码 {e*1000:7.1f}ms | 提示 {pr*1000:7.1f}ms | "
                  f"合计 {tot[-1]:7.1f}ms | 目标 {len(st['scores'])}")
        for name, a in (("图像编码", enc), ("提示前向", prm), ("端到端", tot)):
            a = np.array(a)
            print(f"[汇总] {name}: 均值 {a.mean():.1f}ms | 中位 {np.median(a):.1f}ms | "
                  f"最小 {a.min():.1f}ms | 最大 {a.max():.1f}ms")
        if args.device == "cuda":
            print(f"[显存] 分配 {torch.cuda.memory_allocated()/2**30:.2f}GiB | "
                  f"峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
        return 0

    run_image(processor, args, warmup=True, announce=True)
    print(f"[计时] 端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())