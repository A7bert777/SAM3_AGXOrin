#!/usr/bin/env python3
"""SAM3 用 TensorRT engine 推理（命令行与 sam3_infer.py 完全兼容）。

用法与 run.sh 一致：
    ./run_trt.sh --image inputimage/1.jpg --text "dog" --out outputimage/1_dog.png

原理（monkey-patch，不修改 SAM3 源码）：
    把 model.backbone.forward_image / forward_text 替换为调用 TRT engine，
    并**沿用 PyTorch 原函数返回的 dict 结构**，因此 Sam3Processor 完全无感。

vision 侧必须按语义填充（依据 vl_combiner.py 的 _forward_image_no_act_ckpt）：
    vision_features    = backbone_fpn[-1]     <- 单个张量，是列表的【末元素】
    vision_pos_enc     = 位置编码【列表】
    backbone_fpn       = FPN 特征【列表】
    sam2_backbone_out  = None
且 scalp=1 会丢弃最低分辨率那层。故用原型 dict 推断保留层数，再逐层填入。

可用 --vision-backend / --text-backend 单独切回 pytorch 做 A/B 对比。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("USE_PERFLIB", "0")
sys.path.insert(0, str(ROOT))

_T_LAUNCH = time.time()          # 进程启动时刻（初始化耗时对比基准）

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from trt_runner import TRTEngine  # noqa: E402

_T_IMPORT = time.time() - _T_LAUNCH   # numpy/torch/PIL 导入完成耗时

PALETTE = [(255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 200, 0),
           (255, 0, 255), (0, 220, 220), (160, 80, 255), (255, 128, 0)]


# ---------------------------------------------------------------------- #
# 形状适配：把 engine 输出的排布调成与 PyTorch 槽位一致
# ---------------------------------------------------------------------- #
def _fit(slot: torch.Tensor, val: torch.Tensor):
    """把 val 调成与 slot 同形状；失败返回 None。"""
    if slot.shape == val.shape:
        return val
    if val.dim() == slot.dim():
        cands = [(1, 0, 2), (0, 2, 1), (2, 0, 1), (1, 0)]
        for perm in cands:
            if len(perm) != val.dim():
                continue
            t = val.permute(*perm)
            if t.shape == slot.shape:
                return t.contiguous()
    if val.numel() == slot.numel():
        return val.reshape(slot.shape).contiguous()
    return None


def _describe(d, prefix="", verbose=True):
    """打印嵌套结构，用于确认 scalp 后保留的层数。"""
    if not verbose:
        return
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"    {prefix}{k}: Tensor{tuple(v.shape)}")
        elif isinstance(v, (list, tuple)):
            shapes = [tuple(t.shape) for t in v if torch.is_tensor(t)]
            print(f"    {prefix}{k}: {type(v).__name__} len={len(v)} {shapes}")
        elif isinstance(v, dict):
            print(f"    {prefix}{k}: dict")
            _describe(v, prefix + "  ", verbose)
        elif v is None:
            print(f"    {prefix}{k}: None")
        else:
            print(f"    {prefix}{k}: {type(v).__name__}")


def install_trt_backbone(model, trt, use_vision=True, use_text=True, verbose=True):
    """把 backbone 的 forward_image / forward_text 换成 TRT 版本。"""
    bb = model.backbone
    orig_image = bb.forward_image
    orig_text = bb.forward_text
    state = {"image_ref": None}

    def fit_list(proto_list, vals, tag):
        res = []
        for i, p in enumerate(proto_list):
            if i >= len(vals):
                res.append(p)
                continue
            f = _fit(p, vals[i])
            if f is None:
                if verbose:
                    print(f"[TRT] 警告 {tag}[{i}] 形状不匹配 "
                          f"proto{tuple(p.shape)} vs eng{tuple(vals[i].shape)}，"
                          f"回退 PyTorch 值")
                f = p
            res.append(f)
        return res

    def trt_forward_image(samples):
        eng = trt.vision({"image": samples})
        if state["image_ref"] is None:
            state["image_ref"] = orig_image(samples)
            if verbose:
                print("[TRT] forward_image 原型结构：")
                _describe(state["image_ref"], verbose=verbose)
                print("[TRT] engine 输出: " + ", ".join(
                    f"{k}{tuple(eng[k].shape)}" for k in eng))
        proto = state["image_ref"]
        fpn_all = [eng[f"fpn_{i}"] for i in range(3)]
        pos_all = [eng[f"pos_{i}"] for i in range(3)]

        out = dict(proto)
        out["backbone_fpn"] = fit_list(proto["backbone_fpn"], fpn_all, "backbone_fpn")
        out["vision_pos_enc"] = fit_list(proto["vision_pos_enc"], pos_all, "vision_pos_enc")
        out["vision_features"] = out["backbone_fpn"][-1]
        if out.get("sam2_backbone_out") is not None:
            out["sam2_backbone_out"] = None
        return out

    def trt_forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
        lb = bb.language_backbone
        toks = lb.tokenizer(captions, context_length=int(lb.context_length))
        eng = trt.text({"tokens": toks.to(device).long()})
        out = dict(orig_text(captions, input_boxes, additional_text, device))
        pairs = (("language_features", eng["text_memory"]),
                 ("language_mask", eng["text_mask"]),
                 ("language_embeds", eng["text_embeds"]))
        for k, v in pairs:
            if k not in out:
                continue
            f = _fit(out[k], v)
            if f is None:
                if verbose:
                    print(f"[TRT] 警告 {k} 形状不匹配 "
                          f"proto{tuple(out[k].shape)} vs eng{tuple(v.shape)}，"
                          f"回退 PyTorch 值")
            else:
                out[k] = f
        return out

    if use_vision:
        bb.forward_image = trt_forward_image
    if use_text:
        bb.forward_text = trt_forward_text
    if verbose:
        print(f"[TRT] 已接管: vision={use_vision} text={use_text}")


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
    """点提示（像素坐标）——语义同 Sam3Processor.add_geometric_prompt，
    只是把 append_boxes 换成 append_points。点坐标需归一化到 [0,1]。"""
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
    """只保留掩码中包含任一「正样本」提示点的目标（点分割"点谁分谁"）。"""
    masks = state.get("masks")
    if masks is None or masks.numel() == 0:
        return state
    iw, ih = image.size
    pos = [(x, y) for (x, y), lb in zip(points_px, labels) if lb]
    if not pos:
        return state
    keep = []
    for i in range(masks.shape[0]):
        m = masks[i]
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


def _pick_engine(stem: str) -> Path | None:
    """按 SAM3_PREC 精度选择 engine：<stem>.<prec>.engine，回退旧命名 <stem>.engine。"""
    prec = os.environ.get("SAM3_PREC", "tf32")
    cand = ROOT / "models" / f"{stem}.{prec}.engine"
    if cand.is_file():
        return cand
    legacy = ROOT / "models" / f"{stem}.engine"
    if legacy.is_file():
        return legacy
    return None


class TRTEngines:
    def __init__(self):
        self.vision = None
        self.text = None
        vp = _pick_engine("sam3_vision_encoder")
        tp = _pick_engine("sam3_text_encoder")
        if vp is not None:
            self.vision = TRTEngine(vp)
        if tp is not None:
            self.text = TRTEngine(tp)


# ---------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM3 TensorRT 推理（参数同 sam3_infer.py）",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image", required=True, help="输入图像路径")
    p.add_argument("--text", default=None, help="文本提示，如 'dog'")
    p.add_argument("--box", nargs=4, type=float, default=None, metavar=("X", "Y", "W", "H"))
    p.add_argument("--label", type=int, default=1, help="框提示标签：1 正 / 0 负")
    p.add_argument("--point", nargs="+", type=float, default=None, metavar="X Y",
                   help="点提示：像素坐标，成对给出，可多对，如 --point 490 167 384 90")
    p.add_argument("--point-label", nargs="+", type=int, default=None,
                   help="每个点的标签：1 正 / 0 负，数量需与点一致（默认全 1）")
    p.add_argument("--no-box", action="store_true", help="只绘制掩码，不绘制目标框与分数")
    p.add_argument("--keep-all", action="store_true",
                   help="点提示时保留全部候选目标（默认仅保留包含正样本点的掩码）")
    p.add_argument("--out", default=None, help="输出可视化图像")
    p.add_argument("--mask-dir", default=None, help="导出掩码 PNG 目录")
    p.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    p.add_argument("--max-masks", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--bench", action="store_true", help="打印耗时基准")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bpe", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--vision-backend", choices=["engine", "pytorch"], default="engine")
    p.add_argument("--text-backend", choices=["engine", "pytorch"], default="engine")
    return p.parse_args(argv)


def normalize_mask(mask, size):
    w, h = size
    a = np.asarray(mask).astype(np.float32).squeeze()
    if a.ndim != 2:
        a = a.reshape(a.shape[-2], a.shape[-1])
    if a.max() > 1.0:
        a = a / 255.0
    if a.shape != (h, w):
        a = np.array(Image.fromarray((a * 255).astype(np.uint8))
                     .resize((w, h), Image.NEAREST)) / 255.0
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


def build_model(args):
    from sam3 import build_sam3_image_model
    from patch_sam3_rope import apply as apply_rope

    bpe = Path(args.bpe) if args.bpe else ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / "models" / "sam3.pt"
    for f, how in ((bpe, "./download.sh"), (ckpt, "./download.sh")):
        if not f.is_file():
            sys.exit(f"[错误] 缺少文件 {f}，请先执行 {how}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=str(bpe), checkpoint_path=str(ckpt),
        load_from_HF=False, enable_segmentation=True, device=args.device)
    model.eval()
    apply_rope(model)
    print(f"[信息] 模型加载 {time.time() - t0:.1f}s | 设备 {args.device}")
    return model


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.image).is_file():
        sys.exit(f"[错误] 找不到图像: {args.image}")
    if not args.text and args.box is None and args.point is None:
        sys.exit("[错误] 必须提供 --text、--box 或 --point")
    args.point_pairs, args.point_labels_p = parse_points(args)

    image = Image.open(args.image).convert("RGB")
    print(f"[信息] 图像 {args.image}  {image.size[0]}x{image.size[1]}")

    t_build = time.time()
    model = build_model(args)
    t_model = time.time() - t_build

    print("\n[引擎] 加载 TRT engine ...")
    t0 = time.time()
    trt = TRTEngines()
    t_engine = time.time() - t0
    vname = trt.vision.path.name if trt.vision else "未找到"
    tname = trt.text.path.name if trt.text else "未找到"
    print(f"       vision: {vname}")
    print(f"       text  : {tname}")
    print(f"       耗时 {t_engine:.1f}s")

    use_v = args.vision_backend == "engine" and trt.vision is not None
    use_t = args.text_backend == "engine" and trt.text is not None
    install_trt_backbone(model, trt, use_vision=use_v, use_text=use_t)

    from sam3.model.sam3_image_processor import Sam3Processor
    t_p = time.time()
    processor = Sam3Processor(model, confidence_threshold=args.threshold, device=args.device)
    t_proc = time.time() - t_p

    print(f"[计时] 初始化: 导入三方库 {_T_IMPORT:.1f}s | 模型加载 {t_model:.1f}s | "
          f"TRT 引擎 {t_engine:.1f}s | Processor {t_proc:.2f}s | "
          f"初始化总计 {time.time() - _T_LAUNCH:.1f}s")

    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

    def run_once():
        t0 = time.time()
        st = processor.set_image(image)
        sync()
        t_enc = time.time() - t0
        t0 = time.time()
        if args.text:
            st = processor.set_text_prompt(prompt=args.text, state=st)
        elif args.point is not None:
            st = add_point_prompt(processor, image, st,
                                  args.point_pairs, args.point_labels_p)
        else:
            x, y, w, h = args.box
            iw, ih = image.size
            st = processor.add_geometric_prompt(
                state=st, box=[(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih],
                label=bool(args.label))
        sync()
        return st, t_enc, time.time() - t0

    backend = f"vision={args.vision_backend} text={args.text_backend}"
    if args.bench:
        print(f"\n[基准] 提示={args.text or args.box} 重复{args.repeat}次 | {backend}")
        run_once()
        enc, prm, tot = [], [], []
        for i in range(args.repeat):
            t0 = time.time()
            st, e, pr = run_once()
            tot.append((time.time() - t0) * 1000)
            enc.append(e * 1000)
            prm.append(pr * 1000)
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

    state, t_enc, t_prm = run_once()
    print(f"[计时] 首次推理:   图像编码 {t_enc*1000:7.1f}ms | 提示前向 {t_prm*1000:7.1f}ms "
          f"| 合计 {(t_enc + t_prm)*1000:7.1f}ms")
    # 再跑一次：排除首次 CUDA 内核初始化开销，得到稳态耗时
    _, t_enc2, t_prm2 = run_once()
    print(f"[计时] 热身后推理: 图像编码 {t_enc2*1000:7.1f}ms | 提示前向 {t_prm2*1000:7.1f}ms "
          f"| 合计 {(t_enc2 + t_prm2)*1000:7.1f}ms")
    print(f"[计时] 端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s")

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
    print(f"\n[结果] 提示={desc} | 后端 {backend} | 目标数 {len(scores)}")
    for i in range(min(len(scores), args.max_masks)):
        x0, y0, x1, y1 = boxes[i]
        print(f"       #{i}: score={scores[i]:.3f} box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) "
              f"面积={int(masks[i].sum())}px")

    if args.mask_dir:
        d = Path(args.mask_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        for i in range(len(masks)):
            arr = (normalize_mask(masks[i], image.size) * 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{stem}_mask{i}.png")
        print(f"[输出] 掩码 - {d}/")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        overlay(image, masks, boxes, scores, args.alpha, args.max_masks,
                args.point_pairs, args.point_labels_p, draw_box=not args.no_box).save(out)
        print(f"[输出] 可视化 - {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())