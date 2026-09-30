#!/usr/bin/env python3
"""TensorRT 10.x engine 推理封装（性能优化版）。

关键设计：
  * 用 TRT 10 的 **execute_async_v3 + set_tensor_address**，直接拿 torch CUDA
    张量的 data_ptr 当显存地址 —— **不需要 pycuda / cuda-python**。
  * **专用非默认 stream**：TRT 在默认流上会额外调用 cudaStreamSynchronize()
    保证同步，实测拖慢（vision 822ms）；换独立流后消除该开销。
  * **输入/输出 buffer 缓存**：首次调用后复用，避免每次 torch.empty 分配。

用法：
    eng = TRTEngine("models/sam3_vision_encoder.engine")
    print(eng.info())
    out = eng({"image": x})           # x: torch.Tensor (1,3,1008,1008)
    fpn0 = out["fpn_0"]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def _trt_to_torch_dtype(trt_dtype: trt.DataType):
    """TRT dtype -> torch dtype。"""
    if trt_dtype == trt.bool:
        return torch.bool
    if trt_dtype == trt.int32:
        return torch.int32
    if trt_dtype == trt.int64:
        return torch.int64
    return torch.from_numpy(np.empty(0, dtype=trt.nptype(trt_dtype))).dtype


class TRTEngine:
    """封装一个 .engine，提供 ``engine({name: tensor}) -> {name: tensor}``。"""

    def __init__(self, engine_path, device: str = "cuda", verbose: bool = False):
        self.path = Path(engine_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 engine: {self.path}")

        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.device = torch.device(device)

        with open(self.path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败: {self.path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"创建 execution context 失败: {self.path}")

        # 收集 I/O 名称
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # 专用非默认 stream（关键：避免 TRT 在默认流上的强制同步）
        self.stream = torch.cuda.Stream(self.device)

        # buffer 缓存
        self._in_cache = {}
        self._out_cache = {}
        self._calls = 0

    # ------------------------------------------------------------------ #
    def info(self) -> str:
        lines = [f"engine : {self.path.name}"]
        for n in self.input_names:
            lines.append(f"  IN   {n:14s} {tuple(self.engine.get_tensor_shape(n))} "
                         f"{self.engine.get_tensor_dtype(n)}")
        for n in self.output_names:
            lines.append(f"  OUT  {n:14s} {tuple(self.engine.get_tensor_shape(n))} "
                         f"{self.engine.get_tensor_dtype(n)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def __call__(self, feeds: dict) -> dict:
        """执行一次推理。feeds: {输入名: tensor}；返回 {输出名: tensor}。"""
        missing = set(self.input_names) - set(feeds)
        if missing:
            raise KeyError(f"缺少输入 {missing}，需要 {self.input_names}")

        buffers = {}
        shapes_ready = self._calls > 0

        # ---- 输入：copy_ 进缓存 buffer ----
        for name in self.input_names:
            src = feeds[name]
            if not isinstance(src, torch.Tensor):
                src = torch.from_numpy(np.asarray(src))
            if src.dtype == torch.float64:
                src = src.float()
            src = src.to(self.device).contiguous()

            key = ("in", name, tuple(src.shape), str(src.dtype))
            buf = self._in_cache.get(key)
            if buf is None:
                buf = torch.empty_like(src)
                self._in_cache[key] = buf
                shapes_ready = False
            buf.copy_(src, non_blocking=True)
            buffers[name] = buf

            if not shapes_ready:
                try:
                    self.context.set_input_shape(name, tuple(src.shape))
                except Exception:
                    pass
            if not self.context.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"set_tensor_address 失败（输入 {name}）")

        # ---- 输出：复用 buffer ----
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"输出 {name} shape 未定 {shape}，输入 shape 未绑定？")
            key = ("out", name, shape)
            buf = self._out_cache.get(key)
            if buf is None:
                buf = torch.empty(shape, dtype=_trt_to_torch_dtype(
                    self.engine.get_tensor_dtype(name)), device=self.device)
                self._out_cache[key] = buf
            buffers[name] = buf
            if not self.context.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"set_tensor_address 失败（输出 {name}）")

        # ---- 执行（专用流）----
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"execute_async_v3 失败: {self.path.name}")
        self.stream.synchronize()
        self._calls += 1

        return {n: buffers[n] for n in self.output_names}


# ---------------------------------------------------------------------- #
def _selftest():
    """加载两个 engine 跑一遍，验证可用并测速。"""
    root = Path(__file__).resolve().parent.parent
    torch.manual_seed(0)
    print("=" * 70)
    print(" TensorRT engine 自测")
    print("=" * 70)

    vp = root / "models" / "sam3_vision_encoder.engine"
    tp = root / "models" / "sam3_text_encoder.engine"

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    if vp.is_file():
        print(f"\n[1/2] {vp.name}")
        v = TRTEngine(vp)
        print(v.info())
        x = torch.randn(1, 3, 1008, 1008, device="cuda")
        for _ in range(3):
            out = v({"image": x})
        ev0.record()
        for _ in range(10):
            out = v({"image": x})
        ev1.record()
        torch.cuda.synchronize()
        for n in v.output_names:
            o = out[n]
            print(f"    {n:8s} {tuple(o.shape)} mean={o.float().mean().item():+.4f}")
        print(f"    >>> vision 平均 {ev0.elapsed_time(ev1)/10:.1f} ms")

    if tp.is_file():
        print(f"\n[2/2] {tp.name}")
        te = TRTEngine(tp)
        print(te.info())
        tokens = torch.randint(1, 40000, (1, 32), device="cuda", dtype=torch.int64)
        for _ in range(3):
            tout = te({"tokens": tokens})
        ev0.record()
        for _ in range(20):
            tout = te({"tokens": tokens})
        ev1.record()
        torch.cuda.synchronize()
        for n in te.output_names:
            o = tout[n]
            print(f"    {n:12s} {tuple(o.shape)} dtype={o.dtype}")
        print(f"    >>> text 平均 {ev0.elapsed_time(ev1)/20:.2f} ms")

    print("\n" + "=" * 70)
    print(" 自测完成 ✅")
    print("=" * 70)


if __name__ == "__main__":
    _selftest()