"""
patch_sam3_rope.py  --  让 SAM3 的 ViT 主干可以被导出为 ONNX（FP32）

问题：
    SAM3 的 2D-RoPE 用「复数」实现：
        torch.polar(...)  ->  freqs_cis (complex64)
        torch.view_as_complex / torch.view_as_real
    ONNX 的 Tensor 类型系统里**没有复数**，任何 opset 都不支持
    aten::view_as_complex，所以导出必然失败：
        UnsupportedOperatorError: 'aten::view_as_complex' is not supported

方案：
    复数乘法是可以用实数等价表达的。设
        x   = (a, b)          （view_as_complex 前的最后一维 pair）
        w   = (c, s)          （即 cos + i*sin）
    则  x * w = (a*c - b*s) + i*(a*s + b*c)

    于是：
      1) 把每个 Attention 里的 freqs_cis 由 complex64 换成等价的
         real 张量 view_as_real(freqs_cis)，形状 [..., D/2, 2]
      2) 用实数版 apply_rotary_enc 替换原实现

    数值完全等价（数学上恒等，浮点误差在 1 ulp 量级）。

用法：
    from patch_sam3_rope import apply
    apply(model)        # 传入已经 build 好的 SAM3 模型
"""

import torch
import sam3.model.vitdet as vitdet


def _reshape_for_broadcast_real(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x 形如 (..., L, D/2, 2)；freqs_cis 形如 (L, D/2, 2) -> 广播成 (1,...,1,L,D/2,2)"""
    ndim = x.ndim
    shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def _apply_rotary_enc_real(xq, xk, freqs_cis, repeat_freqs_k: bool = False):
    """与 sam3.model.vitdet.apply_rotary_enc 数值等价的纯实数实现。

    入参 xq/xk: (..., L, D) 实数；freqs_cis: (L, D/2, 2) 实数
    返回: (xq_out, xk_out) 均为 (..., L, D)
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)          # (..., L, D/2, 2)
    xk_ = None if xk is None else xk.float().reshape(*xk.shape[:-1], -1, 2)

    f_real = _reshape_for_broadcast_real(freqs_cis, xq_)     # (1,...,1,L,D/2,2)
    w_r, w_i = f_real[..., 0], f_real[..., 1]

    # 复数乘法 (a+bi)(c+si) = (ac-bs) + (as+bc)i
    def _rot(x, c, s):
        xr, xi = x[..., 0], x[..., 1]
        return torch.stack((xr * c - xi * s, xr * s + xi * c), dim=-1).flatten(3)

    xq_out = _rot(xq_, w_r, w_i)

    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), None

    if repeat_freqs_k:
        # xk 的序列长度是 xq 的 r 倍：在 L 维（倒数第 2 维）上重复 r 次
        r = xk_.shape[-2] // xq_.shape[-2]
        if r != 1:
            w_r = w_r.repeat(*([1] * (w_r.dim() - 2)), r, 1)
            w_i = w_i.repeat(*([1] * (w_i.dim() - 2)), r, 1)

    xk_out = _rot(xk_, w_r, w_i)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def _to_real_freqs(freqs_cis: torch.Tensor) -> torch.Tensor:
    """complex64/128 -> real float32/float64，形状 [..., D/2, 2]"""
    return torch.view_as_real(freqs_cis).clone()


def apply(model) -> int:
    """就地打补丁；返回被处理的 Attention 数量。"""
    # 1) freqs_cis: complex -> real
    n = 0
    for m in model.modules():
        fc = getattr(m, "freqs_cis", None)
        if fc is not None and torch.is_complex(fc):
            real = _to_real_freqs(fc)
            # 删除旧 buffer，注册同名的 real buffer
            if "freqs_cis" in getattr(m, "_buffers", {}):
                del m._buffers["freqs_cis"]
            m.register_buffer("freqs_cis", real)
            n += 1

    # 2) 替换 rotary 内核（进程级，一次即可）
    vitdet.apply_rotary_enc = _apply_rotary_enc_real
    return n