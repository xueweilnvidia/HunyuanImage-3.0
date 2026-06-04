"""Fused UpsampleDCAE post-conv path: depth-to-space + channel-repeat residual add.

The vanilla path does:

    h        = self.conv(x)                                                   # cuDNN, kept as-is
    h        = pixel_shuffle_3d(h, (r1, 2, 2))                                # data movement
    shortcut = x.repeat_interleave(repeats=repeats, dim=1)                    # data expansion
    shortcut = pixel_shuffle_3d(shortcut, (r1, 2, 2))                         # data movement
    return     h + shortcut                                                   # elementwise

The two ``pixel_shuffle`` calls each materialise a permute + reshape, the
``repeat_interleave`` materialises a tensor ``repeats``x larger than x, and the
add reads/writes again. This kernel fuses everything past the conv: one read of
x, one read of h, one write of out.

All tensors are channels_last_3d (NDHWC in memory). For a tensor with logical
shape ``(N, C, T, H, W)``, the element at ``[n, c, t, y, x]`` lives at byte
offset ``n*C*T*H*W + t*H*W*C + y*W*C + x*C + c`` — channels are innermost and
stride-1.

Channel packing convention (inverse of DownsampleDCAE; matches the einops
pattern ``"b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)"``):

    c_packed = r1_idx * (r2 * r3 * C_OUT) + r2_idx * (r3 * C_OUT)
             + r3_idx * C_OUT + c_out
             = block_idx * C_OUT + c_out

For h, ``c_packed`` indexes directly into the conv output. For the residual,
the same ``c_packed`` indexes the repeat-interleaved tensor whose channel
``j`` holds ``x[..., j // REPEATS, ...]`` — so the source x channel is
``(block_idx * C_OUT + c_out) // REPEATS``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_upsample_dcae_kernel(
    x_ptr, h_ptr, out_ptr,
    T_IN, H_IN, W_IN,
    R1: tl.constexpr,
    FACTOR: tl.constexpr,        # R1 * 2 * 2; 4 or 8
    C_OUT: tl.constexpr,         # output channels (per block)
    REPEATS: tl.constexpr,       # FACTOR * C_OUT / IN_C
    IN_C: tl.constexpr,          # source x channels
    H_C: tl.constexpr,           # FACTOR * C_OUT (channels of h)
    PER_BLOCK_X: tl.constexpr,   # C_OUT // REPEATS == IN_C // FACTOR
):
    # One program per input voxel (n, t_in, y_in, x_in); the program walks all
    # FACTOR sub-voxel offsets and writes their C_OUT output channels.
    pid = tl.program_id(0).to(tl.int64)

    spatial = T_IN * H_IN * W_IN
    n = pid // spatial
    rem = pid - n * spatial
    t_in = rem // (H_IN * W_IN)
    rem = rem - t_in * (H_IN * W_IN)
    y_in = rem // W_IN
    x_in = rem - y_in * W_IN

    T_OUT = T_IN * R1
    H_OUT = H_IN * 2
    W_OUT = W_IN * 2

    x_n_off = n * IN_C * T_IN * H_IN * W_IN
    h_n_off = n * H_C * T_IN * H_IN * W_IN
    out_n_off = n * C_OUT * T_OUT * H_OUT * W_OUT

    # Base offsets into the single input voxel shared across all FACTOR blocks.
    x_voxel_base = (x_n_off
                    + t_in * H_IN * W_IN * IN_C
                    + y_in * W_IN * IN_C
                    + x_in * IN_C)
    h_voxel_base = (h_n_off
                    + t_in * H_IN * W_IN * H_C
                    + y_in * W_IN * H_C
                    + x_in * H_C)

    c_off = tl.arange(0, C_OUT)
    # c_off // REPEATS produces the per-block source x channel offset; broadcast
    # pattern, so each unique address is loaded once from L1.
    c_off_div_r = c_off // REPEATS

    for block_idx in tl.static_range(FACTOR):
        r1_idx = block_idx // 4
        r2_idx = (block_idx // 2) % 2
        r3_idx = block_idx % 2

        t_out = t_in * R1 + r1_idx
        y_out = y_in * 2 + r2_idx
        x_out = x_in * 2 + r3_idx

        h_val = tl.load(h_ptr + h_voxel_base + block_idx * C_OUT + c_off)

        # Gather from x: each c_out maps to x channel
        # (block_idx * C_OUT + c_out) // REPEATS == block_idx * PER_BLOCK_X + c_off // REPEATS
        x_chan_off = block_idx * PER_BLOCK_X + c_off_div_r
        x_val = tl.load(x_ptr + x_voxel_base + x_chan_off)

        result = h_val + x_val

        out_voxel = (out_n_off
                     + t_out * H_OUT * W_OUT * C_OUT
                     + y_out * W_OUT * C_OUT
                     + x_out * C_OUT)
        tl.store(out_ptr + out_voxel + c_off, result)


def fused_upsample_dcae_post(
    x: torch.Tensor,
    h: torch.Tensor,
    repeats: int,
    factor: int,
    add_temporal: bool,
) -> torch.Tensor:
    """Fused depth-to-space (on h and on repeat-interleaved x), residual add.

    Args:
        x: (N, in_C, T, H, W) channels_last_3d. The pre-conv input to UpsampleDCAE.
        h: (N, out_C * factor, T, H, W) channels_last_3d. The conv output.
        repeats: factor * out_C / in_C. The per-channel repeat count baked into the residual.
        factor: r1*2*2 = 4 or 8. Determines spatial expansion.
        add_temporal: whether to also upsample T (r1=2 vs r1=1).

    Returns:
        (N, out_C, T*r1, H*2, W*2) channels_last_3d.
    """
    assert x.is_cuda and h.is_cuda
    assert x.is_contiguous(memory_format=torch.channels_last_3d), "x must be channels_last_3d"
    assert h.is_contiguous(memory_format=torch.channels_last_3d), "h must be channels_last_3d"
    assert x.dtype == h.dtype
    assert x.dim() == 5 and h.dim() == 5

    N, in_C, T, H, W = x.shape
    _, h_C, hT, hH, hW = h.shape
    assert (hT, hH, hW) == (T, H, W), "x and h must share spatial dims"

    r1 = 2 if add_temporal else 1
    assert factor == r1 * 4, f"factor={factor} inconsistent with r1={r1}"
    assert h_C % factor == 0, f"h channels {h_C} must be divisible by factor {factor}"
    out_C = h_C // factor
    assert in_C * repeats == h_C, (
        f"in_C * repeats ({in_C}*{repeats}) must equal h channels ({h_C})"
    )
    assert out_C % repeats == 0, (
        f"out_C={out_C} must be divisible by repeats={repeats}"
    )

    T_out, H_out, W_out = T * r1, H * 2, W * 2

    out = torch.empty(
        (N, out_C, T_out, H_out, W_out),
        device=x.device, dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )

    grid = (N * T * H * W,)
    _fused_upsample_dcae_kernel[grid](
        x, h, out,
        T, H, W,
        R1=r1,
        FACTOR=factor,
        C_OUT=out_C,
        REPEATS=repeats,
        IN_C=in_C,
        H_C=h_C,
        PER_BLOCK_X=out_C // repeats,
        num_warps=4,
    )
    return out


__all__ = ["fused_upsample_dcae_post"]
