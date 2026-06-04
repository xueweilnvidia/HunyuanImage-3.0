"""Fused DownsampleDCAE post-conv path: space-to-depth + channel-mean + residual add.

The vanilla path does:

    h        = self.conv(x)                                      # cuDNN, kept as-is
    h        = pixel_unshuffle_3d(h, (r1, 2, 2))                 # data movement
    shortcut = pixel_unshuffle_3d(x, (r1, 2, 2))                 # data movement
    shortcut = shortcut.reshape(.., out_C, group_size).mean(-1)  # reduction
    return     h + shortcut                                      # elementwise

Each ``pixel_unshuffle_3d`` materialises a permute + reshape; the mean and add
each touch device memory again. This file fuses everything past the conv into
one kernel: one read of x, one read of h, one write of out.

All tensors are channels_last_3d (NDHWC in memory). For a tensor with logical
shape ``(N, C, T, H, W)``, the element at ``[n, c, t, y, x]`` lives at byte
offset ``n*C*T*H*W + t*H*W*C + y*W*C + x*C + c`` — i.e. channels are innermost
and stride-1.

Channel packing convention (matches the einops pattern
``"b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w"``):

    c_out = r1_idx * (r2 * r3 * C_in_h) + r2_idx * (r3 * C_in_h)
          + r3_idx * C_in_h + inner
          = block_idx * C_in_h + inner
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_downsample_dcae_kernel(
    x_ptr, h_ptr, out_ptr,
    T_OUT, H_OUT, W_OUT,
    R1: tl.constexpr,
    FACTOR: tl.constexpr,        # R1 * 2 * 2; 4 or 8
    C_IN_H: tl.constexpr,        # out_C / FACTOR
    GROUP_SIZE: tl.constexpr,    # in_C / C_IN_H
    OUT_C: tl.constexpr,         # FACTOR * C_IN_H (passed explicitly so tl.arange sees a constexpr)
    IN_C: tl.constexpr,          # GROUP_SIZE * C_IN_H
):
    # One program per output voxel (n, t_o, y_o, x_o); the program walks all
    # FACTOR spatial-neighbor blocks and writes their C_IN_H output channels.
    pid = tl.program_id(0).to(tl.int64)

    spatial = T_OUT * H_OUT * W_OUT
    n = pid // spatial
    rem = pid - n * spatial
    t_o = rem // (H_OUT * W_OUT)
    rem = rem - t_o * (H_OUT * W_OUT)
    y_o = rem // W_OUT
    x_o = rem - y_o * W_OUT

    T_IN = T_OUT * R1
    H_IN = H_OUT * 2
    W_IN = W_OUT * 2

    x_n_off = n * IN_C * T_IN * H_IN * W_IN
    h_n_off = n * C_IN_H * T_IN * H_IN * W_IN
    out_n_off = n * OUT_C * T_OUT * H_OUT * W_OUT

    out_voxel_base = (out_n_off
                     + t_o * H_OUT * W_OUT * OUT_C
                     + y_o * W_OUT * OUT_C
                     + x_o * OUT_C)

    inner_off = tl.arange(0, C_IN_H)
    x_chan_off = tl.arange(0, IN_C)
    inv_group = 1.0 / GROUP_SIZE

    for block_idx in tl.static_range(FACTOR):
        r1_idx = block_idx // 4
        r2_idx = (block_idx // 2) % 2
        r3_idx = block_idx % 2

        t_in = t_o * R1 + r1_idx
        y_in = y_o * 2 + r2_idx
        x_in = x_o * 2 + r3_idx

        # h voxel: C_IN_H contiguous channels at (n, *, t_in, y_in, x_in).
        h_voxel = (h_n_off
                   + t_in * H_IN * W_IN * C_IN_H
                   + y_in * W_IN * C_IN_H
                   + x_in * C_IN_H)
        h_val = tl.load(h_ptr + h_voxel + inner_off).to(tl.float32)

        # x voxel: load all IN_C contiguous channels, reduce GROUP_SIZE-at-a-time.
        x_voxel = (x_n_off
                   + t_in * H_IN * W_IN * IN_C
                   + y_in * W_IN * IN_C
                   + x_in * IN_C)
        x_all = tl.load(x_ptr + x_voxel + x_chan_off).to(tl.float32)
        # Channel layout in x is [c0, c1, ..., c_{IN_C-1}] with c = inner*GROUP_SIZE + g,
        # so reshape (C_IN_H, GROUP_SIZE) is row-major and sum(axis=1) gives mean group.
        x_2d = tl.reshape(x_all, (C_IN_H, GROUP_SIZE))
        mean_val = tl.sum(x_2d, axis=1) * inv_group

        result = h_val + mean_val

        tl.store(out_ptr + out_voxel_base + block_idx * C_IN_H + inner_off, result)


def fused_downsample_dcae_post(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
    factor: int,
    add_temporal: bool,
) -> torch.Tensor:
    """Fused space-to-depth (on h and x), channel-mean (on x), residual add.

    Args:
        x: (N, in_C, T, H, W) channels_last_3d. The pre-conv input.
        h: (N, C_in_h, T, H, W) channels_last_3d. The conv output.
        group_size: in_C / C_in_h. Mean-pool window over the s2d-expanded x.
        factor: r1*2*2 = 4 or 8. Determines spatial reduction.
        add_temporal: whether to also downsample T (r1=2 vs r1=1).

    Returns:
        (N, factor * C_in_h, T/r1, H/2, W/2) channels_last_3d.
    """
    assert x.is_cuda and h.is_cuda
    assert x.is_contiguous(memory_format=torch.channels_last_3d), "x must be channels_last_3d"
    assert h.is_contiguous(memory_format=torch.channels_last_3d), "h must be channels_last_3d"
    assert x.dtype == h.dtype
    assert x.dim() == 5 and h.dim() == 5

    N, in_C, T, H, W = x.shape
    _, C_in_h, hT, hH, hW = h.shape
    assert (hT, hH, hW) == (T, H, W), "x and h must share spatial dims"

    r1 = 2 if add_temporal else 1
    assert factor == r1 * 4, f"factor={factor} inconsistent with r1={r1}"
    assert in_C == group_size * C_in_h, (
        f"in_C={in_C} must equal group_size*C_in_h ({group_size}*{C_in_h})"
    )
    assert T % r1 == 0 and H % 2 == 0 and W % 2 == 0

    T_out, H_out, W_out = T // r1, H // 2, W // 2
    out_C = factor * C_in_h

    out = torch.empty(
        (N, out_C, T_out, H_out, W_out),
        device=x.device, dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )

    grid = (N * T_out * H_out * W_out,)
    _fused_downsample_dcae_kernel[grid](
        x, h, out,
        T_out, H_out, W_out,
        R1=r1,
        FACTOR=factor,
        C_IN_H=C_in_h,
        GROUP_SIZE=group_size,
        OUT_C=out_C,
        IN_C=in_C,
        num_warps=4,
    )
    return out


__all__ = ["fused_downsample_dcae_post"]
