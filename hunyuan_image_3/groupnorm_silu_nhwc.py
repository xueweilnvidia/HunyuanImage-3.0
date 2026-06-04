"""Fused GroupNorm + SiLU for NHWC (channels-last) activations.

Strict NHWC contract: the caller must hand the input in
``torch.channels_last_3d`` (5-D) or ``torch.channels_last`` (4-D) memory
format. Non-channels-last input raises ``ValueError`` — we don't silently
re-lay out, because that would mask a layout bug in the surrounding model
(e.g. a conv that wasn't converted to channels-last).

Numerics target ``hunyuan_image_3.group_norm_silu.apply_group_norm_silu``
within the tolerances in ``tests/test_correctness.py``.

Layout, briefly. For ``x.shape == (N, C, T, H, W)`` in ``channels_last_3d``,
memory order is ``N T H W C`` with strides ``(C*T*H*W, 1, H*W*C, W*C, C)``.
For a fixed sample ``n`` and group ``g``, the group's ``C/G`` channels are
contiguous and live at offset ``n*C*T*H*W + s*C + g*(C/G) + c`` for spatial
index ``s in [0, T*H*W)`` and ``c in [0, C/G)``. A tile of shape
``[BLOCK_SPATIAL, C_PER_GROUP]`` therefore loads contiguous channels at
stride ``C`` along the spatial axis.

Three implementations are available; the path and its launch params are
chosen by **runtime autotune** on first use of a new ``(dtype, shape,
num_groups)`` triple, then cached. Caller pays a one-time tune cost
(~hundreds of ms, dominated by Triton kernel compilation) per unseen
shape; subsequent calls are an O(1) cache lookup.

The three implementations:

* **single-pass** (one kernel): one program per ``(n, group)`` walks the
  group's spatial range in chunks of ``BLOCK_SPATIAL``. Two passes over
  the same memory in registers — accumulate stats, then apply. Wins
  when launch overhead dominates (small-to-medium ``group_size``).
* **two-pass** (two kernels): stats kernel parallelises the reduction
  across ``(n, group, spatial_chunk)``; apply kernel finalises stats
  inline and applies. Wins when 32 programs would underutilise the GPU
  but cacheline waste isn't the bottleneck.
* **multi-group two-pass** (three kernels: stats, finalise, apply): each
  stats/apply program owns ``GROUPS_PER_TILE`` adjacent groups so each
  spatial-step NHWC load spans a full cacheline. Wins for the bandwidth-
  bound regime with small ``c_per_group``, where a single-group NHWC
  load reads ``c_per_group`` bytes and then jumps ``channels`` bytes.

Autotune is the only dispatch path. Set
``GROUPNORM_NHWC_AUTOTUNE_VERBOSE=1`` to print the tune trace on first
call for each shape.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

import triton
import triton.language as tl


_AUTOTUNE_VERBOSE = os.environ.get("GROUPNORM_NHWC_AUTOTUNE_VERBOSE", "0") != "0"

# The c_per_group values supported by the templated kernels. Anything else
# falls back to the PyTorch reference. Single source of truth — used by
# both the public dispatcher (to decide whether to tune at all) and the
# config enumerator (to choose the BLOCK_SPATIAL range).
_SUPPORTED_C_PER_GROUP = (4, 8, 16, 32)


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ============================================================================
# Single-pass kernel (one program per (n, group))
# ============================================================================

@triton.jit
def _gn_silu_nhwc_single_pass_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_stride,
    channels,
    spatial_size,
    num_groups,
    eps,
    inv_group_size,
    BLOCK_SPATIAL: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)

    n = row // num_groups
    g = row - n * num_groups
    base = n * n_stride + g * C_PER_GROUP

    channel_off = tl.arange(0, C_PER_GROUP)
    spatial_arange = tl.arange(0, BLOCK_SPATIAL)
    num_chunks = tl.cdiv(spatial_size, BLOCK_SPATIAL)

    # ---------- pass 1: reduce ----------
    s_acc = tl.zeros((), dtype=tl.float32)
    sq_acc = tl.zeros((), dtype=tl.float32)
    for c in range(0, num_chunks):
        spatial_off = c * BLOCK_SPATIAL + spatial_arange
        mask = spatial_off < spatial_size
        ptrs = input_ptr + base + spatial_off[:, None].to(tl.int64) * channels + channel_off[None, :]
        x = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        s_acc += tl.sum(x)
        sq_acc += tl.sum(x * x)

    mean = s_acc * inv_group_size
    # Match the baseline's E[X^2] - E[X]^2 form rather than Welford so the
    # numerical paths line up at fp32 precision.
    var = sq_acc * inv_group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + g * C_PER_GROUP + channel_off).to(tl.float32)
    b = tl.load(bias_ptr + g * C_PER_GROUP + channel_off).to(tl.float32)

    # ---------- pass 2: apply ----------
    for c in range(0, num_chunks):
        spatial_off = c * BLOCK_SPATIAL + spatial_arange
        mask = spatial_off < spatial_size
        ptrs = input_ptr + base + spatial_off[:, None].to(tl.int64) * channels + channel_off[None, :]
        x = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * w[None, :] + b[None, :]
        y = y * tl.sigmoid(y)
        out_ptrs = output_ptr + base + spatial_off[:, None].to(tl.int64) * channels + channel_off[None, :]
        tl.store(out_ptrs, y, mask=mask[:, None])


# ============================================================================
# Two-pass: stats kernel writes partial sums; apply kernel finalizes inline.
# ============================================================================

@triton.jit
def _gn_silu_nhwc_stats_kernel(
    input_ptr,
    partial_sum_ptr,
    partial_sq_ptr,
    n_stride,
    channels,
    spatial_size,
    chunks_per_row,
    num_groups,
    BLOCK_SPATIAL: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)

    n = row // num_groups
    g = row - n * num_groups
    base = n * n_stride + g * C_PER_GROUP

    spatial_off = chunk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    channel_off = tl.arange(0, C_PER_GROUP)
    mask = spatial_off < spatial_size

    ptrs = input_ptr + base + spatial_off[:, None].to(tl.int64) * channels + channel_off[None, :]
    x = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

    s = tl.sum(x)
    sq = tl.sum(x * x)

    idx = row * chunks_per_row + chunk
    tl.store(partial_sum_ptr + idx, s)
    tl.store(partial_sq_ptr + idx, sq)


@triton.jit
def _gn_silu_nhwc_apply_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_ptr,
    partial_sq_ptr,
    output_ptr,
    n_stride,
    channels,
    spatial_size,
    chunks_per_row,
    num_groups,
    eps,
    inv_group_size,
    BLOCK_SPATIAL: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)

    n = row // num_groups
    g = row - n * num_groups
    base = n * n_stride + g * C_PER_GROUP

    # Inline finalize: sum partials for this row to get mean / rstd. Saves a
    # whole kernel launch versus a dedicated finalize. ``BLOCK_CHUNKS`` is the
    # next power of 2 ≥ ``chunks_per_row`` so this is one masked load + one
    # tree reduction per program — cheap relative to the spatial load below.
    chunk_arange = tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_arange < chunks_per_row
    s_partials = tl.load(partial_sum_ptr + row * chunks_per_row + chunk_arange, mask=chunk_mask, other=0.0)
    sq_partials = tl.load(partial_sq_ptr + row * chunks_per_row + chunk_arange, mask=chunk_mask, other=0.0)
    s_total = tl.sum(s_partials)
    sq_total = tl.sum(sq_partials)

    mean = s_total * inv_group_size
    var = sq_total * inv_group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    channel_off = tl.arange(0, C_PER_GROUP)
    spatial_off = chunk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    mask = spatial_off < spatial_size

    w = tl.load(weight_ptr + g * C_PER_GROUP + channel_off).to(tl.float32)
    b = tl.load(bias_ptr + g * C_PER_GROUP + channel_off).to(tl.float32)

    ptrs = input_ptr + base + spatial_off[:, None].to(tl.int64) * channels + channel_off[None, :]
    x = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
    y = (x - mean) * rstd
    y = y * w[None, :] + b[None, :]
    y = y * tl.sigmoid(y)
    out_ptrs = output_ptr + base + spatial_off[:, None].to(tl.int64) * channels + channel_off[None, :]
    tl.store(out_ptrs, y, mask=mask[:, None])


# ============================================================================
# Multi-group two-pass: stats over a tile of GROUPS_PER_TILE adjacent groups,
# dedicated finalize, then apply over the same tile.
# Each spatial step in the load tile spans GROUPS_PER_TILE * C_PER_GROUP
# contiguous channels — full cacheline utilisation even at C_PER_GROUP=4.
# ============================================================================

@triton.jit
def _gn_silu_nhwc_multi_stats_kernel(
    input_ptr,
    partial_sum_ptr,
    partial_sq_ptr,
    n_stride,
    channels,
    spatial_size,
    chunks_per_row,
    num_groups,
    BLOCK_SPATIAL: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)

    num_tiles_per_n = num_groups // GROUPS_PER_TILE
    n = pid // num_tiles_per_n
    gt = pid - n * num_tiles_per_n
    base = n * n_stride + gt * GROUPS_PER_TILE * C_PER_GROUP

    spatial_off = chunk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    g_off = tl.arange(0, GROUPS_PER_TILE)
    c_off = tl.arange(0, C_PER_GROUP)
    mask = spatial_off < spatial_size

    ptrs = (input_ptr + base
            + spatial_off[:, None, None].to(tl.int64) * channels
            + g_off[None, :, None] * C_PER_GROUP
            + c_off[None, None, :])
    x = tl.load(ptrs, mask=mask[:, None, None], other=0.0).to(tl.float32)

    # Per-group sums: reduce over spatial (axis 0) and channel (axis 2).
    s = tl.sum(tl.sum(x, axis=2), axis=0)
    sq = tl.sum(tl.sum(x * x, axis=2), axis=0)

    out_idx = (n * num_groups + gt * GROUPS_PER_TILE + g_off) * chunks_per_row + chunk
    tl.store(partial_sum_ptr + out_idx, s)
    tl.store(partial_sq_ptr + out_idx, sq)


@triton.jit
def _gn_silu_nhwc_multi_finalize_kernel(
    partial_sum_ptr,
    partial_sq_ptr,
    mean_ptr,
    rstd_ptr,
    chunks_per_row,
    eps,
    inv_group_size,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk_off = tl.arange(0, BLOCK_CHUNKS)

    s_acc = tl.zeros((), dtype=tl.float32)
    sq_acc = tl.zeros((), dtype=tl.float32)
    # Loop in BLOCK_CHUNKS-sized strides so we don't have to size the tile to
    # the largest chunks_per_row across all shapes.
    for c_start in range(0, chunks_per_row, BLOCK_CHUNKS):
        c_idx = c_start + chunk_off
        mask = c_idx < chunks_per_row
        idx = row * chunks_per_row + c_idx
        s = tl.load(partial_sum_ptr + idx, mask=mask, other=0.0)
        sq = tl.load(partial_sq_ptr + idx, mask=mask, other=0.0)
        s_acc += tl.sum(s)
        sq_acc += tl.sum(sq)

    mean = s_acc * inv_group_size
    var = sq_acc * inv_group_size - mean * mean
    rstd = tl.rsqrt(var + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _gn_silu_nhwc_multi_apply_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_ptr,
    n_stride,
    channels,
    spatial_size,
    num_groups,
    BLOCK_SPATIAL: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)

    num_tiles_per_n = num_groups // GROUPS_PER_TILE
    n = pid // num_tiles_per_n
    gt = pid - n * num_tiles_per_n
    base = n * n_stride + gt * GROUPS_PER_TILE * C_PER_GROUP

    spatial_off = chunk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    g_off = tl.arange(0, GROUPS_PER_TILE)
    c_off = tl.arange(0, C_PER_GROUP)
    mask = spatial_off < spatial_size

    stat_idx = n * num_groups + gt * GROUPS_PER_TILE + g_off
    mean = tl.load(mean_ptr + stat_idx)
    rstd = tl.load(rstd_ptr + stat_idx)

    # weight/bias are indexed by absolute channel = gt*GROUPS_PER_TILE*C_PER_GROUP
    # + g_in_tile*C_PER_GROUP + c.
    chan_idx = (gt * GROUPS_PER_TILE * C_PER_GROUP
                + g_off[:, None] * C_PER_GROUP
                + c_off[None, :])
    w = tl.load(weight_ptr + chan_idx).to(tl.float32)
    b = tl.load(bias_ptr + chan_idx).to(tl.float32)

    ptrs = (input_ptr + base
            + spatial_off[:, None, None].to(tl.int64) * channels
            + g_off[None, :, None] * C_PER_GROUP
            + c_off[None, None, :])
    x = tl.load(ptrs, mask=mask[:, None, None], other=0.0).to(tl.float32)

    y = (x - mean[None, :, None]) * rstd[None, :, None]
    y = y * w[None, :, :] + b[None, :, :]
    y = y * tl.sigmoid(y)

    out_ptrs = (output_ptr + base
                + spatial_off[:, None, None].to(tl.int64) * channels
                + g_off[None, :, None] * C_PER_GROUP
                + c_off[None, None, :])
    tl.store(out_ptrs, y, mask=mask[:, None, None])


# ============================================================================
# Python dispatch
# ============================================================================

def _expected_memory_format(ndim: int) -> torch.memory_format:
    if ndim == 5:
        return torch.channels_last_3d
    if ndim == 4:
        return torch.channels_last
    raise ValueError(f"only 4-D / 5-D inputs supported, got ndim={ndim}")


def _run_single_pass(x, weight, bias, num_groups, c_per_group, eps, params):
    BLOCK_SPATIAL = params["BLOCK_SPATIAL"]
    num_warps = params["num_warps"]
    num_stages = params["num_stages"]

    N = x.shape[0]
    C = x.shape[1]
    spatial_size = math.prod(x.shape[2:]) if x.ndim > 2 else 1
    group_size = c_per_group * spatial_size
    rows = N * num_groups
    n_stride = x.stride(0)

    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _gn_silu_nhwc_single_pass_kernel[(rows,)](
            x, weight, bias, y,
            n_stride, C, spatial_size, num_groups,
            eps, 1.0 / group_size,
            BLOCK_SPATIAL=BLOCK_SPATIAL, C_PER_GROUP=c_per_group,
            num_warps=num_warps, num_stages=num_stages,
        )
    return y


def _run_two_pass(x, weight, bias, num_groups, c_per_group, eps, params):
    BLOCK_SPATIAL = params["BLOCK_SPATIAL"]
    num_warps = params["num_warps"]
    num_stages = params["num_stages"]

    N = x.shape[0]
    C = x.shape[1]
    spatial_size = math.prod(x.shape[2:]) if x.ndim > 2 else 1
    group_size = c_per_group * spatial_size
    rows = N * num_groups
    chunks_per_row = triton.cdiv(spatial_size, BLOCK_SPATIAL)
    n_stride = x.stride(0)

    y = torch.empty_like(x)
    partial_sum = torch.empty((rows, chunks_per_row), device=x.device, dtype=torch.float32)
    partial_sq = torch.empty_like(partial_sum)

    BLOCK_CHUNKS = max(1, triton.next_power_of_2(chunks_per_row))

    with torch.cuda.device(x.device):
        _gn_silu_nhwc_stats_kernel[(rows, chunks_per_row)](
            x, partial_sum, partial_sq,
            n_stride, C, spatial_size, chunks_per_row, num_groups,
            BLOCK_SPATIAL=BLOCK_SPATIAL, C_PER_GROUP=c_per_group,
            num_warps=num_warps, num_stages=num_stages,
        )

        _gn_silu_nhwc_apply_kernel[(rows, chunks_per_row)](
            x, weight, bias, partial_sum, partial_sq, y,
            n_stride, C, spatial_size, chunks_per_row, num_groups,
            eps, 1.0 / group_size,
            BLOCK_SPATIAL=BLOCK_SPATIAL, C_PER_GROUP=c_per_group,
            BLOCK_CHUNKS=BLOCK_CHUNKS,
            num_warps=num_warps, num_stages=num_stages,
        )
    return y


def _run_multi_group(x, weight, bias, num_groups, c_per_group, eps, params):
    BLOCK_SPATIAL = params["BLOCK_SPATIAL"]
    GROUPS_PER_TILE = params["GROUPS_PER_TILE"]
    num_warps = params["num_warps"]
    num_stages = params["num_stages"]

    N = x.shape[0]
    C = x.shape[1]
    spatial_size = math.prod(x.shape[2:]) if x.ndim > 2 else 1
    group_size = c_per_group * spatial_size
    rows = N * num_groups
    num_tiles_per_n = num_groups // GROUPS_PER_TILE
    chunks_per_row = triton.cdiv(spatial_size, BLOCK_SPATIAL)
    n_stride = x.stride(0)

    y = torch.empty_like(x)
    # rows × chunks_per_row partial buffers (fp32). For (1, 128, 4, 512, 512)
    # at BLOCK_SPATIAL=256 this is 32×4096 = 0.5 MiB per buffer — small vs.
    # the 256 MiB activation, but allocated per call.
    partial_sum = torch.empty((rows, chunks_per_row), device=x.device, dtype=torch.float32)
    partial_sq = torch.empty_like(partial_sum)
    mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)

    # The finalize kernel loops over its row in tiles of BLOCK_CHUNKS. Cap it
    # so we don't blow registers / shared memory; 1024 is enough that even at
    # chunks_per_row=8192 we only loop 8 times.
    BLOCK_CHUNKS = min(1024, max(64, triton.next_power_of_2(chunks_per_row)))

    with torch.cuda.device(x.device):
        _gn_silu_nhwc_multi_stats_kernel[(N * num_tiles_per_n, chunks_per_row)](
            x, partial_sum, partial_sq,
            n_stride, C, spatial_size, chunks_per_row, num_groups,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            GROUPS_PER_TILE=GROUPS_PER_TILE,
            C_PER_GROUP=c_per_group,
            num_warps=num_warps, num_stages=num_stages,
        )
        _gn_silu_nhwc_multi_finalize_kernel[(rows,)](
            partial_sum, partial_sq, mean, rstd,
            chunks_per_row, eps, 1.0 / group_size,
            BLOCK_CHUNKS=BLOCK_CHUNKS,
            num_warps=4, num_stages=2,
        )
        _gn_silu_nhwc_multi_apply_kernel[(N * num_tiles_per_n, chunks_per_row)](
            x, weight, bias, mean, rstd, y,
            n_stride, C, spatial_size, num_groups,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            GROUPS_PER_TILE=GROUPS_PER_TILE,
            C_PER_GROUP=c_per_group,
            num_warps=num_warps, num_stages=num_stages,
        )
    return y


# ============================================================================
# Runtime autotune
# ============================================================================

# Maps (dtype, shape, num_groups) -> (path_name, params_dict). Populated
# lazily on first call; subsequent calls do an O(1) lookup.
_TUNE_CACHE: dict[tuple, tuple[str, dict]] = {}

_RUN_FNS = {
    "single": _run_single_pass,
    "two": _run_two_pass,
    "multi": _run_multi_group,
}


def _tune_key(x: torch.Tensor, num_groups: int) -> tuple:
    return (x.dtype, x.ndim, tuple(x.shape), num_groups)


def _enumerate_configs(c_per_group: int, num_groups: int, group_size: int) -> list:
    """Candidate (path, params) pairs for a given shape.

    The ranges are deliberately narrow: each path's BLOCK_SPATIAL is chosen so
    the per-program tile sits in ~16-64 KiB of registers across the c_per_group
    table — wide enough to capture the optimum, narrow enough that compile
    time on first call stays in the low seconds.
    """
    if c_per_group == 4:
        sp_bs_list = (1024, 2048)
    elif c_per_group == 8:
        sp_bs_list = (512, 1024, 2048)
    elif c_per_group == 16:
        sp_bs_list = (256, 512, 1024)
    elif c_per_group == 32:
        sp_bs_list = (128, 256, 512)
    else:
        return []

    configs: list[tuple[str, dict]] = []

    for bs in sp_bs_list:
        for nw in (4, 8):
            configs.append(("single", dict(BLOCK_SPATIAL=bs, num_warps=nw, num_stages=3)))

    # Two-pass: parallelism across spatial chunks only pays off once a single
    # (n, group) program would do non-trivial work. ``group_size >= 1<<16``
    # rules out the smallest VAE shapes where launch overhead dominates.
    if group_size >= 1 << 16:
        for bs in sp_bs_list:
            for nw in (4, 8):
                configs.append(("two", dict(BLOCK_SPATIAL=bs, num_warps=nw, num_stages=3)))

    # Multi-group: only beneficial when c_per_group is so small that a
    # single-group NHWC load wastes most of a cacheline. Also requires
    # num_groups divisible by GROUPS_PER_TILE.
    if c_per_group in (4, 8):
        for bs in (128, 256, 512):
            for gpt in (4, 8, 16):
                if num_groups % gpt != 0:
                    continue
                if gpt * c_per_group > 64:  # >2 cachelines per spatial step, no extra benefit
                    continue
                if bs * gpt * c_per_group > 8192:  # cap tile elements to keep shmem reasonable
                    continue
                for nw in (4, 8):
                    configs.append(("multi", dict(
                        BLOCK_SPATIAL=bs, GROUPS_PER_TILE=gpt,
                        num_warps=nw, num_stages=2,
                    )))

    return configs


def _time_config(path: str, params: dict, x, weight, bias, num_groups, c_per_group, eps,
                 warmup: int = 3, iters: int = 10) -> float:
    fn = _RUN_FNS[path]
    for _ in range(warmup):
        fn(x, weight, bias, num_groups, c_per_group, eps, params)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x, weight, bias, num_groups, c_per_group, eps, params)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _tune_shape(x, weight, bias, num_groups, c_per_group, eps) -> tuple[str, dict] | None:
    spatial_size = math.prod(x.shape[2:]) if x.ndim > 2 else 1
    group_size = c_per_group * spatial_size
    configs = _enumerate_configs(c_per_group, num_groups, group_size)
    if not configs:
        return None

    best_path: str | None = None
    best_params: dict | None = None
    best_ms = float("inf")
    if _AUTOTUNE_VERBOSE:
        print(f"[tune] shape={tuple(x.shape)} dtype={x.dtype} "
              f"c_per_group={c_per_group} group_size={group_size} "
              f"-> {len(configs)} configs")
    for path, params in configs:
        try:
            ms = _time_config(path, params, x, weight, bias, num_groups, c_per_group, eps)
        except Exception as exc:  # OutOfResources, compile failure, etc.
            if _AUTOTUNE_VERBOSE:
                print(f"[tune]   skip {path} {params}: {type(exc).__name__}")
            continue
        if _AUTOTUNE_VERBOSE:
            print(f"[tune]   {path:6} {params} -> {ms:.4f} ms")
        if ms < best_ms:
            best_ms = ms
            best_path = path
            best_params = params

    if best_path is None:
        return None
    if _AUTOTUNE_VERBOSE:
        print(f"[tune]   best: {best_path} {best_params} @ {best_ms:.4f} ms")
    return best_path, best_params


def clear_autotune_cache() -> None:
    """Drop all cached tune decisions. Useful for repeatable benchmarks."""
    _TUNE_CACHE.clear()


def autotune_cache_snapshot() -> dict:
    """Return a copy of the current tune cache for inspection."""
    return dict(_TUNE_CACHE)


def apply_group_norm_silu_nhwc(x: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
    if not isinstance(norm, nn.GroupNorm):
        raise TypeError(f"expected nn.GroupNorm, got {type(norm).__name__}")
    if not norm.affine:
        raise ValueError("apply_group_norm_silu_nhwc requires affine=True GroupNorm")
    if not x.is_cuda:
        raise ValueError("apply_group_norm_silu_nhwc requires a CUDA tensor")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype {x.dtype}")

    expected_mf = _expected_memory_format(x.ndim)
    if not x.is_contiguous(memory_format=expected_mf):
        raise ValueError(
            f"apply_group_norm_silu_nhwc requires {expected_mf} memory format; "
            f"got strides={x.stride()} for shape={tuple(x.shape)}"
        )

    C = x.shape[1]
    G = norm.num_groups
    if C % G != 0:
        raise ValueError(f"channels {C} not divisible by num_groups {G}")
    c_per_group = C // G

    weight = norm.weight
    bias = norm.bias
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    if bias.dtype != x.dtype:
        bias = bias.to(x.dtype)

    if c_per_group not in _SUPPORTED_C_PER_GROUP:
        # The kernels are templated on c_per_group; only {4, 8, 16, 32} are
        # instantiated. Anything else falls back to PyTorch. F.group_norm
        # preserves memory format for channels-last inputs, so NHWC stays NHWC.
        return F.silu(F.group_norm(x, G, weight=weight, bias=bias, eps=norm.eps))

    key = _tune_key(x, G)
    chosen = _TUNE_CACHE.get(key)
    if chosen is None:
        chosen = _tune_shape(x, weight, bias, G, c_per_group, norm.eps)
        if chosen is None:
            return F.silu(F.group_norm(x, G, weight=weight, bias=bias, eps=norm.eps))
        _TUNE_CACHE[key] = chosen

    path, params = chosen
    return _RUN_FNS[path](x, weight, bias, G, c_per_group, norm.eps, params)


__all__ = [
    "apply_group_norm_silu_nhwc",
    "clear_autotune_cache",
    "autotune_cache_snapshot",
]
