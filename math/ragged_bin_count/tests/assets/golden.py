#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""RaggedBinCount multi-path golden in the TestSpec format.

Only one registration key is needed: kernel and GEIR both resolve the snake-case
operator name and share :class:`RaggedBinCountKernelSpec`.  The legacy
``__golden__`` entry stays available and calls the same Torch computation core.

The reference math is a Torch competitor composition (``torch.bincount`` per
ragged row), not a hand-written NumPy formula, so it cannot repeat the kernel's
own reasoning error.  ``torch.bincount`` is semantically the per-row form of the
TensorFlow ``RaggedBincount`` operator this IR is compatible with.

Frozen contract points the golden must reproduce (docs/spec.yaml §3.4):
  * ``weights`` is judged by element count only, never by rank: zero elements
    (``[0]``, ``[0, 3]``, ``[2, 0]`` alike) accumulates 1.0 per hit, and a
    non-empty ``weights`` pairs with ``values`` in flattened order;
  * ``binary_output=True`` writes exactly 0.0/1.0 and ignores ``weights``
    entirely, including NaN/Inf/negative weights;
  * ``values >= M`` is legal and ignored;
  * if any split is illegal or any value is negative, the **whole** output is
    bitwise ``+0.0`` and no weight is ever read.
"""

import numpy as np
import torch


__spec__ = {
    # kernel + GEIR share this snake-case key and this one Spec class.
    "ragged_bin_count": "RaggedBinCountKernelSpec",
}

__golden__ = {
    "kernel": {"ragged_bin_count": "ragged_bin_count_golden"},
}


_TOL = {
    # The only native output dtype is FP32 (ragged_bin_count_def.cpp).
    #
    # ``cross_check`` is the right standard because the reference is an FP64
    # accumulation rounded once (see :func:`_compute`): the NPU error and the
    # competitor error are then both measured against the correctly-rounded
    # answer, and their ratio says "is the NPU within the same order of accuracy
    # as a mature GPU implementation" -- which is the question worth asking for
    # an operator whose summation order is deliberately not reproducible
    # (``asc_atomic_add`` across 56 AIV cores).
    #
    # This requires a reachable remote XPU endpoint to execute ``third_party``.
    # Runs without one must pass ``--compare close`` (or ``stat_rel_err``)
    # explicitly on the command line, which overrides this entry.
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    """CSV attributes arrive as strings; coerce them to the default's type."""
    value = kwargs.get(name, default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false", "yes", "no", "1", "0"):
            return text in ("true", "yes", "1")
        try:
            return type(default)(value)
        except Exception:
            return default
    return value


def _as_tensor(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(np.ascontiguousarray(array))


def _contract_inputs(splits, values, size, weights):
    """Flatten to the shapes the frozen contract works in."""
    splits_flat = _as_tensor(splits).reshape(-1).to(torch.int64)
    values_flat = _as_tensor(values).reshape(-1).to(torch.int64)
    size_flat = _as_tensor(size).reshape(-1).to(torch.int64)
    weights_flat = _as_tensor(weights).reshape(-1)
    return splits_flat, values_flat, size_flat, weights_flat


def _runtime_invalid(splits_flat, values_flat):
    """Reproduce the whole-output +0.0 guard: illegal split or negative value."""
    value_count = values_flat.numel()
    if splits_flat.numel() < 2:
        return True
    if bool((splits_flat < 0).any()) or bool((splits_flat > value_count).any()):
        return True
    if int(splits_flat[0].item()) != 0:
        return True
    if int(splits_flat[-1].item()) != value_count:
        return True
    if bool((splits_flat[1:] < splits_flat[:-1]).any()):
        return True
    if bool((values_flat < 0).any()):
        return True
    return False


def _compute(splits, values, size, weights, **kwargs):
    """Return ``[output]`` as a Torch tensor in def.cpp output order."""
    binary_output = _attr(kwargs, "binary_output", False)
    splits_flat, values_flat, size_flat, weights_flat = _contract_inputs(
        splits, values, size, weights
    )

    rows = max(splits_flat.numel() - 1, 0)
    bins = int(size_flat[0].item()) if size_flat.numel() > 0 else 0
    output = torch.zeros((rows, bins), dtype=torch.float32)

    # Invalid input returns bitwise +0.0 before any weight is read.
    if _runtime_invalid(splits_flat, values_flat) or rows == 0 or bins == 0:
        return [output]

    has_weights = weights_flat.numel() > 0
    # Accumulate in FP64 and round once at the end.  The NPU sums a bin with
    # ``asc_atomic_add`` across 56 cores and any competitor sums it in its own
    # order; both are FP32 reorderings of the same exact sum.  Comparing them
    # against another FP32 reordering would only measure "how differently did
    # these two reorder", which is not a precision statement about either.  An
    # FP64 accumulation rounded to FP32 is the correctly-rounded answer, so the
    # deviation of each implementation from it is its real error -- this is the
    # "CPU-fp64 golden" the three-way precision standard asks for.
    accumulator = torch.zeros((rows, bins), dtype=torch.float64)
    weights_fp64 = weights_flat.to(torch.float64) if has_weights else None

    for row in range(rows):
        begin = int(splits_flat[row].item())
        end = int(splits_flat[row + 1].item())
        segment = values_flat[begin:end]
        # values >= M are legal and ignored.
        keep = segment < bins
        kept = segment[keep]
        if kept.numel() == 0:
            continue
        if binary_output:
            # Bitwise exact 0/1; weights are never read on this path.
            accumulator[row] = accumulator[row].index_fill(0, kept, 1.0)
        elif has_weights:
            segment_weights = weights_fp64[begin:end][keep]
            accumulator[row] = torch.bincount(
                kept, weights=segment_weights, minlength=bins
            )[:bins]
        else:
            accumulator[row] = torch.bincount(kept, minlength=bins)[:bins].to(
                torch.float64
            )

    output = accumulator.to(torch.float32)
    return [output]


def _output_dtype_names(kwargs):
    output_dtypes = kwargs.get("output_dtypes") or []
    names = []
    for dtype in output_dtypes:
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        names.append(str(dtype))
    return names


def _kernel_golden(splits, values, size, weights, **kwargs):
    """Kernel/GEIR container adapter: NumPy in, NumPy list out."""
    outputs = _compute(splits, values, size, weights, **kwargs)
    output_dtypes = _output_dtype_names(kwargs)
    result = []
    for index, output in enumerate(outputs):
        array = output.detach().cpu().contiguous().numpy()
        if index < len(output_dtypes):
            array = array.astype(output_dtypes[index], copy=False)
        result.append(np.ascontiguousarray(array))
    return result


def _competitor_forward(splits_flat, values_flat, bins, weights_flat, binary_output):
    """Optimal single-pass competitor form (see :class:`_RaggedBinCountCompose`)."""
    device = values_flat.device
    rows = max(splits_flat.numel() - 1, 0)
    output = torch.zeros(rows * bins, dtype=torch.float32, device=device)

    # Row id per value, from the ragged offsets: one repeat_interleave instead
    # of a Python loop over rows.
    counts = splits_flat[1:] - splits_flat[:-1]
    row_id = torch.repeat_interleave(
        torch.arange(rows, device=device, dtype=torch.int64), counts
    )

    keep = values_flat < bins  # values >= M are legal and ignored
    flat_index = row_id * bins + values_flat
    flat_index = flat_index[keep]

    if binary_output:
        # Weights are never read on this path; writes are idempotent 1.0.
        output[flat_index] = 1.0
    elif weights_flat.numel() > 0:
        output.index_add_(0, flat_index, weights_flat.to(torch.float32)[keep])
    else:
        output.index_add_(
            0,
            flat_index,
            torch.ones(flat_index.numel(), dtype=torch.float32, device=device),
        )

    return output.view(rows, bins)


class _RaggedBinCountCompose:
    """Independent competitor composition, executed on the remote GPU.

    Two requirements pull in different directions and both are met here:

    * It must not share an implementation with :func:`_compute`, or the two
      reference paths cannot cross-validate each other.  ``_compute`` is a
      per-row ``torch.bincount`` accumulating in FP64; this is a single flat
      ``index_add_`` in FP32 over a ``repeat_interleave``-derived row index.
    * It must be the competitor's **optimal** form, or the G/N ratio is
      inflated and the comparison is not an honest one.  The whole ragged
      problem is expressed as one scatter over the flattened output, so torch
      launches a constant number of kernels regardless of the row count -- the
      earlier per-row Python loop cost 66 ms on an A100 for a case the fused
      form finishes in tens of microseconds.
    """

    def __init__(self, **kwargs):
        self.binary_output = _attr(kwargs, "binary_output", False)

    def __call__(self, splits, values, size, weights, **kwargs):
        del kwargs
        splits_flat, values_flat, size_flat, weights_flat = _contract_inputs(
            splits, values, size, weights
        )
        rows = max(splits_flat.numel() - 1, 0)
        bins = int(size_flat[0].item()) if size_flat.numel() > 0 else 0
        if _runtime_invalid(splits_flat, values_flat) or rows == 0 or bins == 0:
            return [
                torch.zeros(
                    (rows, bins), dtype=torch.float32, device=values_flat.device
                )
            ]

        # NPU output is always FP32; align the competitor dtype.
        return [
            _competitor_forward(
                splits_flat, values_flat, bins, weights_flat, self.binary_output
            )
        ]


class RaggedBinCountKernelSpec:
    """Shared kernel/GEIR TestSpec; parameter names follow def.cpp."""

    @staticmethod
    def golden(splits, values, size, weights, **kwargs):
        return _kernel_golden(splits, values, size, weights, **kwargs)

    third_party = {"torch": _RaggedBinCountCompose}
    tolerance = _TOL


def ragged_bin_count_golden(splits, values, size, weights, *args, **kwargs):
    """Compatibility entry for the historical ``__golden__`` kernel loader."""
    del args
    legacy_kwargs = dict(kwargs)
    if not legacy_kwargs.get("output_dtypes"):
        # The old entry predates TestSpec and promises the fixed FP32 output.
        legacy_kwargs["output_dtypes"] = [["float32"]]
    return _kernel_golden(splits, values, size, weights, **legacy_kwargs)[0]


# 【不存在】ACLNN 通路：CMakeLists.txt 显式配置 ``ACLNNTYPE aclnn_exclude``，
# 本算子既无 op_host/op_api 目录，也不交付 aclnnRaggedBinCount 符号。
# 【不存在】e2e 通路：获批支持面为 Graph IR only（README.md 调用说明），
# torch_npu 二进制中无本算子的 aclnn 绑定入口。
