#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
import hashlib

__golden__ = {
    "kernel": {
        "cumulative_logsumexp": "cumulative_logsumexp_golden",
    }
}

__input__ = {
    "kernel": {
        "cumulative_logsumexp": "cumulative_logsumexp_input",
    }
}


def _stable_rng(testcase_name):
    seed = int(hashlib.md5(str(testcase_name).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _range_for_x(input_ranges):
    try:
        low, high = input_ranges[0]
    except (TypeError, IndexError):
        return -8.0, 8.0
    if low is None or high is None:
        return -8.0, 8.0
    return float(low), float(high)


def cumulative_logsumexp_input(
    x, axis, *, testcase_name="", input_ranges=None, **kwargs
):
    rng = _stable_rng(testcase_name)
    low, high = _range_for_x(input_ranges)
    x_data = rng.uniform(low, high, x.shape).astype(x.dtype, copy=False)
    return x_data, axis


def _normalize_axis(axis, rank):
    axis = (
        int(axis.item())
        if hasattr(axis, "item")
        else int(np.asarray(axis).reshape(-1)[0])
    )
    if axis < 0:
        axis += rank
    return axis


def _torch_logcumsumexp(x, axis, exclusive, reverse):
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    if reverse:
        tensor = torch.flip(tensor, dims=(axis,))
    result = torch.logcumsumexp(tensor, dim=axis)
    if exclusive:
        first_shape = list(result.shape)
        first_shape[axis] = 1
        first = torch.full(first_shape, -float("inf"), dtype=result.dtype)
        tail = result.narrow(axis, 0, result.shape[axis] - 1)
        result = torch.cat((first, tail), dim=axis)
    if reverse:
        result = torch.flip(result, dims=(axis,))
    return result.numpy()


def cumulative_logsumexp_golden(x, axis, *, exclusive=False, reverse=False, **kwargs):
    axis = _normalize_axis(axis, x.ndim)
    out_dtype = x.dtype
    result = _torch_logcumsumexp(x, axis, exclusive, reverse)
    return result.astype(out_dtype, copy=False)
