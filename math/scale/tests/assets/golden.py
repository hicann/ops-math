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
"""
aclnnScale golden: y = x * scale [+ bias]

scale/bias broadcast against x at [axis : axis+span], rest dims size 1.
scaleFromBlob=True : span = numAxes (numAxes==-1 -> to last axis; 0 -> scale shape [1])
scaleFromBlob=False: span = rank(scale) (numAxes ignored)
Mirrors kernel-direct golden so aclnn / kernel cross-check stays consistent.
"""

from types import SimpleNamespace

import numpy as np


__golden__ = {
    "aclnn": {
        "aclnnScale": "scale_golden",
    }
}


def _align(x_shape, axis, span, t):
    rank = len(x_shape)
    if axis < 0:
        axis += rank
    bshape = [1] * rank
    flat = np.asarray(t).reshape(-1)
    n = flat.size
    bshape[axis : axis + span] = list(x_shape[axis : axis + span])
    if int(np.prod(bshape)) != n:
        bshape = [1] * rank
    return flat.reshape(bshape)


def scale_golden(x, scale, bias, axis, numAxes, scaleFromBlob, y, **kwargs):
    del y, kwargs
    context = SimpleNamespace(
        tensors=(x, scale, bias),
        attributes={
            "axis": axis,
            "numAxes": numAxes,
            "scaleFromBlob": scaleFromBlob,
        },
    )
    return _scale(context)


def _scale(context):
    import torch

    x = context.tensors[0].to(torch.float32).numpy()
    scale = context.tensors[1].to(torch.float32).numpy()
    bias = (
        context.tensors[2].to(torch.float32).numpy()
        if len(context.tensors) > 2 and context.tensors[2] is not None
        else None
    )
    if bias is not None and bias.size == 0:
        bias = None  # 空 bias 张量(shape含0)视为无 bias

    attr = context.attributes or {}
    axis = int(attr.get("axis", 1))
    num_axes = int(attr.get("numAxes", attr.get("num_axes", 1)))
    from_blob = bool(attr.get("scaleFromBlob", attr.get("scale_from_blob", True)))

    rank = len(x.shape)
    a = axis + rank if axis < 0 else axis
    if from_blob:
        span = (rank - a) if num_axes == -1 else num_axes
    else:
        span = scale.ndim
    span = max(0, min(span, rank - a))

    s = _align(x.shape, a, span, scale)
    y = x * s
    if bias is not None:
        y = y + _align(x.shape, a, span, bias)
    return torch.from_numpy(np.asarray(y, dtype=np.float32)).to(
        context.tensors[0].dtype
    )
