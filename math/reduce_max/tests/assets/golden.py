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


__golden__ = {"kernel": {"reduce_max": "reduce_max_golden"}}


def reduce_max_golden(x, axes=None, keep_dims: bool = False, **kwargs):
    """
    Kernel golden for reduce_max.
    All the parameters follow @reduce_max_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import numpy as np

    input_dtype = x.dtype
    if str(input_dtype) == "bfloat16":
        x = x.astype(np.float32)

    if axes is not None:
        axis = tuple(int(a) for a in np.asarray(axes).flatten())
    else:
        axis = None

    # Try TensorFlow first (supports empty tensors), fallback to PyTorch, then NumPy
    try:
        import tensorflow as tf

        x_tensor = tf.constant(x)
        res_tensor = tf.reduce_max(x_tensor, axis=axis, keepdims=keep_dims)
        res = res_tensor.numpy()
    except (ImportError, Exception):
        try:
            import torch

            x_torch = torch.from_numpy(x)
            if axis is not None:
                res_torch = torch.amax(x_torch, dim=axis, keepdim=keep_dims)
            else:
                res_torch = torch.amax(x_torch, keepdim=keep_dims)
            res = res_torch.numpy()
        except (ImportError, Exception):
            # Fallback to NumPy
            res = np.max(x, axis=axis, keepdims=keep_dims)
    return res.astype(input_dtype, copy=False)
