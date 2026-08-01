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
"""Torch CPU golden for ExpSegsumGrad."""

import numpy as np
import torch

__golden__ = {"kernel": {"exp_segsum_grad": "exp_segsum_grad_golden"}}


def _to_f32_array(arr):
    if hasattr(arr, "astype"):
        return arr.astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def exp_segsum_grad_golden(grad_output, grad_self, **kwargs):
    out_dtype = getattr(grad_output, "dtype", None)
    grad_output_t = torch.as_tensor(_to_f32_array(grad_output), dtype=torch.float32)
    grad_self_t = torch.as_tensor(_to_f32_array(grad_self), dtype=torch.float32)
    t_dim = grad_output_t.shape[-1]

    tril_mask = torch.tril(torch.ones(t_dim, t_dim, dtype=torch.bool), diagonal=0)
    zero = torch.zeros((), dtype=torch.float32)
    y = torch.where(tril_mask, grad_output_t * grad_self_t, zero)
    y = torch.flip(y, dims=[-2])
    y = torch.cumsum(y, dim=-2, dtype=torch.float32)
    y = torch.flip(y, dims=[-2])
    strict_tril = torch.tril(torch.ones(t_dim, t_dim, dtype=torch.bool), diagonal=-1)
    y = torch.where(strict_tril, y, zero)
    grad_input = torch.sum(y, dim=-1, dtype=torch.float32)

    result = grad_input.numpy()
    out_dtype_name = (
        getattr(out_dtype, "name", str(out_dtype)).lower()
        if out_dtype is not None
        else ""
    )
    if "float16" in out_dtype_name or "bfloat16" in out_dtype_name:
        return result.astype(out_dtype, copy=False)
    return result.astype(np.float32)
