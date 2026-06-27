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

import numpy as np

__golden__ = {
    "kernel": {"sparse_bincount": "sparse_bincount_golden"}
}


def sparse_bincount_golden(indices, values, dense_shape, size, weights, *, binary_output=False, **kwargs):
    '''
    Golden function for sparse_bincount.
    All the parameters (names and order) follow @sparse_bincount_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        indices: np.ndarray, shape=(N, R), dtype=int64, sparse tensor indices
        values: np.ndarray, shape=(N,), dtype=int32/int64, sparse tensor values (bin index)
        dense_shape: np.ndarray, shape=(R,), dtype=int64, sparse tensor dense shape
        size: np.ndarray, shape=(1,), dtype=int32/int64, number of bins
        weights: np.ndarray, shape=(N,) or (0,), dtype=float32, weights
        binary_output: bool, whether to output binary values
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    num_size = int(size.flat[0])
    weights_num = len(weights)
    dense_shape_num = len(dense_shape)
    is_1d = (dense_shape_num == 1)

    if is_1d:
        out = np.zeros((num_size,), dtype=np.float32)
        if binary_output:
            for i in range(len(values)):
                value = int(values[i])
                if value < num_size:
                    out[value] = 1.0
        else:
            if weights_num > 0:
                for i in range(len(values)):
                    value = int(values[i])
                    if value < num_size:
                        out[value] += float(weights[i])
            else:
                for i in range(len(values)):
                    value = int(values[i])
                    if value < num_size:
                        out[value] += 1.0
        return out
    else:
        dense_shape_rows = int(dense_shape[0])
        output_shape = dense_shape_rows * num_size
        out = np.zeros((output_shape,), dtype=np.float32)
        for i in range(len(values)):
            if i < len(indices):
                value_batch = int(indices[i][0])
                value_bin = int(values[i])
                offset = value_batch * num_size + value_bin
                if (offset < dense_shape_rows * num_size) and (value_bin < num_size):
                    if binary_output:
                        out[offset] = 1.0
                    else:
                        if weights_num > 0:
                            out[offset] += float(weights[i])
                        else:
                            out[offset] += 1.0
        out = out.reshape((dense_shape_rows, num_size))
        return out
