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
PopulationCount kernel-direct golden.

Uses a NumPy byte lookup table as the golden reference.
Formula: y = sum(bit_i(x)), i in [0, 15]
  x : int16 or uint16 input
  y : uint8 population count in [0, 16]

The kernel test context supplies NumPy arrays. Counting the raw bytes preserves
the two's-complement bit representation of signed int16 inputs.
"""

from types import SimpleNamespace

import numpy as np


_BYTE_POPCOUNT = np.array(
    [bin(value).count("1") for value in range(256)],
    dtype=np.uint8,
)


__golden__ = {
    "kernel": {
        "population_count": "population_count_golden",
    }
}


def population_count_golden(x, **kwargs):
    del kwargs
    context = SimpleNamespace(input_arrays=(x,))
    return _population_count(context)


def _population_count(context):
    arrs = context.input_arrays
    x = arrs[0]

    # Count each byte independently so signed inputs keep their original bits.
    x_bytes = np.ascontiguousarray(x).view(np.uint8)
    x_bytes = x_bytes.reshape(x.size, x.dtype.itemsize)
    result = _BYTE_POPCOUNT[x_bytes].sum(axis=1, dtype=np.uint8)
    return result.reshape(x.shape)
