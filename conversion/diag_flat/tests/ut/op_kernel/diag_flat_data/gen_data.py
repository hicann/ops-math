#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np


input1_int8 = np.arange(1, 255).astype(np.int8)
input1_int16 = np.arange(1, 255).astype(np.int16)
input1_int32 = np.arange(1, 255).astype(np.int32)
input1_int64 = np.arange(1, 257).astype(np.int64)

input2_int8 = np.arange(1, 260).astype(np.int8)
input2_int16 = np.arange(1, 260).astype(np.int16)
input2_int64 = np.arange(1, 260).astype(np.int64)


input3_int8 = np.arange(1, 60).astype(np.int8)
input3_int16 = np.arange(1, 60).astype(np.int16)
input3_int64 = np.arange(1, 60).astype(np.int64)


input_data1 = [(np.random.randint(1, 20, (1,)) + 
               1.j * np.random.randint(1, 20, (1,)))[0] for _ in range(1, 259 + 1)]
input4_complex128 = np.asarray(input_data1).astype(np.complex128)

input_data2 = [(np.random.randint(1, 20, (1,)) + 
               1.j * np.random.randint(1, 20, (1,)))[0] for _ in range(1, 59 + 1)]
input5_complex128 = np.asarray(input_data2).astype(np.complex128)

case0_params = {
    "input": input1_int8,
    "offset": -2,
    "d_type": np.int8,
}

case1_params = {
    "input": input1_int16,
    "offset": -2,
    "d_type": np.int16,
}

case2_params = {
    "input": input1_int32,
    "offset": -2,
    "d_type": np.int32,
}

case3_params = {
    "input": input1_int64,
    "offset": -2,
    "d_type": np.int64,
}

case4_params = {
    "input": input2_int8,
    "offset": 0,
    "d_type": np.int8,
}

case5_params = {
    "input": input2_int16,
    "offset": 0,
    "d_type": np.int16,
}

case6_params = {
    "input": input2_int64,
    "offset": 0,
    "d_type": np.int64,
}

case7_params = {
    "input": input3_int16,
    "offset": -2,
    "d_type": np.int16,
}

case8_params = {
    "input": input2_int64,
    "offset": -2,
    "d_type": np.int64,
}

case9_params = {
    "input": input4_complex128,
    "offset": 100,
    "d_type": np.complex128,
}

case10_params = {
    "input": input5_complex128,
    "offset": 0,
    "d_type": np.complex128,
}


test_cast = {
    'case0': case0_params,
    'case1': case1_params,
    'case2': case2_params,
    'case3': case3_params,
    'case4': case4_params,
    'case5': case5_params,
    'case6': case6_params,
    'case7': case7_params,
    'case8': case8_params,
    'case9': case9_params,
    'case10':case10_params,
}


def gen_data_and_golden(case_num):
    case_params = test_cast[str(case_num)]
    input = case_params['input']
    d_type = case_params['d_type']

    input = input.astype(d_type)
    input.tofile('./input.bin')
    offset = case_params['offset']
    golden = np.diagflat(input, offset)
    golden.tofile('./golden.bin')


if __name__ == "__main__":
    gen_data_and_golden(sys.argv[1])