"""
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import numpy as np
import torch
# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')

# prama1: file_name: the file which store the data
# param2: dtype: data type
# param3: delim: delimiter which is used to split data
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()

# prama1: file_name: the file which store the data
# param2: delim: delimiter which is used to split data
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

# prama1: data_file: the file which store the generation datatf
# param2: shape: data shape
# param3: dtype: data type
# param4: rand_type: the method of generate data, select from "randint, uniform"
# param5: data lower limit
# param6: data upper limit
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    np.random.seed(23457)
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
        write_file_txt(data_file, data, fmt="%s")
    return data

def gen_random_data_float32(decimals_value):
    data_files=["round/data/round_data_input1_float32_",
                "round/data/round_data_output1_float32_"]
    shape1 = [4, 4]
    input_data1 = gen_data_file(data_files[0] + str(decimals_value) + ".txt", shape1, np.float32, "", -10, 10)
    x = torch.from_numpy(np.array(input_data1, dtype = np.float32))
    y = torch.round(x, decimals = decimals_value)
    write_file_txt(data_files[1] + str(decimals_value) + ".txt", y, fmt="%s")

def gen_random_data_float64(decimals_value):
    data_files=["round/data/round_data_input1_float64_",
                "round/data/round_data_output1_float64_"]
    shape1 = [4, 4]
    input_data1 = gen_data_file(data_files[0] + str(decimals_value) + ".txt", shape1, np.float64, "", -10, 10)
    x = torch.from_numpy(np.array(input_data1, dtype = np.float64))
    y = torch.round(x, decimals = decimals_value)
    write_file_txt(data_files[1] + str(decimals_value) + ".txt", y, fmt="%s")

def run():
    gen_random_data_float32(1)
    gen_random_data_float32(2)
    gen_random_data_float32(-1)
    gen_random_data_float32(-2)
    gen_random_data_float64(1)

run()