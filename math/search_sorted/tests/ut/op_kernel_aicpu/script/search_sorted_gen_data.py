# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import torch
import numpy as np


def write_file_txt(file_name, data, fmt="%s"):
    """
    prama1: file_name: the file which store the data
    param2: data: data which will be stored
    param3: fmt: format
    """
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def read_file_txt(file_name, dtype, delim=None):
    """
    prama1: file_name: the file which store the data
    param2: dtype: data type
    param3: delim: delimiter which is used to split data
    """
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_txt_to_bool(file_name, delim=None):
    """
    prama1: file_name: the file which store the data
    param2: delim: delimiter which is used to split data
    """
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    """
    prama1: data_file: the file which store the generation data
    param2: shape: data shape
    param3: dtype: data type
    param4: rand_type: the method of generate data, select from "randint, uniform"
    param5: data lower limit
    param6: data upper limit
    """
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


def gen_sorted_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.sort(np.random.randint(low, high, size=shape))
    else:
        rand_data = np.sort(np.random.uniform(low, high, size=shape))
    if dtype == np.float16:
        data = np.sort(np.array(rand_data, dtype=np.float32))
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.sort(np.array(rand_data, dtype=dtype))
        write_file_txt(data_file, data, fmt="%s")
    return data


def gen_random_data_float():
    data_files = ["search_sorted/data/search_sorted_float_input1.txt",
                "search_sorted/data/search_sorted_float_input2.txt",
                "search_sorted/data/search_sorted_float_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 3, 10]
    shape_x2 = [4, 3, 3]
    a = gen_sorted_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)
    data = torch.searchsorted(torch.from_numpy(a), torch.from_numpy(b))
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_double():
    data_files = ["search_sorted/data/search_sorted_double_input1.txt",
                "search_sorted/data/search_sorted_double_input2.txt",
                "search_sorted/data/search_sorted_double_output.txt"]
    np.random.seed(23457)
    shape_x1 = [100]
    shape_x2 = [200]
    a = gen_sorted_data_file(data_files[0], shape_x1, np.float64, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float64, "uniform", 0, 200)
    data = torch.searchsorted(torch.from_numpy(a), torch.from_numpy(b), out_int32=True, right=True)
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int8():
    data_files = ["search_sorted/data/search_sorted_int8_input1.txt",
                "search_sorted/data/search_sorted_int8_input2.txt",
                "search_sorted/data/search_sorted_int8_input3.txt",
                "search_sorted/data/search_sorted_int8_output.txt"]
    np.random.seed(23457)
    shape_x1 = [20, 10]
    shape_x2 = [20, 100]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", 0, 200)
    c = np.argsort(a)
    write_file_txt(data_files[2], c, fmt="%s")
    data = torch.searchsorted(torch.from_numpy(np.sort(a)), torch.from_numpy(b), out_int32=True, right=False)
    write_file_txt(data_files[3], data, fmt="%s")


def gen_random_data_int32():
    data_files = ["search_sorted/data/search_sorted_int32_input1.txt",
                "search_sorted/data/search_sorted_int32_input2.txt",
                "search_sorted/data/search_sorted_int32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 3, 10]
    shape_x2 = [4, 3, 3]
    a = gen_sorted_data_file(data_files[0], shape_x1, np.int32, "randint", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", 0, 200)
    data = torch.searchsorted(torch.from_numpy(a), torch.from_numpy(b), right=True)
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["search_sorted/data/search_sorted_int64_input1.txt",
                "search_sorted/data/search_sorted_int64_input2.txt",
                "search_sorted/data/search_sorted_int64_input3.txt",
                "search_sorted/data/search_sorted_int64_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 3, 10]
    shape_x2 = [4, 3, 3]
    a = gen_data_file(data_files[0], shape_x1, np.int64, "randint", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.int64, "randint", 0, 200)
    c = np.argsort(a)
    write_file_txt(data_files[2], c, fmt="%s")
    data = torch.searchsorted(torch.from_numpy(np.sort(a)), torch.from_numpy(b), right=True)
    write_file_txt(data_files[3], data, fmt="%s")


def run():
    gen_random_data_float()
    gen_random_data_double()
    gen_random_data_int8()
    gen_random_data_int32()
    gen_random_data_int64()


if __name__ == '__main__':
    os.system("mkdir -p search_sorted/data")
    run()