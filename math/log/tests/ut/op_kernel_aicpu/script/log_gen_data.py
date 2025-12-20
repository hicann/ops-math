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
import numpy as np
from pickle import TRUE
import torch


def write_file_txt(file_name, data):
    """
    Write to text file

    Parameters
    ----------
    file_name: str
        the file which store the data
    data: 2-D or a multi-dimensional array
        data which will be stored
    """
    if np.iscomplexobj(data):
        fmt = "(%s,%s)"
    else:
        fmt = "%s"
    if file_name is None:
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter="", newline="\n")


def gen_data(shape, dtype, rand_type, low, high):
    """
    Generate random data

    Parameters
    ----------
    shape: int or tuple of ints
        data shape
    dtype: dtype
        data type
    rand_type: str
        the method of generate data, select from "randint, uniform"
    low: int or array-like of ints
        data lower limit
    high: int or array-like of ints
        data upper limit
    """
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif issubclass(dtype, np.complexfloating):
        rand_data = np.random.uniform(low, high, size=shape) + 1j * np.random.uniform(
            low, high, size=shape
        )
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    return np.array(rand_data, dtype=dtype)


def gen_random_data(nptype, shape, rand_type, low, high):
    np.random.seed(23457)
    input_data = gen_data(shape, nptype, rand_type, low, high)
    write_file_txt("log/data/log_data_input_" + nptype.__name__ + ".txt", input_data)
    x = torch.tensor(input_data)
    output_result = torch.log(x)
    output_data = output_result.numpy()
    write_file_txt("log/data/log_data_output_" + nptype.__name__ + ".txt", output_data)


def run():
    gen_random_data(np.uint8, [5, 4, 3], "uniform", 0, 100)
    gen_random_data(np.int8, [5, 4, 3], "uniform", 0, 100)
    gen_random_data(np.int16, [5, 4, 3], "uniform", 0, 100)
    gen_random_data(np.int32, [5, 4, 3], "uniform", 0, 100)
    gen_random_data(np.int64, [5, 4, 3], "uniform", 0, 100)
    gen_random_data(np.float32, [5, 4, 3], "uniform", 0, 100)
    gen_random_data(np.float64, [5, 4, 3], "uniform", 0, 100)


if __name__ == '__main__':
    os.system("mkdir -p log/data")
    run()