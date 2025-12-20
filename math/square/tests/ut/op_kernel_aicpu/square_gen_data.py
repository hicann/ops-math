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
from numpy import testing
import tensorflow as tf
import numpy as np

# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if file_name is None:
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
def read_file_txt_to_boll(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)

# param1: shape: data shape
# param2: dtype: data type
# param3: rand_type: the method of generate data, select from "randint, uniform"
# param4: data lower limit
# param5: data upper limit
def gen_data(shape, dtype, rand_type, low, high):
    if rand_type == "randint":        
        if dtype == "complex64":
            x_real = np.random.randint(low, high, size=shape)
            x_imag = np.random.randint(low, high, size=shape)
            rand_data_x = np.complex64(x_real)
            rand_data_x.imag = x_imag
        elif dtype == "complex128":
            x_real = np.random.randint(low, high, size=shape)
            x_imag = np.random.randint(low, high, size=shape)
            rand_data_x = np.complex128(x_real)
            rand_data_x.imag = x_imag
        else:
            rand_data_x = np.random.randint(low, high, size=shape)
    else:
        if dtype == "complex64":
            x_real = np.random.uniform(low, high, size=shape)
            x_imag = np.random.uniform(low, high, size=shape)
            rand_data_x = np.complex64(x_real)
            rand_data_x.imag = x_imag
        elif dtype == "complex128":
            x_real = np.random.uniform(low, high, size=shape)
            x_imag = np.random.uniform(low, high, size=shape)
            rand_data_x = np.complex128(x_real)
            rand_data_x.imag = x_imag
        else:
            rand_data_x = np.random.uniform(low, high, size=shape)
    return np.array(rand_data_x, dtype=dtype)

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    return session_config

def gen_random_data(nptype, shape, rand_type, low, high, flag):
    if flag == 1:
        np.random.seed(23457)
        x_batch = gen_data(shape, nptype, rand_type, low, high)
        write_file_txt("square/data/square_bigdata_input_" + nptype.__name__ + ".txt", x_batch, fmt="%s")
        x_batch.tofile("square/data/square_bigdata_input_" + nptype.__name__ + ".bin")
        write_file_txt("square/data/square_bigdata_" + nptype.__name__ + "_dim.txt", np.array([len(shape)]), fmt="%s")
        write_file_txt("square/data/square_bigdata_" + nptype.__name__ + "_shape.txt", np.array(shape), fmt="%s")
        tf.compat.v1.disable_eager_execution()
        x_placeholder = tf.compat.v1.placeholder(nptype, shape=shape)
        result = tf.math.square(x_placeholder)
        with tf.compat.v1.Session(config=config('cpu')) as session:
            data = session.run(result, feed_dict={x_placeholder: x_batch})
        write_file_txt("square/data/square_bigdata_output_" + nptype.__name__ + ".txt", data, fmt="%s")
        data.tofile("square/data/square_bigdata_output_" + nptype.__name__ + ".bin")

    else:
        np.random.seed(23498)
        x_batch = gen_data(shape, nptype, rand_type, low, high)
        write_file_txt("square/data/square_data_input_" + nptype.__name__ + ".txt", x_batch, fmt="%s")
        x_batch.tofile("square/data/square_data_input_" + nptype.__name__ + ".bin")
        write_file_txt("square/data/square_data_" + nptype.__name__ + "_dim.txt", np.array([len(shape)]), fmt="%s")
        write_file_txt("square/data/square_data_" + nptype.__name__ + "_shape.txt", np.array(shape), fmt="%s")
        tf.compat.v1.disable_eager_execution()
        x_placeholder = tf.compat.v1.placeholder(nptype, shape=shape)
        result = tf.math.square(x_placeholder)
        with tf.compat.v1.Session(config=config('cpu')) as session:
            data = session.run(result, feed_dict={x_placeholder: x_batch})
        write_file_txt("square/data/square_data_output_" + nptype.__name__ + ".txt", data, fmt="%s")
        data.tofile("square/data/square_data_output_" + nptype.__name__ + ".bin")
def run():
    gen_random_data(np.int32, [7, 12, 30], "uniform", -1000, 1000, 0)
    gen_random_data(np.int32, [32768], "uniform", -100, 100, 1)
    gen_random_data(np.int64, [7, 12, 30], "uniform", -100000, 100000, 0)
    gen_random_data(np.float16, [7, 12, 30], "uniform", -1000, 1000, 0)
    gen_random_data(np.float16, [99000], "uniform", -1000, 1000, 1)
    gen_random_data(np.float32, [7, 12, 30], "uniform", -100, 100, 0)
    gen_random_data(np.float64, [7, 12, 30], "uniform", -100000, 100000, 0)
    gen_random_data(np.complex64, [15, 12, 30], "uniform", -10, 10, 0)
    gen_random_data(np.complex128, [7, 12, 30], "uniform", -999, 999, 0)

if __name__ == '__main__':
    run()