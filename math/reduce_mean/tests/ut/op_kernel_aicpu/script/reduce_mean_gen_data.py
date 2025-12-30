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
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes


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
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    else:
        r1 = np.random.uniform(low, high, size=shape)
        r2 = np.random.uniform(low, high, size=shape)
        rand_data = np.empty(shape=shape, dtype=dtype)
        for x1 in range(0, shape[0]):
            for x2 in range(0, shape[1]):
                for x3 in range(0, shape[2]):
                    for x4 in range(0, shape[3]):
                        rand_data[x1, x2, x3, x4] = complex(r1[x1, x2, x3, x4], r2[x1, x2, x3, x4])
    if dtype == np.float16:
        data = np.array(rand_data, dtype=np.float32)
        write_file_txt(data_file, data, fmt="%s")
        data = np.array(data, dtype=np.float16)
    else:
        data = np.array(rand_data, dtype=dtype)
        write_file_txt(data_file, data, fmt="%s")
    return data


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config


def gen_random_data_int32():
    data_files = ["reduce_mean/data/reduce_mean_int32_input1.txt",
                  "reduce_mean/data/reduce_mean_int32_input2.txt",
                  "reduce_mean/data/reduce_mean_int32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 4, 2]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 1, 10)
    x2 = np.arange(len(shape_x1), dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int32_2():
    data_files = ["reduce_mean/data/reduce_mean_int32_2_input1.txt",
                  "reduce_mean/data/reduce_mean_int32_2_input2.txt",
                  "reduce_mean/data/reduce_mean_int32_2_output.txt"]
    np.random.seed(23457)
    shape_x1 = [1, 1, 1]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 1, 10)
    x2 = np.array([0], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int32_3():
    data_files = ["reduce_mean/data/reduce_mean_int32_3_input1.txt",
                  "reduce_mean/data/reduce_mean_int32_3_input2.txt",
                  "reduce_mean/data/reduce_mean_int32_3_output.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 3, 1, 4]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", 1, 10)
    x2 = np.array([1], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["reduce_mean/data/reduce_mean_int64_input1.txt",
                  "reduce_mean/data/reduce_mean_int64_input2.txt",
                  "reduce_mean/data/reduce_mean_int64_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 6]
    a = gen_data_file(data_files[0], shape_x1, np.int64, "randint", 1, 5)
    x2 = np.array([1], dtype=np.int64)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int64, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float():
    data_files = ["reduce_mean/data/reduce_mean_float_input1.txt",
                  "reduce_mean/data/reduce_mean_float_input2.txt",
                  "reduce_mean/data/reduce_mean_float_output.txt"]
    np.random.seed(23457)
    shape_x1 = [1, 3, 6]
    a = gen_data_file(data_files[0], shape_x1, np.float, "uniform", -1, 1)
    x2 = np.array([0], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float2():
    data_files = ["reduce_mean/data/reduce_mean_float2_input1.txt",
                  "reduce_mean/data/reduce_mean_float2_input2.txt",
                  "reduce_mean/data/reduce_mean_float2_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 3, 6]
    a = gen_data_file(data_files[0], shape_x1, np.float, "uniform", -1, 1)
    x2 = np.array([0], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float3():
    data_files = ["reduce_mean/data/reduce_mean_float3_input1.txt",
                  "reduce_mean/data/reduce_mean_float3_input2.txt",
                  "reduce_mean/data/reduce_mean_float3_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 3, 5, 6]
    a = gen_data_file(data_files[0], shape_x1, np.float, "uniform", -1, 1)
    x2 = np.array([2], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_float16():
    data_files = ["reduce_mean/data/reduce_mean_float16_input1.txt",
                  "reduce_mean/data/reduce_mean_float16_input2.txt",
                  "reduce_mean/data/reduce_mean_float16_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3]
    a = gen_data_file(data_files[0], shape_x1, np.float16, "uniform", 1, 5)
    x2 = np.array([0, 2], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.float16, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_double():
    data_files = ["reduce_mean/data/reduce_mean_double_input1.txt",
                  "reduce_mean/data/reduce_mean_double_input2.txt",
                  "reduce_mean/data/reduce_mean_double_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2, 3, 2, 3]
    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", -1, 1)
    x2 = np.array([1, 3, 5, 7], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_double2():
    data_files = ["reduce_mean/data/reduce_mean_double2_input1.txt",
                  "reduce_mean/data/reduce_mean_double2_input2.txt",
                  "reduce_mean/data/reduce_mean_double2_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2, 3, 2, 3]
    a = gen_data_file(data_files[0], shape_x1, np.float64, "uniform", -1, 1)
    x2 = np.array([0, 2, 4, 6], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.float64, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int8():
    data_files = ["reduce_mean/data/reduce_mean_int8_input1.txt",
                  "reduce_mean/data/reduce_mean_int8_input2.txt",
                  "reduce_mean/data/reduce_mean_int8_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2, 3, 2]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", 1, 5)
    x2 = np.array([0, 2, 4, 6], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint8():
    data_files = ["reduce_mean/data/reduce_mean_uint8_input1.txt",
                  "reduce_mean/data/reduce_mean_uint8_input2.txt",
                  "reduce_mean/data/reduce_mean_uint8_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2, 3, 2]
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", 1, 5)
    x2 = np.array([1, 3, 5], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int16():
    data_files = ["reduce_mean/data/reduce_mean_int16_input1.txt",
                  "reduce_mean/data/reduce_mean_int16_input2.txt",
                  "reduce_mean/data/reduce_mean_int16_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2, 3]
    a = gen_data_file(data_files[0], shape_x1, np.int16, "randint", 1, 5)
    x2 = np.array([0, 2, 4], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.int16, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint16():
    data_files = ["reduce_mean/data/reduce_mean_uint16_input1.txt",
                  "reduce_mean/data/reduce_mean_uint16_input2.txt",
                  "reduce_mean/data/reduce_mean_uint16_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2, 3]
    a = gen_data_file(data_files[0], shape_x1, np.uint16, "randint", 1, 5)
    x2 = np.array([1, 3, 5], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.uint16, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_uint32():
    data_files = ["reduce_mean/data/reduce_mean_uint32_input1.txt",
                  "reduce_mean/data/reduce_mean_uint32_input2.txt",
                  "reduce_mean/data/reduce_mean_uint32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2]
    a = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", 1, 5)
    x2 = np.array([0, 2, 4], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_uint64():
    data_files = ["reduce_mean/data/reduce_mean_uint64_input1.txt",
                  "reduce_mean/data/reduce_mean_uint64_input2.txt",
                  "reduce_mean/data/reduce_mean_uint64_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 3, 2, 3, 2]
    a = gen_data_file(data_files[0], shape_x1, np.uint64, "randint", 1, 5)
    x2 = np.array([1, 3], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex64():
    data_files = ["reduce_mean/data/reduce_mean_complex64_input1.txt",
                  "reduce_mean/data/reduce_mean_complex64_input2.txt",
                  "reduce_mean/data/reduce_mean_complex64_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 5, 3, 7]
    a = gen_data_file(data_files[0], shape_x1, np.complex64, "complex", 0, 5)
    x2 = np.array([0], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex128():
    data_files = ["reduce_mean/data/reduce_mean_complex128_input1.txt",
                  "reduce_mean/data/reduce_mean_complex128_input2.txt",
                  "reduce_mean/data/reduce_mean_complex128_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 5, 1, 3]
    a = gen_data_file(data_files[0], shape_x1, np.complex128, "complex", 0, 5)
    x2 = np.array([1], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex64_2():
    data_files = ["reduce_mean/data/reduce_mean_complex64_2_input1.txt",
                  "reduce_mean/data/reduce_mean_complex64_2_input2.txt",
                  "reduce_mean/data/reduce_mean_complex64_2_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 5, 3, 6]
    a = gen_data_file(data_files[0], shape_x1, np.complex64, "complex", 0, 5)
    x2 = np.array([0, 2], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.complex64, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_complex128_2():
    data_files = ["reduce_mean/data/reduce_mean_complex128_2_input1.txt",
                  "reduce_mean/data/reduce_mean_complex128_2_input2.txt",
                  "reduce_mean/data/reduce_mean_complex128_2_output.txt"]
    np.random.seed(23457)
    shape_x1 = [4, 5, 2, 1]
    a = gen_data_file(data_files[0], shape_x1, np.complex128, "complex", 0, 5)
    x2 = np.array([1, 3], dtype=np.int32)
    write_file_txt(data_files[1], x2, fmt="%s")
    x1 = tf.compat.v1.placeholder(tf.complex128, shape=shape_x1)
    re = tf.math.reduce_mean(input_tensor=x1, axis=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a})
    write_file_txt(data_files[2], data, fmt="%s")


def run():
    gen_random_data_int32()
    gen_random_data_int32_2()
    gen_random_data_int32_3()
    gen_random_data_int64()
    gen_random_data_float()
    gen_random_data_float2()
    gen_random_data_float3()
    gen_random_data_float16()
    gen_random_data_double()
    gen_random_data_double2()
    gen_random_data_int8()
    gen_random_data_uint8()
    gen_random_data_int16()
    gen_random_data_uint16()
    gen_random_data_uint32()
    gen_random_data_uint64()
    gen_random_data_complex64()
    gen_random_data_complex128()
    gen_random_data_complex64_2()
    gen_random_data_complex128_2()
	
if __name__ == '__main__':
    os.system("mkdir -p reduce_mean/data")
    run()
