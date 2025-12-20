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
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes

tf.compat.v1.disable_eager_execution()

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
            bool_data.append(Falsetf)
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
        rand_data = np.empty((shape[0], shape[1], shape[2]), dtype=dtype)
        for i in range(0, shape[0]):
            for p in range(0, shape[1]):
                for k in range(0, shape[2]):
                    rand_data[i, p, k] = complex(r1[i, p, k], r2[i, p, k])
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


def gen_random_data_int32_broadcast1():
    data_files = ["not_equal/data/broadcast1_not_equal_data_int32_input1.txt",
                  "not_equal/data/broadcast1_not_equal_data_int32_input2.txt",
                  "not_equal/data/broadcast1_not_equal_data_int32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [2, 4]
    shape_x2 = [2, 2, 4]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.raw_ops.NotEqual(x=x1, y=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int32_broadcast2():
    data_files = ["not_equal/data/broadcast2_not_equal_data_int32_input1.txt",
                  "not_equal/data/broadcast2_not_equal_data_int32_input2.txt",
                  "not_equal/data/broadcast2_not_equal_data_int32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [8, 1024]
    shape_x2 = [1024]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.raw_ops.NotEqual(x=x1, y=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int32_broadcast3():
    data_files = ["not_equal/data/broadcast3_not_equal_data_int32_input1.txt",
                  "not_equal/data/broadcast3_not_equal_data_int32_input2.txt",
                  "not_equal/data/broadcast3_not_equal_data_int32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [8, 1024]
    shape_x2 = [1]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.raw_ops.NotEqual(x=x1, y=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_int32_broadcast4():
    data_files = ["not_equal/data/broadcast4_not_equal_data_int32_input1.txt",
                  "not_equal/data/broadcast4_not_equal_data_int32_input2.txt",
                  "not_equal/data/broadcast4_not_equal_data_int32_output.txt"]
    np.random.seed(23457)
    shape_x1 = [1]
    shape_x2 = [8, 1024]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", -10, 10)
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", -10, 10)

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.raw_ops.NotEqual(x=x1, y=x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")

def run():
    gen_random_data_int32_broadcast1()
    gen_random_data_int32_broadcast2()
    gen_random_data_int32_broadcast3()
    gen_random_data_int32_broadcast4()

run()