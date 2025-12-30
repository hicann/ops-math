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
from __future__ import print_function
from __future__ import division

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op, dtypes


def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

    
def read_file_txt(file_name, dtype, delim=None):
    """
    # prama1: file_name: the file which store the data
    # param2: dtype: data type
    # param3: delim: delimiter which is used to split data
    """
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


def read_file_txt_to_boll(file_name, delim=None):
    """
    # prama1: file_name: the file which store the data
    # param2: delim: delimiter which is used to split data
    """
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def write_file_txt(file_name, data, fmt="%s"):
    """
    prama1: file_name: the file which store the data
    param2: data: data which will be stored
    param3: fmt: format
    """
    if (file_name is None):
        logging.error("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


def gen_data_file(data_file, shape, dtype, rand_type, value_range):
    """
    # prama1: data_file: the file which store the generation data
    # param2: shape: data shape
    # param3: dtype: data type
    # param4: rand_type: the method of generate data, select from "randint, uniform"
    # param5: value_range: data range tuple(low, high)
    """
    low, high = value_range
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data


def gen_random_data_int8():
    data_files = ["right_shift/data/right_shift_data_input1_1.txt",
                "right_shift/data/right_shift_data_input2_1.txt",
                "right_shift/data/right_shift_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x2 = [12]
    shape_x1 = [6, 12]
    
    b = gen_data_file(data_files[1], shape_x2, np.int8, "randint", (0, 10))
    a = gen_data_file(data_files[0], shape_x1, np.int8, "randint", (0, 100))

    x2 = tf.compat.v1.placeholder(tf.int8, shape=shape_x2)
    x1 = tf.compat.v1.placeholder(tf.int8, shape=shape_x1)

    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int16():
    data_files = ["right_shift/data/right_shift_data_input1_2.txt",
                "right_shift/data/right_shift_data_input2_2.txt",
                "right_shift/data/right_shift_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [1024, 8]
    shape_x2 = [1024, 8]
    a = gen_data_file(data_files[0], shape_x1, np.int16, "randint", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.int16, "randint", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.int16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int16, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int32():
    data_files = ["right_shift/data/right_shift_data_input1_3.txt",
                "right_shift/data/right_shift_data_input2_3.txt",
                "right_shift/data/right_shift_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [1024]
    shape_x2 = [4, 1024]
    a = gen_data_file(data_files[0], shape_x1, np.int32, "randint", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.int32, "randint", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.int32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int32, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["right_shift/data/right_shift_data_input1_4.txt",
                "right_shift/data/right_shift_data_input2_4.txt",
                "right_shift/data/right_shift_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [15, 12, 30]
    shape_x2 = [15, 12, 30]
    a = gen_data_file(data_files[0], shape_x1, np.int64, "uniform", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.int64, "uniform", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.int64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.int64, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint8():
    data_files = ["right_shift/data/right_shift_data_input1_5.txt",
                "right_shift/data/right_shift_data_input2_5.txt",
                "right_shift/data/right_shift_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [6, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.uint8, "randint", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.uint8, "randint", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint8, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint16():
    data_files = ["right_shift/data/right_shift_data_input1_6.txt",
                "right_shift/data/right_shift_data_input2_6.txt",
                "right_shift/data/right_shift_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [5, 12]
    shape_x2 = [5, 12]
    a = gen_data_file(data_files[0], shape_x1, np.uint16, "randint", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.uint16, "randint", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.uint16, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint16, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint32():
    data_files = ["right_shift/data/right_shift_data_input1_7.txt",
                "right_shift/data/right_shift_data_input2_7.txt",
                "right_shift/data/right_shift_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [3, 12]
    shape_x2 = [12]
    a = gen_data_file(data_files[0], shape_x1, np.uint32, "randint", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.uint32, "randint", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint32, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def gen_random_data_uint64():
    data_files = ["right_shift/data/right_shift_data_input1_8.txt",
                "right_shift/data/right_shift_data_input2_8.txt",
                "right_shift/data/right_shift_data_output1_8.txt"]
    np.random.seed(23457)
    shape_x1 = [15, 12, 30]
    shape_x2 = [15, 12, 30]
    a = gen_data_file(data_files[0], shape_x1, np.uint64, "uniform", (0, 100))
    b = gen_data_file(data_files[1], shape_x2, np.uint64, "uniform", (0, 10))

    x1 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.uint64, shape=shape_x2)
    re = tf.bitwise.right_shift(x1, x2, name="right_shift")
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1: a, x2: b})
    write_file_txt(data_files[2], data, fmt="%s")


def run():
    gen_random_data_int8()
    gen_random_data_int16()
    gen_random_data_int32()
    gen_random_data_int64()
    gen_random_data_uint8()
    gen_random_data_uint16()
    gen_random_data_uint32()
    gen_random_data_uint64()


if __name__ == '__main__':
    os.system("mkdir -p right_shift/data")
    run()