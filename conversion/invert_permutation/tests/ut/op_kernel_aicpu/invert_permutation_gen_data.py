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
from tensorflow.python.ops import array_ops

tf.compat.v1.disable_eager_execution()


"""
prama1: file_name: the file which store the data
param2: data: data which will be stored
param3: fmt: format
"""


def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')


"""
prama1: file_name: the file which store the data
param2: dtype: data type
param3: delim: delimiter which is used to split data
"""


def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()


"""
prama1: file_name: the file which store the data
param2: delim: delimiter which is used to split data
"""


def read_file_txt_to_boll(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)


def gen_data_file(data_file, dtype, len):
    rand_data = []
    for i in range(0, len):
        t = len - i - 1
        rand_data.append(t)
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
    data_files = ["invert_permutation/data/input_int32.txt", 
                "invert_permutation/data/output_int32.txt"]
    shape_x = [1024]
    len = 1024
    a = gen_data_file(data_files[0], np.int32, len)
    x = tf.compat.v1.placeholder(tf.int32, shape=shape_x)
    re = array_ops.invert_permutation(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_int32_para():
    data_files = ["invert_permutation/data/input_int32_para.txt", 
                "invert_permutation/data/output_int32_para.txt"]
    shape_x = [128 * 1024]
    len = 128 * 1024
    a = gen_data_file(data_files[0], np.int32, len)
    x = tf.compat.v1.placeholder(tf.int32, shape=shape_x)
    re = array_ops.invert_permutation(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_int64():
    data_files = ["invert_permutation/data/input_int64.txt", 
                "invert_permutation/data/output_int64.txt"]
    shape_x = [1024]
    len = 1024
    a = gen_data_file(data_files[0], np.int64, len)
    x = tf.compat.v1.placeholder(tf.int64, shape=shape_x)
    re = array_ops.invert_permutation(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def gen_random_data_int64_para():
    data_files = ["invert_permutation/data/input_int64_para.txt", 
                "invert_permutation/data/output_int64_para.txt"]
    shape_x = [128 * 1024]
    len = 128 * 1024
    a = gen_data_file(data_files[0], np.int64, len)
    x = tf.compat.v1.placeholder(tf.int64, shape=shape_x)
    re = array_ops.invert_permutation(x)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x: a})
    write_file_txt(data_files[1], data, fmt="%s")


def run():
    gen_random_data_int32()
    gen_random_data_int32_para()
    gen_random_data_int64()
    gen_random_data_int64_para()


if __name__ == '__main__':
    run()