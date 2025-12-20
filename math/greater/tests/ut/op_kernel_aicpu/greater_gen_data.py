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

tf.compat.v1.disable_eager_execution()


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

# prama1: data_file: the file which store the generation data
# param2: shape: data shape
# param3: dtype: data type
# param4: rand_type: the method of generate data, select from "randint, uniform"
# param5: data lower limit
# param6: data upper limit
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
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

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_float():
    data_files=["greater/data/greater_data_input1_1.txt",
                "greater/data/greater_data_input2_1.txt",
                "greater/data/greater_data_output1_1.txt"]
    np.random.seed(23457)
    shape_x1 = []
    shape_x2 = [4]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.math.greater(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float2():
    data_files=["greater/data/greater_data_input1_2.txt",
                "greater/data/greater_data_input2_2.txt",
                "greater/data/greater_data_output1_2.txt"]
    np.random.seed(23457)
    shape_x1 = [4]
    shape_x2 = []
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.math.greater(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float0():
    data_files=["greater/data/greater_data_input1_0.txt",
                "greater/data/greater_data_input2_0.txt",
                "greater/data/greater_data_output1_0.txt"]
    np.random.seed(23457)
    shape_x1 = []
    shape_x2 = []
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.math.greater(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float3():
    data_files=["greater/data/greater_data_input1_3.txt",
                "greater/data/greater_data_input2_3.txt",
                "greater/data/greater_data_output1_3.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2]
    shape_x2 = [2,1,1,1,1,1,1,2]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 100)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 0, 100)

    x1 = tf.compat.v1.placeholder(tf.float32, shape=shape_x1)
    x2 = tf.compat.v1.placeholder(tf.float32, shape=shape_x2)
    re = tf.math.greater(x1, x2)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(re, feed_dict={x1:a, x2:b})
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float4():
    data_files=["greater/data/greater_data_input1_4.txt",
                "greater/data/greater_data_input2_4.txt",
                "greater/data/greater_data_output1_4.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2]
    shape_x2 = [2,1,1,1,1,1,2,1]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 50)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 51, 100)

    data = np.zeros(shape_x1, dtype=bool)
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float5():
    data_files=["greater/data/greater_data_input1_5.txt",
                "greater/data/greater_data_input2_5.txt",
                "greater/data/greater_data_output1_5.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2]
    shape_x2 = [1,1,1,1,2,1,2,1]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 50)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 51, 100)

    data = np.zeros(shape_x1, dtype=bool)
    write_file_txt(data_files[2], data, fmt="%s")
def gen_random_data_float6():
    data_files=["greater/data/greater_data_input1_6.txt",
                "greater/data/greater_data_input2_6.txt",
                "greater/data/greater_data_output1_6.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2]
    shape_x2 = [2,2,2,1,2,1,2,1]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 51, 100)

    data = np.zeros(shape_x1, dtype=bool)
    write_file_txt(data_files[2], data, fmt="%s")
def gen_random_data_float7():
    data_files=["greater/data/greater_data_input1_7.txt",
                "greater/data/greater_data_input2_7.txt",
                "greater/data/greater_data_output1_7.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2]
    shape_x2 = [1,1,2,1,2,1,2,1]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 51, 100)

    data = np.zeros(shape_x1, dtype=bool)
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float8():
    data_files=["greater/data/greater_data_input1_8.txt",
                "greater/data/greater_data_input2_8.txt",
                "greater/data/greater_data_output1_8.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2]
    shape_x2 = [2,1,2,1,2,1,2,1]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 51, 100)

    data = np.zeros(shape_x1, dtype=bool)
    write_file_txt(data_files[2], data, fmt="%s")

def gen_random_data_float9():
    data_files=["greater/data/greater_data_input1_9.txt",
                "greater/data/greater_data_input2_9.txt",
                "greater/data/greater_data_output1_9.txt"]
    np.random.seed(23457)
    shape_x1 = [2,2,2,2,2,2,2,2,2]
    shape_x2 = [1,2,1,2,1,2,1,2,1]
    a = gen_data_file(data_files[0], shape_x1, np.float32, "uniform", 0, 10)
    b = gen_data_file(data_files[1], shape_x2, np.float32, "uniform", 51, 100)

    data = np.zeros(shape_x1, dtype=bool)
    write_file_txt(data_files[2], data, fmt="%s")
def run():
    gen_random_data_float0()
    gen_random_data_float()
    gen_random_data_float2()
    gen_random_data_float3()
    gen_random_data_float4()
    gen_random_data_float5()
    gen_random_data_float6()
    gen_random_data_float7()
    gen_random_data_float8()
    gen_random_data_float9()

if __name__ == '__main__':
    run()
