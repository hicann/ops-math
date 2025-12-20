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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

# prama1: file_name: the file which store the data
# param2: data: data which will be stored
def write_file_txt(file_name, data):
    if np.iscomplexobj(data):
    	fmt="(%s,%s)"
    else:
    	fmt="%s"
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


# param1: shape: data shape
# param2: dtype: data type
# param3: rand_type: the method of generate data, select from "randint, uniform"
# param4: data lower limit
# param5: data upper limit
def gen_data(shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif issubclass(dtype, np.complexfloating):
        rand_data = np.random.uniform(low, high, size=shape) + 1j * np.random.uniform(low, high, size=shape)
    else:
        rand_data = np.random.uniform(low, high, size=shape)
    return np.array(rand_data, dtype=dtype)

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    return session_config

def gen_random_data(nptype, shape, rand_type, low, high):
    np.random.seed(23457)
    x_batch = gen_data(shape, nptype, rand_type, low, high)
    write_file_txt("atan/data/atan_data_input_0_" + nptype.__name__ + ".txt", x_batch)
    x_batch.tofile("atan/data/atan_data_input_0_" + nptype.__name__ + ".bin")
    x_placeholder = tf.compat.v1.placeholder(nptype, shape=shape)

    write_file_txt("atan/data/atan_data_" + nptype.__name__ + "_dim.txt", np.array([len(shape)]))
    write_file_txt("atan/data/atan_data_" + nptype.__name__ + "_shape.txt", np.array(shape))
    result = tf.atan(x_placeholder)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(result, feed_dict={x_placeholder: x_batch})
    write_file_txt("atan/data/atan_data_output_" + nptype.__name__ + "_expect.txt", data)
    data.tofile("atan/data/atan_data_output_" + nptype.__name__ + "_expect.bin")

def run():
    gen_random_data(np.float16, [12, 130, 3], "uniform", -100, 100)
    gen_random_data(np.float32, [12, 130, 3], "uniform", -100, 100)
    gen_random_data(np.float64, [12, 130, 3], "uniform", -100, 100)

tf.compat.v1.disable_eager_execution()
run()