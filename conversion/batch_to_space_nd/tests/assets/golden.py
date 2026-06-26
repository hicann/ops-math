#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


__golden__ = {
    "kernel": {
        "batch_to_space_nd": "batch_to_space_nd_golden"
    }
}


def _trans_nc1hwc0_to_nhwc(data):
    in_shape = data.shape
    axis_n = in_shape[0]
    axis_c1 = in_shape[1]
    axis_h = in_shape[2]
    axis_w = in_shape[3]
    axis_c0 = in_shape[4]
    tmp_input_tensor = np.transpose(data, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_h, axis_w, axis_c1 * axis_c0)
    return tmp_input_tensor


def _trans_nhwc_to_nc1hwc0(data):
    in_shape = data.shape
    axis_n = in_shape[0]
    axis_h = in_shape[1]
    axis_w = in_shape[2]
    axis_c = in_shape[3]
    axis_c0 = 16
    axis_c1 = (axis_c + axis_c0 - 1) // axis_c0
    tmp_input_tensor = data.reshape(axis_n, axis_h, axis_w, axis_c1, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 3, 1, 2, 4))
    return tmp_input_tensor


def _shape_6d_2_5d(data):
    in_shape = data.shape
    axis_d = in_shape[0]
    axis_c1 = in_shape[1]
    axis_h = in_shape[2]
    axis_w = in_shape[3]
    axis_n = in_shape[4]
    axis_c0 = in_shape[5]
    axis_n = axis_n * axis_d
    axis_c = axis_c1 * axis_c0
    tmp_input_tensor = data.reshape(axis_n, axis_c1, axis_h, axis_w, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_h, axis_w, axis_c)
    return tmp_input_tensor


def _shape_5d_2_6d(data):
    in_shape = data.shape
    axis_n = in_shape[0]
    axis_h = in_shape[1]
    axis_w = in_shape[2]
    axis_c = in_shape[3]
    axis_c0 = 16
    axis_c1 = (axis_c + axis_c0 - 1) // axis_c0
    axis_d = 1
    tmp_input_tensor = data.reshape(axis_d, axis_n, axis_h, axis_w, axis_c1, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 4, 2, 3, 1, 5))
    return tmp_input_tensor


def batch_to_space_nd_golden(x, block_shape, crops, **kwargs):
    '''
    Kernel golden for batch_to_space_nd.
    All the parameters follow @batch_to_space_nd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    input_format = kwargs.get('input_formats', ['ND'])[0]
    
    if input_format == 'NC1HWC0':
        x = _trans_nc1hwc0_to_nhwc(x)
    elif input_format == "NDC1HWC0":
        x = _shape_6d_2_5d(x)
    
    tensor_x = tf.placeholder(x.dtype, shape=x.shape)
    block_shape_tensor = tf.constant(block_shape.tolist())
    crops_tensor = tf.constant(crops.tolist())
    out = tf.batch_to_space_nd(tensor_x, block_shape_tensor, crops_tensor)
    
    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: x})
    
    if input_format == "NC1HWC0":
        res = _trans_nhwc_to_nc1hwc0(res)
    elif input_format == "NDC1HWC0":
        res = _shape_5d_2_6d(res)
    
    return res