#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import random
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(0)

def gen_golden_data(dtype, batchSize, targetNum):
    input_1 = np.random.rand(1500, 512, 1).astype(np.float32)
    input_1.tofile(f"input_x1.bin")
    input_2 = np.random.rand(1500, 1, 128).astype(np.float32)
    input_2.tofile(f"input_target.bin")

if name == "main":
    os.system("rm -rf *.bin")
    gen_golden_data(sys.argv[1], sys.argv[2], sys.argv[3])



