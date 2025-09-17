#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import os
import numpy as np
import re
import torch

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def segsum(x):
    T = x.size(-1)
    x = torch.unsqueeze(x, -1)
    shape_list = list(x.shape)
    shape_list[-1] = T
    x = x.expand(*shape_list)
    mask = torch.ones(T, T, device=x.device, dtype=bool)
    mask = torch.tril(mask, diagonal=-1)
    mask = ~mask
    x = x.masked_fill(mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.ones(T, T, device=x.device, dtype=bool)
    mask = torch.tril(mask, diagonal=0)
    mask = ~mask
    x_segsum = x_segsum.masked_fill(mask, -torch.inf)
    x_segsum = torch.exp(x_segsum)
    return x_segsum

def gen_data_and_golden(input_shape_str, output_size_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,

    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)

    size = np.prod(input_shape)
    tmp_input = np.random.random(size).reshape(input_shape).astype(np_type)
    x_tensor = torch.tensor(tmp_input, dtype=torch.float32)
    y_golden = segsum(x_tensor)
    tmp_golden = np.array(y_golden).astype(np_type)

    tmp_input.astype(np_type).tofile(f"{d_type}_input_segsum.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_segsum.bin")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
