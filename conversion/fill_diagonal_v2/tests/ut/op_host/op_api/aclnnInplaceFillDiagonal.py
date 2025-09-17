#
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
#/
import torch
import numpy
from ut_golden_common import numpy_to_torch_tensor

def gen_golden(case_info: dict):
  input_desc = case_info["input_desc"]
  self_tensor = numpy_to_torch_tensor(input_desc[0]["value"])
  scalar = input_desc[1]["value"]
  wrap = bool(input_desc[2]["value"])

  if input_desc[0]["view_shape"] != input_desc[0]["storage_shape"]:
    self_tensor = self_tensor.transpose(0, 1)

  self_tensor.fill_diagonal_(scalar, wrap)
  
  if input_desc[0]["view_shape"] != input_desc[0]["storage_shape"]:
    self_tensor = self_tensor.transpose(0, 1)

  return self_tensor.numpy()