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

import numpy as np
import sys

caseflat_params = [326, 326, 0, 326, 3, 160, 6, 2101, 128, 59, 64, 64, 64, 64, 0, 0, 0, 0]
case0_params = [326, 326, 0, 326, 3, 160, 6, 2101, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
case1_params = [326, 326, 0, 326, 3, 160, 6, 2101, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0]
case2_params = [64, 64, 0, 64, 2, 32, 32, 2405, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0]
case3_params = [128, 128, 0, 128, 2, 64, 64, 2104, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0]
case4_params = [326, 326, 0, 326, 3, 160, 6, 2101, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
case5_params = [64, 64, 0, 64, 2, 32, 32, 2405, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
    "caseflat": caseflat_params,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()