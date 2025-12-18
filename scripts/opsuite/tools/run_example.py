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

import logging
import subprocess

logger = logging.getLogger(__name__)


def run(options, script_name):
    option = ["--run_example"]
    if len(options) < 1 or len(options) > 2:
        error_message = "run_example命令后至少有一个算子名称作为参数，最多有两个参数"
        logger.error(error_message)
        return
    ops_name = options[0]
    option.append(ops_name)
    mode = "eager"
    if len(options) > 1 and options[1] == "graph":
        mode = options[1]
    option.append(mode)
    option.append("cust")
    option.append("--vendor_name=custom")

    try:
        logger.info(f"正在执行 %s %s", " ".join(script_name), " ".join(option))
        result = subprocess.run(script_name + option, capture_output=False, text=True)
        if result.returncode != 0:
            logger.error(f"执行构建脚本{script_name}  {option} 失败，请查看日志\n")
            return
    except Exception as e:
        logger.error(f"执行{script_name} {option} 失败： {e}")
        return
