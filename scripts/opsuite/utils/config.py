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

# 工具配置信息
VERSION = "1.0"

COMMAND_SCRIPT_MAP = {
    "debug": ["msdebug"],
    "build": ["bash", "./build.sh"],
    "opprof": ["msprof", "op"],
    "deploy_op": [],
    "run_example": ["bash", "./build.sh"]
}

debug_help = f"""
作用：对算子工程进行msdebug调试
命令行举例： python opsuite.py debug
"""

build_help = f"""
作用：调用算子的编译工程脚本入口build.sh(默认)或者通过--script指定的shell或者python脚本，目的是编译出算子的二进制文件:
指定编译工程脚本入口文件场景举例： python opsuite.py build --script=../build.sh --pkg
"""

oppprof_help = f"""
作用：采集算子运行的关键性能指标，有上板(onboard)和仿真(simulator)两种运行模式:
--type=onboard/simulator （默认为onboard）
命令行举例： python opsuite.py opprof --type=simulator --output=./output_data ./build/test_aclnn_abs
"""

deploy_op_help = f"""
作用：执行算子安装包。
命令行举例： python opsuite.py deploy_op ./custom_*.run
"""

run_example_help = f"""
作用：编译并执行算子的调用者example。
命令行举例： python opsuite.py run_example abs eager，其中abs是算子名称，必填项；eager则是控制模式，可选项有eager和graph，不填默认为eager
"""

COMMAND_HELP_MAP = {
    "debug": debug_help,
    "build": build_help,
    "opprof": oppprof_help,
    "deploy_op": deploy_op_help,
    "run_example": run_example_help
}

SCRIPT_SUPPORTED_COMMANDS = {"build", "run_example"}
