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


def run(options, script_name, debug, sanitizer):
    if debug:
        options = create_debug_ops(options)
    elif sanitizer:
        options = create_sanitizer_ops(options)
    # 默认添加"--pkg选项""
    options = add_pkg_option(options)
    # 默认添加"--vender_name选项""
    options = add_vender_name(options)
    # 对构建选项做去重处理
    unique_build_ops = list(dict.fromkeys(options))
    logger.info(f"正在执行 %s %s", " ".join(script_name), " ".join(unique_build_ops))
    result = subprocess.run(script_name + unique_build_ops, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"执行构建脚本 %s %s 失败", " ".join(script_name), " ".join(unique_build_ops))
        return


def create_debug_ops(options):
    logger.info("调试构建场景，正在对编译选项进行整理，以满足调试的要求")
    ops_option = "--ops="
    if not any(ops_option in str(item) for item in options):
        logger.info("未指定算子--ops=，则所有算子都添加调试编译选项")
    options = [opt for opt in options if opt not in ["-O0", "-O1", "-O2", "-O3", "--debug"]]
    options.extend(["-O0", "--debug"])
    return options


def create_sanitizer_ops(options):
    logger.info("sanitizer检测构建场景，正在对编译选项进行整理，以满足sanitizer检测的要求")
    ops_option = "--ops="
    if not any(ops_option in str(item) for item in options):
        logger.info("未指定算子--ops=，则所有算子都添加bisheng的sanitizer编译选项")
    options = [opt for opt in options if opt not in ["-O0", "--debug", "--sanitizer"]]
    options.append("--extra_compile_options='--cce-enable-sanitizer'")
    return options


def add_pkg_option(options):
    # 测试过程不能添加--pkg选项，如果是测试场景，需要删除--pkg选项
    logger.info("正在检测是否需要自动添加--pkg选项")
    test_pattern_check = any('_test' in str(item) for item in options)
    ophost_check = any(opt in options for opt in ["-u", "--ophost", "--opapi", "--opgraph"])
    if test_pattern_check or ophost_check:
        options = [item for item in options if "--pkg" not in item]
        return options
    options.append("--pkg")
    return options


def add_vender_name(options):
    # 只有指定了算子名称的场景下，才需要添加--vendor_name选项
    logger.info("正在检测是否需要自动添加--vendor_name选项")
    ops_option = "--ops="
    if any(ops_option in str(item) for item in options):
        options.append("--vendor_name=custom")
    return options