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

import sys
import logging
from . import config

logger = logging.getLogger(__name__)


def parse_option_value(option):
    """
    解析命令行参数中的 --option或者--option 参数
    返回脚本名称，如果未找到则返回空字符串
    """
    option = "--" + option
    option_value = ""
    # 先检查 --script=value 格式
    for arg in sys.argv:
        if arg.startswith(option + "="):
            option_value = arg.split("=", 1)[1]
            return option_value
    # 再检查 --script value 格式
    if option in sys.argv:
        index = sys.argv.index(option)
        if index + 1 < len(sys.argv):
            option_value = sys.argv[index + 1]
    return option_value


def get_script_name(cmd_name, name):
    script_name = config.COMMAND_SCRIPT_MAP.get(cmd_name)        
    if name != "" and cmd_name in config.SCRIPT_SUPPORTED_COMMANDS:
        if name.lower().endswith(".sh"):
            script_name = ["bash", name]
        elif name.lower().endswith(".py"):
            script_name = [sys.executable, name]
        else:
            logger.error("不支持的脚本入口格式： %s，当前只支持.sh和.py", name)
            return ""
    return script_name


def remove_opt_from_options(options, opt):
    """
    从options中移除与script_name相关的参数，支持--script在任何位置
    """
    result = []
    i = 0
    while i < len(options):
        current_option = options[i]
        
        # 情况1: 当前选项是 opt，下一个是opt的值
        if current_option == opt and i + 1 < len(options):
            i += 2  # 跳过这两个元素
            continue
        # 情况2: 当前选项是 opt=值 格式
        elif current_option.startswith(opt + '='):
            i += 1
            continue
        # 其他情况：保留该选项
        else:
            result.append(current_option)
            i += 1
    return result