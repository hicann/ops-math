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

import subprocess
import re
import logging

logger = logging.getLogger(__name__)


# 获取工具执行-h时得到的帮助内容
def get_tools_help(tool_name) -> str:
    try:
        result = subprocess.run(tool_name + ["-h"], capture_output=True, text=True, timeout=100)
        if result.returncode == 0:
            # 将帮助信息中的时间信息清除，只保留帮助信息本身
            help_without_time = re.sub(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ', '', result.stdout.strip())
            # 将帮助信息中的"bash build.sh"或者"./build.sh"替换为"python opsuite.py build"
            return re.sub(r'(bash\s+|\.\/)?build\.sh', 'python opsuite.py build', help_without_time)
        else:
            logger.info(f"无法获取 %s 的详细帮助信息: \n{result.stderr.strip()}", " ".join(tool_name))
            return ""
    except Exception as e:
        logger.info(f"执行 %s -h 失败： {e}", " ".join(tool_name))
        return ""


# 重新定义help的显示逻辑
def get_help(tool, origin_help_info):
    # 获取工具本身的帮助内容
    tool_help = get_tools_help(tool)

    #构造自定义帮助文本
    help_info = f"""
    {origin_help_info}
    {tool_help}
    """
    return help_info
