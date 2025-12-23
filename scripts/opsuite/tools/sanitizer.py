#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import logging
import subprocess
from utils import script

logger = logging.getLogger(__name__)


def run(input_options, cmd_name):
    # 自动添加--tool=memcheck 和 --leak-check=yes选项
    options = input_options
    if not any("--tool" in str(item) for item in input_options):
        options.append("--tool=memcheck")
    if not any("--leak-check" in str(item) for item in input_options):
        options.append("--leak-check=yes")
    cmd_with_opt = cmd_name + options
    try:
        logger.info(f"正在执行 %s", " ".join(cmd_with_opt))
        result = subprocess.run(cmd_with_opt, capture_output=False, text=True)
        if result.returncode != 0:
            logger.error(f"执行%s 失败，请查看日志\n", " ".join(cmd_with_opt))
            return
    except Exception as e:
        logger.error(f"执行 %s 失败： {e}", " ".join(cmd_with_opt))
        return
