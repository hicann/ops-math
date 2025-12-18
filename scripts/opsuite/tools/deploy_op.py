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

import os
import stat
import logging
import subprocess

logger = logging.getLogger(__name__)


def run(options):
    if not options or len(options) == 0:
        error_message = "未指定run包路径"
        logger.error(error_message)
        return

    run_file = os.path.abspath(options[0])
    if not os.path.isfile(run_file):
        error_message = f"{run_file} 不存在，请检查路径是否正确"
        logger.error(error_message)
        return

    if not run_file.endswith('.run'):
        error_message = f"{run_file} 不是一个 .run 包，请检查文件类型"
        logger.error(error_message)
        return

    if not os.access(run_file, os.X_OK):
        logger.warning(f"{run_file} 没有执行权限，尝试自动添加...")
        try:
            current_stat = os.stat(run_file)
            os.chmod(run_file, current_stat.st_mode | stat.S_IXUSR)

            if os.access(run_file, os.X_OK):
                logger.info(f"成功为 {run_file} 添加了执行权限。")
            else:
                error_message = f"为 {run_file} 添加执行权限失败，文件可能仍然不可执行。"
                logger.error(error_message)
                return

        except OSError as e:
            # 如果 os.chmod 调用失败，会抛出 OSError
            error_message = f"尝试为 {run_file} 添加执行权限时失败: {e.strerror}"
            logger.error(error_message)
            return

    cmd = [run_file]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        error_message = f"算子部署失败，返回码{e.returncode}"
        logger.error(error_message)

    logger.info("部署成功")