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
import argparse
from functools import partial
import logging
from . import custom_help, script, config

logger = logging.getLogger(__name__)


# 重写parser的format_help方法
class EnhanceHelpFormater(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog, cmd_name=None, script_name=None, option=None):
        super().__init__(prog)
        self._cmd_name = cmd_name
        self._script_name = script_name
        self._option = option
    
    def format_help(self) -> str:
        original_help = super().format_help()
        opt = self._option
        if self._cmd_name == "run_example":
            opt.append("--run_example")
        extend_help = config.COMMAND_HELP_MAP.get(self._cmd_name) + original_help
        # "run_example"和"deploy_op"是扩展命令，不需要执行工具本身的帮助命令
        if self._cmd_name != "run_example" and self._cmd_name != "deploy_op":
            extend_help += custom_help.get_tools_help(self._script_name + opt)
        return extend_help


def create_parser(name):
    parser = argparse.ArgumentParser(description="算子工具一站式平台")
    parser.add_argument("--version", "-v", action="store_true", help="显示算子工具一站式平台的版本号")
    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    for cmd_name, cmd_help in config.COMMAND_HELP_MAP.items():
        # 如果通过--script指定了脚本，则使用用户定义脚本
        script_name = script.get_script_name(cmd_name, name)
        options = script.remove_opt_from_options(sys.argv, "script")
        formatter_class = partial(EnhanceHelpFormater, cmd_name=cmd_name, script_name=script_name, option=options[2:])
        subparser = subparsers.add_parser(cmd_name, help=cmd_help, formatter_class=formatter_class)
        if cmd_name in config.SCRIPT_SUPPORTED_COMMANDS:
            script_help = "指定构建脚本入口，当前只有build和run_example命令支持此参数"
            subparser.add_argument("--script", default=config.COMMAND_SCRIPT_MAP.get(cmd_name), help=script_help)
        if cmd_name == "opprof":
            type_help = "--type=onboard/simulator （运行模式onboard(上板)和simulator(仿真)两种运行模式，默认为onboard）"
            subparser.add_argument("--type", default="onboard", help=type_help)
        if cmd_name == "build":
            subparser.add_argument("--debug", action="store_true", help="调试构建场景，自动添加所需构建参数")
            subparser.add_argument("--sanitizer", action="store_true", help="mssanitizer检测构建场景，自动添加所需构建参数")
    return parser