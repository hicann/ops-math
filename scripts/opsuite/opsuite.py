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
import sys
import logging
from utils import parser_args, script, config
from utils.logger import setup, get_logger
from tools import build, debug, opprof, deploy_op, run_example, sanitizer


def main():
    log_level = logging.INFO
    setup(
        app_name="opsuite",
        log_to_console=True,
        log_level=log_level
    )
    logger = get_logger(__name__)

    current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(current_dir + "/../../")

    script_name = script.parse_option_value("script")
    parser = parser_args.create_parser(script_name)    
    args, options = parser.parse_known_args()
    options = script.remove_opt_from_options(options, '--script')
    if args.command == "build":
        build.run(options, args.script, args.debug, args.sanitizer)
    elif args.command == "deploy_op":
        deploy_op.run(options)
    elif args.command == "run_example":
        run_example.run(options, script.get_script_name(args.command, script_name))
    elif args.command == "debug":
        debug.run(options, script.get_script_name(args.command, script_name))
    elif args.command == "opprof":
        opprof.run(options, config.COMMAND_SCRIPT_MAP.get(args.command), args.type)
    elif args.command == "sanitizer":
        sanitizer.run(options, config.COMMAND_SCRIPT_MAP.get(args.command))
    elif args.version:
        logger.info("算子工具一站式平台版本: %s", config.VERSION)
    else:
        logger.error("命令行参数无效，请参考如下帮助信息：")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()