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

import logging
import os
from datetime import datetime, timezone
from utils import tensor_dump


logger = logging.getLogger(__name__)


def run(options, cmd):
    if not options or len(options) == 0:
        logger.error("wrong parameter")
        return

    op_name = options[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    output_folder = os.path.join(os.path.abspath("."), "dump_" + timestamp)
    os.makedirs(output_folder)
    dumpbin_file = os.path.join(output_folder, "dumpbin")
    cmd += options
    tensor_dump.build_and_dump(cmd, op_name, dumpbin_file, output_folder)
