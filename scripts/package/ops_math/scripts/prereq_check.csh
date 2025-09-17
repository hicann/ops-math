#!/bin/csh
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set FILE_NOT_EXIST="0x0080"

set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
echo "[OpsMath][${cur_date}][INFO]: Start pre installation check of ops_math module."
python3 --version >/dev/null 2>&1
if ($status != "0") then
set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
exit 0
endif
set python_version=`python3 --version`
set idx=`echo ${python_version} | awk '{print index($0, "3.7.5")}'`
if ($idx > 0) then
set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
exit 0
else
set cur_date=`date +"%Y-%m-%d %H:%M:%S"`
exit 0
endif
