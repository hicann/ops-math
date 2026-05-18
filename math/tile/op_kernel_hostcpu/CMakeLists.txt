# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# This file enables constant folding (hostcpu) for tile operator.
# The CMake function add_aicpu_host_kernel_modules in func.cmake will:
# 1. Check if this CMakeLists.txt exists (done by checking the parent dir)
# 2. Compile op_kernel_aicpu/tile_aicpu.cpp as x86 OBJECT
# 3. Link it into libmath_constant_folding_ops.so for OPP constant folding