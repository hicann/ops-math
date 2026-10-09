# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/floor_mod_data DESTINATION ${CMAKE_BINARY_DIR}/tests/ut/op_kernel)
    # 算子自己的tiling文件路径
    set(floor_mod_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/floor_mod_tiling.cpp
    )

    # 使用AddOpTestCase
    # param1：算子名称
    # param2：soc版本
    # param3：自定义编译选项，这里不需要特殊的宏定义来选择代码路径，
    #        但如果代码中有 #ifdef 依赖 DTYPE_X 等宏，需要在这里传入。
    #        floor_mod kernel 是模板，在 cpp 测试文件中直接实例化，这里留空或根据需要添加。
    # param4：该算子依赖的所有tiling源码文件

    AddOpTestCase(floor_mod "ascend910b" "-DHIGH_PERFORMANCE=1" "${floor_mod_tiling_files}")
endif()
