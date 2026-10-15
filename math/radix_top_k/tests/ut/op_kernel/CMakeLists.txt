# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(radix_top_k_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/radix_top_k_tiling.cpp
    )
    AddOpTestCase(radix_top_k "ascend910b"
        "-DDTYPE_X=half -D__CCE_UT_TEST__ -DTestUtDefaultTilingStruct=RadixTopKTilingData"
        "${radix_top_k_tiling_files}")

    if(TARGET ${OP_KERNEL_MODULE_NAME}_common_obj)
        target_compile_definitions(${OP_KERNEL_MODULE_NAME}_common_obj
            PRIVATE BUILD_SOC_VERSION=ascend910b)
        message(STATUS "[radix_top_k UT] BUILD_SOC_VERSION=ascend910b -> ${OP_KERNEL_MODULE_NAME}_common_obj")
    else()
        cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL target_compile_definitions
            ${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE BUILD_SOC_VERSION=ascend910b)
        message(STATUS "[radix_top_k UT] BUILD_SOC_VERSION=ascend910b -> ${OP_KERNEL_MODULE_NAME}_common_obj (deferred)")
    endif()
endif()
