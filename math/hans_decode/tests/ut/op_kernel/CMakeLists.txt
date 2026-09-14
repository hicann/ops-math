# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 需要将Tiling依赖的文件添加到CMakeLists.txt中
    # set(elewise_common_tiling_files
    #         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
    #         )
    # 算子自己的tiling文件路径
    set(hans_decode_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/hans_decode_tiling.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："ascend950;ascend910b"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(hans_decode "ascend910b" "-D__CCE_UT_TEST__" "${hans_decode_tiling_files}")
    if (ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
        add_subdirectory(ascend950)
    endif()
endif()

# add_all_ut_sources loads this file for both host and kernel UT builds.
if (UT_TEST_ALL OR OP_HOST_UT)
    if (TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        target_include_directories(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${ASCEND_DIR}/pkg_inc)
        # The HANS platform assertions call APIs outside the dynamically loaded host library.
        target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj INTERFACE tiling_api)
    endif()
endif()
