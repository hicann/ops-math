# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_API_UT)
    # 直接添加测试源文件，绕过 add_modules_ut_sources 中的 ASCEND_OP_NAME 列表查找问题
    # 先调用 add_opapi_ut_modules 初始化 cases_obj 目标
    add_opapi_ut_modules(${OP_API_MODULE_NAME})
    file(GLOB OPAPI_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_one_hot.cpp)
    target_sources(${OP_API_MODULE_NAME}_cases_obj PRIVATE ${OPAPI_CASES_SRC})
endif()
