# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    # HistogramV2 has two tiling templates selected by SoC (IsRegbaseSocVersion):
    #   ascend950               -> arch35 (HistogramV2SimtTiling)
    #   ascend310p/910_93/910b  -> arch32 (HistogramV2MembaseTiling)
    # The tiling UT is compiled once per BUILD_SOC_VERSION, and tiling_context_faker sets the SoC
    # from BUILD_SOC_VERSION, so only the matching template is capable. Gate the arch-specific
    # tiling cases by ASCEND_COMPUTE_UNIT so each set is compiled only where its template runs.
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} TILING_DIR arch35)
    else()
        add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} TILING_DIR arch32)
    endif()
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
