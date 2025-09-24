# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in isfinite")
# ISFINITE operation sources
file(GLOB ISFINITE_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cce")

# set(ISFINITE_SOURCES ${ISFINITE_CPP_HEADER} ${ISFINITE_CPP_SOURCES} ${ISFINITE_NPU_SOURCES})
set(ISFINITE_SOURCES ${ISFINITE_NPU_SOURCES})
# Mark .cce files with special properties
set_source_files_properties(
    ${ISFINITE_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync"
)

# Create object library
add_library(isfinite_objects OBJECT ${ISFINITE_SOURCES})

target_compile_options(isfinite_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(isfinite_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
