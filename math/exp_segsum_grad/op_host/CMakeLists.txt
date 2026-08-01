# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_modules_sources(OPTYPE exp_segsum_grad ACLNNTYPE aclnn_exclude)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950" AND TARGET ${OPHOST_NAME}_tiling_obj)
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/exp_segsum_grad_tiling_arch35.cpp)
    get_target_property(EXP_SEGSUM_GRAD_TILING_SRCS ${OPHOST_NAME}_tiling_obj SOURCES)
    list(REMOVE_ITEM EXP_SEGSUM_GRAD_TILING_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/exp_segsum_grad_tiling.cpp")
    set_property(TARGET ${OPHOST_NAME}_tiling_obj PROPERTY SOURCES ${EXP_SEGSUM_GRAD_TILING_SRCS})
endif()
