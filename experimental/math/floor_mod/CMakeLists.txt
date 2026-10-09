# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(OPTYPE floor_mod ACLNNTYPE aclnn_exclude DEPENDENCIES squeeze)

# INT32 floor-mod uses the FP32 vector implementation selected for performance.
# Do not expose a high-precision binary: the runtime otherwise prefers it and
# bypasses the intended fast path for every dtype.
add_kernel_sources(
    COMPUTE_UNITS ascend910b
    IMPL_MODE high_performance
)

# Keep graph_base loaded for the Host UT shared library under --as-needed linkers.
if(OP_HOST_UT AND TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj INTERFACE
        "-Wl,--no-as-needed,$<TARGET_FILE:graph_base>,--as-needed")
endif()
