# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(parallel_concat_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/parallel_concat_tiling_arch35.cpp
        )
    # parallel_concat_struct.h includes "ascendc/host_api/tiling/template_argument.h"
    # (resolved under <CANN>/<arch>-linux/include) and the force-included generated
    # tiling header includes "kernel_tiling/kernel_tiling.h"
    # (resolved under <CANN>/<arch>-linux/ascendc/include/highlevel_api).
    AddOpTestCase(parallel_concat "ascend950"
                  "-I${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include -I${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/highlevel_api"
                  "${parallel_concat_tiling_files}")
endif()
