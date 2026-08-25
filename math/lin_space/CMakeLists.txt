# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(OPTYPE lin_space ACLNNTYPE aclnn_exclude HOSTCPU TRUE)

if(ENABLE_TEST AND (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    list(FIND ASCEND_OP_NAME lin_space LIN_SPACE_INDEX)
    if("${ASCEND_OP_NAME}" STREQUAL "" OR NOT LIN_SPACE_INDEX EQUAL -1)
        add_aicpu_op_test_case(lin_space)
    endif()
endif()
