/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file population_count.cpp
 * @brief L0 API implementation for PopulationCount
 */

#include "population_count.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(PopulationCount);

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT16, DataType::DT_UINT16
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "PopulationCount: only Ascend950 (arch35/DAV_3510) is supported, got npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x->GetDataType(), X_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "PopulationCount: x dtype %d not supported (must be INT16/UINT16).",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static const aclTensor* PopulationCountAiCore(
    const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(PopulationCountAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PopulationCount,
        OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PopulationCountAiCore failed."),
             return nullptr);
    return out;
}

/**
 * @brief L0 entry: InferShape + IsAiCoreSupport + AllocTensor + Kernel dispatch.
 *
 * Output shape = input shape; output dtype is fixed to UINT8.
 */
const aclTensor* PopulationCount(const aclTensor* x, aclOpExecutor* executor)
{
    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    // Output: same shape as x, dtype fixed UINT8
    const aclTensor* out = executor->AllocTensor(x->GetViewShape(), DataType::DT_UINT8);
    // Explicit nullptr check (AllocTensor can fail under extreme UB pressure)
    OP_CHECK_NULL(out, return nullptr);

    return PopulationCountAiCore(x, out, executor);
}

} // namespace l0op
