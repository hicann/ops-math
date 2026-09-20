/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "im2col.h"
#include <limits>
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Im2col);

static const string PADDING_MODE = "CALCULATED";

static bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool CalculateOutputDim(int64_t input, int64_t kernel, int64_t dilation, int64_t paddingBefore,
                               int64_t paddingAfter, int64_t stride, int64_t& output)
{
    const __int128 effectiveKernel = static_cast<__int128>(dilation) * (kernel - 1) + 1;
    const __int128 numerator = static_cast<__int128>(input) + paddingBefore + paddingAfter - effectiveKernel;
    if (numerator < 0) {
        return false;
    }
    const __int128 result = numerator / stride + 1;
    if (result <= 0 || result > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    output = static_cast<int64_t>(result);
    return true;
}

static bool Im2colInferShape(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                             const aclIntArray* padding, const aclIntArray* stride, op::Shape& outShape)
{
    int64_t outH = 0;
    int64_t outW = 0;
    if (!CalculateOutputDim(self->GetViewShape().GetDim(2), (*kernelSize)[0], (*dilation)[0], (*padding)[0],
                            (*padding)[1], (*stride)[0], outH) ||
        !CalculateOutputDim(self->GetViewShape().GetDim(3), (*kernelSize)[1], (*dilation)[1], (*padding)[2],
                            (*padding)[3], (*stride)[1], outW)) {
        return false;
    }
    int64_t outChannels = 0;
    if (!SafeMul(self->GetViewShape().GetDim(1), (*kernelSize)[0], outChannels) ||
        !SafeMul(outChannels, (*kernelSize)[1], outChannels)) {
        return false;
    }
    int64_t outSpatial = 0;
    if (!SafeMul(outH, outW, outSpatial)) {
        return false;
    }
    outShape = {self->GetViewShape().GetDim(0), outChannels, outSpatial};
    return true;
}

const aclTensor* Im2col(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                        const aclIntArray* padding, const aclIntArray* stride, aclOpExecutor* executor)
{
    L0_DFX(Im2col, self, kernelSize, dilation, padding, stride);
    op::Shape outShape;
    if (!Im2colInferShape(self, kernelSize, dilation, padding, stride, outShape)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "im2col infer shape failed.");
        return nullptr;
    }
    auto out = executor->AllocTensor(outShape, self->GetDataType(), self->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Im2col, OP_INPUT(self), OP_OUTPUT(out),
                                           OP_ATTR(kernelSize, stride, dilation, PADDING_MODE, padding));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "Im2col ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op
