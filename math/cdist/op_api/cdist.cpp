/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cdist.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
constexpr size_t MIN_DIM_LEN = 2;

OP_TYPE_REGISTER(Cdist);
const aclTensor* Cdist(const aclTensor *x1, const aclTensor *x2, float p, int64_t compute_mode, 
                       aclOpExecutor *executor) {
    L0_DFX(Cdist, x1, x2, p, compute_mode);
    // 计算out的shape
    op::Shape outShape;
    int64_t x1DimNum = x1->GetViewShape().GetDimNum();
    int64_t x2DimNum = x2->GetViewShape().GetDimNum();
    for(int64_t i = 0; i < x1DimNum - 1; i++) {
        int64_t dim = x1->GetViewShape().GetDim(i);
        outShape.AppendDim(dim);
    }
    outShape.AppendDim(x2->GetViewShape().GetDim(x2DimNum - MIN_DIM_LEN));

    auto out = executor->AllocTensor(outShape, x1->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(out != nullptr, nullptr);
    INFER_SHAPE(Cdist, OP_INPUT(x1, x2), OP_OUTPUT(out), OP_ATTR(p, compute_mode));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cdist, 
                                           OP_INPUT(x1, x2), 
                                           OP_ATTR(p, compute_mode), 
                                           OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CdistAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}
}   // namespace l0op