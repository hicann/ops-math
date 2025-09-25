/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ger.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Ger);

static constexpr size_t EXPECT_DIM = 2;

const aclTensor* Ger(const aclTensor* self, const aclTensor* vec2, aclOpExecutor* executor)
{
    L0_DFX(Ger, self, vec2);

    op::Shape outShape;
    outShape.SetDimNum(EXPECT_DIM);
    outShape.SetDim(0, self->GetViewShape().GetDim(0));
    outShape.SetDim(1, vec2->GetViewShape().GetDim(0));

    auto out = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Ger, OP_INPUT(self, vec2), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GerAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
} // namespace l0op
