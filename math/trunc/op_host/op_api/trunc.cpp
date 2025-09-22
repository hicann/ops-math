/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trunc.cpp
 * \brief
 */
#include "trunc.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Trunc);

static bool TruncInferShape(const op::Shape& selfShape, op::Shape& outShape)
{
    outShape = selfShape;
    return true;
}

const aclTensor* Trunc(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Trunc, self);

    op::Shape outShape;
    if (!TruncInferShape(self->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infer shape %s and %s failed.",
            op::ToString(self->GetStorageShape()).GetString(),
            op::ToString(outShape).GetString());
    return nullptr;
    }

    auto out = executor->AllocTensor(outShape, self->GetDataType());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Trunc, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

}  // namespace l0op
