/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "diag_flat.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

// *******************************************************************************************************
// ******************************** [AI CORE] DiagFlat 算子入口 ***************************************
// *******************************************************************************************************

namespace l0op {

OP_TYPE_REGISTER(DiagFlat);

// AICORE算子kernel
static const aclTensor* DiagFlatAiCore(
    const aclTensor* self, int64_t diagonal, aclTensor* diagFlatOut, aclOpExecutor* executor)
{
    L0_DFX(DiagFlatAiCore, self, diagonal, diagFlatOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore DiagFlat算子加入任务队列
    // self是算子的输入，diagonal是算子的属性，out是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(DiagFlat, OP_INPUT(self), OP_OUTPUT(diagFlatOut), OP_ATTR(diagonal));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "DiagFlat ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return diagFlatOut;
}

const aclTensor* DiagFlat(const aclTensor* self, int64_t diagonal, aclOpExecutor* executor)
{
    auto outputLength = self->Numel() + std::abs(diagonal);
    op::Shape diagflatOutShape = self->GetViewShape();

    diagflatOutShape.SetDimNum(2);
    diagflatOutShape.SetDim(0, outputLength);
    diagflatOutShape.SetDim(1, outputLength);

    auto diagFlatOut = executor->AllocTensor(diagflatOutShape, self->GetDataType(), op::Format::FORMAT_ND);
    return DiagFlatAiCore(self, diagonal, diagFlatOut, executor);
}

} // namespace l0op
