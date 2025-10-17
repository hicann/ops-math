/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat.cpp
 * \brief
 */
#include "concat.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(ConcatD);

static op::Shape ConcatDInferShape(const aclTensorList* inputs, int64_t dim)
{
    op::Shape concatShape = (*inputs)[0]->GetViewShape();
    int64_t concatDimSize = 0;
    for (uint64_t i = 0; i < inputs->Size(); i++) {
        concatDimSize += (*inputs)[i]->GetViewShape().GetDim(dim);
    }
    concatShape.SetDim(dim, concatDimSize);
    return concatShape;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, op::DataType outDtype, aclOpExecutor* executor)
{
    L0_DFX(ConcatD, inputs, dim, outDtype);
    op::Shape concatShape = ConcatDInferShape(inputs, dim);
    auto out = executor->AllocTensor(concatShape, outDtype, (*inputs)[0]->GetViewFormat());

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ConcatD, OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ConcatD ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, aclOpExecutor* executor)
{
    L0_DFX(ConcatD, inputs, dim);
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    size_t catMaxInputSize = (socVersion == op::SocVersion::ASCEND910_95) ? 512 : 32;
    if (inputs->Size() == 0 || inputs->Size() > catMaxInputSize) {
        OP_LOGE(
            ACLNN_ERR_INNER, "Inputs tensor list's size should be (0, %zu]) but current size is %zu.", catMaxInputSize,
            inputs->Size());
        return nullptr;
    }
    op::Shape concatShape = ConcatDInferShape(inputs, dim);
    auto out = executor->AllocTensor(concatShape, (*inputs)[0]->GetDataType(), (*inputs)[0]->GetViewFormat());

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ConcatD, OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ConcatD ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

} // namespace l0op
