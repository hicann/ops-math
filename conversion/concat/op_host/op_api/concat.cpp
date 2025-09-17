/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
