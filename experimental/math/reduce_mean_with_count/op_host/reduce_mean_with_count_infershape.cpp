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
 * @file reduce_mean_with_count_infershape.cpp
 * @brief ReduceMeanWithCount shape inference implementation
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "reduce_mean_with_count_common.h"
#include <vector>

using namespace ge;

namespace ops {

using ops_reduce_mean_with_count::IsReduceDim;
using ops_reduce_mean_with_count::NormalizeAxes;

static void BuildOutputDims(
    const gert::Shape* inputShape, const std::vector<int64_t>& axes, bool keepdim,
    std::vector<int64_t>& outDims)
{
    int64_t rank = static_cast<int64_t>(inputShape->GetDimNum());
    for (int64_t d = 0; d < rank; d++) {
        if (IsReduceDim(axes, d)) {
            if (keepdim) {
                outDims.push_back(1);
            }
        } else {
            outDims.push_back(inputShape->GetDim(d));
        }
    }
}

static void ApplyOutputShape(
    gert::Shape* meanShape, gert::Shape* countShape, const std::vector<int64_t>& outDims)
{
    meanShape->SetDimNum(outDims.size());
    countShape->SetDimNum(outDims.size());
    for (size_t i = 0; i < outDims.size(); i++) {
        meanShape->SetDim(i, outDims[i]);
        countShape->SetDim(i, outDims[i]);
    }
}

static ge::graphStatus InferShape4ReduceMeanWithCount(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* meanShape = context->GetOutputShape(0);
    gert::Shape* countShape = context->GetOutputShape(1);
    if (meanShape == nullptr || countShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto axisListPtr = attrs->GetListInt(0);
    const bool* keepdimPtr = attrs->GetBool(1);
    bool keepdim = (keepdimPtr != nullptr) ? *keepdimPtr : false;

    int64_t rank = static_cast<int64_t>(inputShape->GetDimNum());
    if (rank == 0) {
        meanShape->SetDimNum(0);
        countShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    std::vector<int64_t> axes = NormalizeAxes(axisListPtr, rank);
    std::vector<int64_t> outDims;
    BuildOutputDims(inputShape, axes, keepdim, outDims);
    ApplyOutputShape(meanShape, countShape, outDims);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceMeanWithCount).InferShape(InferShape4ReduceMeanWithCount);

} // namespace ops
