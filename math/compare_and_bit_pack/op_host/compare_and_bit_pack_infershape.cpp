/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops {
namespace {
constexpr size_t kInputXIdx = 0;
constexpr size_t kInputThresholdIdx = 1;
constexpr size_t kOutputYIdx = 0;
constexpr int64_t kBitsPerByte = 8;
} // namespace

static ge::graphStatus InferShape4CompareAndBitpack(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(kInputXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* thresholdShape = context->GetInputShape(kInputThresholdIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, thresholdShape);
    gert::Shape* yShape = context->GetOutputShape(kOutputYIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    OP_CHECK_IF(xShape->GetDimNum() == 0, OP_LOGE(context->GetNodeName(), "The rank of x must be at least 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*thresholdShape) && thresholdShape->GetDimNum() != 0,
                OP_LOGE(context->GetNodeName(), "The threshold must be a scalar."), return ge::GRAPH_FAILED);

    *yShape = *xShape;
    const size_t lastDimIdx = xShape->GetDimNum() - 1;
    const int64_t lastDim = xShape->GetDim(lastDimIdx);
    if (lastDim == ge::UNKNOWN_DIM) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        lastDim % kBitsPerByte != 0,
        OP_LOGE(context->GetNodeName(), "The last dimension of x must be divisible by 8, but got %ld.", lastDim),
        return ge::GRAPH_FAILED);

    yShape->SetDim(lastDimIdx, lastDim / kBitsPerByte);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CompareAndBitpack).InferShape(InferShape4CompareAndBitpack);

} // namespace ops
