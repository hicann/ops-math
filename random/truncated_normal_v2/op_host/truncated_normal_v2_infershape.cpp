/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncated_normal_v2_infershape.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
// -------------------TruncatedNormalV2 Ops START---------------------
const int32_t INDEX_INPUT_SHAPE = 0;
const int32_t INDEX_OUTPUT_Y = 0;
const int32_t INDEX_OUTPUT_OFFSET = 1;

template <typename T>
ge::graphStatus GetValueToShape(const gert::Tensor* const_tensor, gert::Shape* const_shape)
{
    const T* constValue = const_tensor->GetData<T>();
    const size_t constNum = const_tensor->GetShapeSize();
    if (static_cast<int32_t>(constNum) < 0) {
        return ge::GRAPH_FAILED;
    }
    const_shape->SetDimNum(constNum);
    for (size_t i = 0; i < constNum; ++i) {
        const_shape->SetDim(i, constValue[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTruncatedNormalV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), " TruncatedNormalV2 runtime2.0 is begin");
    const gert::Tensor* shape_tensor = context->GetInputTensor(INDEX_INPUT_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
    ge::DataType shape_dtype = shape_tensor->GetDataType();
    gert::Shape* y_shape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    gert::Shape* offset_shape = context->GetOutputShape(INDEX_OUTPUT_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context, offset_shape);
    offset_shape->SetDimNum(1);
    offset_shape->SetDim(0, 1);

    switch (shape_dtype) {
        case ge::DT_INT32: {
            return GetValueToShape<int32_t>(shape_tensor, y_shape);
        }
        case ge::DT_INT64: {
            return GetValueToShape<int64_t>(shape_tensor, y_shape);
        }
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TruncatedNormalV2).InputsDataDependency({0}).InferShape(InferShapeForTruncatedNormalV2);
// -------------------TruncatedNormalV2 Ops END---------------------
} // namespace ops