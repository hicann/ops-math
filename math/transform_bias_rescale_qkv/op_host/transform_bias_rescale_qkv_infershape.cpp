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
 * \file transform_bias_rescale_qkv_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;

static ge::graphStatus InferShape4TransformBiasRescaleQkv(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4InferShape4TransformBiasRescaleQkv");

    // get input shapes
    auto qkvShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);

    auto qkvBiasShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvBiasShape);

    // get output shapes
    auto qShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);

    auto kShape = context->GetOutputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);

    auto vShape = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);

    OP_LOGD(context, "End to do InferShape4InferShape4TransformBiasRescaleQkv");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4TransformBiasRescaleQkv(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4TransformBiasRescaleQkv");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);
    context->SetOutputDataType(IDX_1, input_dtype);
    context->SetOutputDataType(IDX_2, input_dtype);

    OP_LOGD(context, "End to do InferDataType4TransformBiasRescaleQkv");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TransformBiasRescaleQkv)
    .InferShape(InferShape4TransformBiasRescaleQkv)
    .InferDataType(InferDataType4TransformBiasRescaleQkv);
} // namespace ops