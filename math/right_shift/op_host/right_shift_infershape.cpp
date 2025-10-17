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
 * \file right_shift_infershape.cpp
 * \brief
 */
#include "runtime_util.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForRightShift(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForRightShift");
    const gert::Shape* x_shape = context->GetInputShape(kInputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* y_shape = context->GetInputShape(kInputIndex1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* z_shape = context->GetOutputShape(kOutputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, z_shape);
    OP_LOGE_IF(
        !BroadcastShape(x_shape, y_shape, z_shape), ge::GRAPH_FAILED, context->GetNodeName(),
        "call BroadcastShape failed.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RightShift).InferShape(InferShapeForRightShift);
} // namespace ops
