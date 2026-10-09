/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file floor_mod_infer.cpp
 * \brief
 */
#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kInputX1 = 0U;
constexpr size_t kInputX2 = 1U;
constexpr size_t kOutputY = 0U;

ge::graphStatus InferShapeFloorMod(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(kInputX1);
    const gert::Shape* x2Shape = context->GetInputShape(kInputX2);
    gert::Shape* yShape = context->GetOutputShape(kOutputY);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (x1Shape->GetDimNum() == 0U) {
        *yShape = *x2Shape;
        return ge::GRAPH_SUCCESS;
    }
    if (x2Shape->GetDimNum() == 0U) {
        *yShape = *x1Shape;
        return ge::GRAPH_SUCCESS;
    }
    return Ops::Base::InferShape4Broadcast(context);
}
} // namespace

IMPL_OP_INFERSHAPE(FloorMod).InferShape(InferShapeFloorMod);
} // namespace ops
