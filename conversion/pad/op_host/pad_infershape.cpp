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
 * \file pad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "../../pad_v3/op_host/pad_infershape_common.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static constexpr size_t PAD_IN_IDX_X = 0;
static constexpr size_t PAD_IN_IDX_PADDINGS = 1;
static constexpr size_t PAD_OUT_IDX_Y = 0;

static ge::graphStatus InferShapeForPad(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(PAD_IN_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(PAD_OUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    const gert::Tensor* paddings_tensor = context->GetInputTensor(PAD_IN_IDX_PADDINGS);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddings_tensor);

    // if x_shape is unknown rank [-2] that means cannot know how many ranks,
    // which make output unknown rank.
    if (IsUnknownRank(*x_shape)) {
        SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS;
    }

    return InferShapeForPadWithPaddingTensor(context, x_shape, paddings_tensor, y_shape);
}

IMPL_OP_INFERSHAPE(Pad).InferShape(InferShapeForPad).InputsDataDependency({PAD_IN_IDX_PADDINGS});
} // namespace ops
