/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill_diagonal_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

ge::graphStatus FillDiagonalV2InferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do FillDiagonalV2InferShapeFunc");
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    *y_shape = *x_shape;
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(*y_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

graphStatus FillDiagonalV2InferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do FillDiagonalV2InferDataTypeFunc");
    auto inputDtype = context->GetInputDataType(0);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());
    OP_LOGD(context, "before set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    context->SetOutputDataType(0, inputDtype);
    OP_LOGD(context, "after set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    OP_LOGD(context, "End to do FillDiagonalV2InferDataTypeFunc end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FillDiagonalV2)
    .InferShape(FillDiagonalV2InferShapeFunc)
    .InferDataType(FillDiagonalV2InferDataTypeFunc);
} // namespace ops