/**
 * Copyright (c) Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_bernoulli_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"

using namespace ge;

namespace ops {
const int32_t INDEX_OUTPUT_Y = 0;
const int32_t INDEX_ATTR = 0;

static ge::graphStatus InferDataType4StatelessBernoulli(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), " InferDataType4StatelessBernoulli runtime2.0 is begin.");
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    auto dTypePtr = attrPtr->GetAttrPointer<int32_t>(INDEX_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, dTypePtr);
    ge::DataType outDtype = static_cast<ge::DataType>(*dTypePtr);
    context->SetOutputDataType(INDEX_OUTPUT_Y, outDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(StatelessBernoulli).InferDataType(InferDataType4StatelessBernoulli);
} // namespace ops