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
 * \file stateless_randperm_graph_infer.cpp
 * \brief stateless_randperm operater graph infer resource
 */

#include <set>

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t STATELESS_RANDPERM_ATTR_IDX_DTYPE = 1; // attrs: layout(0), dtype(1)
static constexpr size_t STATELESS_RANDPERM_Y_OUTPUT_IDX = 0;

static ge::graphStatus InferDataTypeStatelessRandperm(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferDataTypeStatelessRandperm");
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const ge::DataType* attrDtype = attrs->GetAttrPointer<ge::DataType>(STATELESS_RANDPERM_ATTR_IDX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDtype);

    const std::set<ge::DataType> supportDtype = {ge::DT_INT64,   ge::DT_INT32, ge::DT_INT16,  ge::DT_UINT8, ge::DT_INT8,
                                                 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_BF16};
    if (supportDtype.count(*attrDtype) == 0) {
        OP_LOGE(context->GetNodeName(),
                "The dtype only supports int64, int32, int16, uint8, int8, float16, float32, double, bfloat16.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(STATELESS_RANDPERM_Y_OUTPUT_IDX, *attrDtype);
    OP_LOGI(context->GetNodeName(), "End to do InferDataTypeStatelessRandperm");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(StatelessRandperm).InferDataType(InferDataTypeStatelessRandperm);
} // namespace ops
