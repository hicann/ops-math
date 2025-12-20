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
 * \file truncated_normal_v2_graph_infer.cpp
 * \brief truncated_normal_v2 operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t TruncatedNormalV2_IN_X_IDX = 0;
static constexpr size_t TruncatedNormalV2_OUT_Y_IDX = 0;
static constexpr size_t TruncatedNormalV2_DTYPE_IDX = 2;
static constexpr int64_t DEFAULT_VALUE = 0;

static graphStatus TruncatedNormalV2InferDataTypeByAttr(gert::InferDataTypeContext* context, const size_t dtype_index,
                                                     const std::set<DataType> &support_dtype) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* attrDtype = attrs->GetAttrPointer<int64_t>(dtype_index);
    int64_t dtype = attrDtype == nullptr ? DEFAULT_VALUE : *attrDtype;
    ge::DataType yDtype = static_cast<ge::DataType>(dtype);
    OP_CHECK_IF(support_dtype.count(yDtype) == 0,
        OP_LOGE(context, "out shape dtype should be float, float16, bfloat16 but got %s.",
        Ops::Base::ToString(yDtype).c_str()), return ge::GRAPH_FAILED);
    
    context->SetOutputDataType(0, yDtype);
    OP_LOGD(context, "TruncatedNormalV2InferDataTypeByAttr end. Data type is %s.", Ops::Base::ToString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4TruncatedNormalV2(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(1, ge::DT_INT64);

    std::set<ge::DataType> support_dtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    return TruncatedNormalV2InferDataTypeByAttr(context, TruncatedNormalV2_DTYPE_IDX, support_dtype);
}

IMPL_OP(TruncatedNormalV2).InferDataType(InferDataType4TruncatedNormalV2);
}; // namespace ops