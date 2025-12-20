/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
/*!
 * \file random_standard_normal_v2_graph_infer.cpp
 * \brief random_standard_normal_v2 operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static graphStatus RandomStandardNormalV2InferDataTypeByAttr(
    gert::InferDataTypeContext* context, const size_t dtype_index, const std::set<DataType>& support_dtype)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "RandomStandardNormalV2InferDataTypeByAttr begin.");
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* attrDtype = attrs->GetAttrPointer<int64_t>(dtype_index);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrDtype);
    ge::DataType yDtype = static_cast<ge::DataType>(*attrDtype);
    OP_CHECK_IF(support_dtype.count(yDtype) == 0,
        OP_LOGE(context, "out shape dtype should be float, float16, bfloat16 but got %s.",
        Ops::Base::ToString(yDtype).c_str()), return ge::GRAPH_FAILED);
    
    context->SetOutputDataType(0, yDtype);
    OP_LOGD(
        context, "RandomStandardNormalV2InferDataTypeByAttr end. Data type is %s.",
        Ops::Base::ToString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

static graphStatus RandomStandardNormalV2InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(1, ge::DT_INT64);

    std::set<ge::DataType> support_dtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    return RandomStandardNormalV2InferDataTypeByAttr(context, 0, support_dtype);
}

IMPL_OP(RandomStandardNormalV2).InferDataType(RandomStandardNormalV2InferDataType);
}; // namespace ops