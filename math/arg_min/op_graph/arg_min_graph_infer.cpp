/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferDataType4ArgMin(gert::InferDataTypeContext* context)
{
    OP_LOGD("Begin InferDataType4ArgMin");
    ge::DataType outputDtype = ge::DT_INT32;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto attr_dtype_ptr = attrs->GetAttrPointer<int64_t>(0);

    if (attr_dtype_ptr != nullptr) {
        outputDtype = static_cast<ge::DataType>(*attr_dtype_ptr);
    } else {
        OP_LOGW(context->GetNodeName(), "get attr dtype failed.");
    }
    context->SetOutputDataType(0, outputDtype);
    OP_LOGD("End InferDataType4ArgMin");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ArgMin).InferDataType(InferDataType4ArgMin);
} // namespace ops
