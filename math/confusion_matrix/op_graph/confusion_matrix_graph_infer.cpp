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
 * \file confusion_matrix_graph_infer.cpp
 * \brief Infer data type implementation for ConfusionMatrix operator
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

#include <string>

using namespace ge;
namespace ops {
static constexpr size_t kAttrIndexDtype = 1U;
static constexpr size_t kOutputIndexY = 0U;

static ge::graphStatus InferDataTypeForConfusionMatrix(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferDataTypeForConfusionMatrix");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* dtypeStr = attrs->GetAttrPointer<char>(kAttrIndexDtype);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeStr);

    ge::DataType outputDtype = ge::DT_FLOAT;
    std::string dtypeString(dtypeStr);
    if (dtypeString == "float32") {
        outputDtype = ge::DT_FLOAT;
    } else if (dtypeString == "int32") {
        outputDtype = ge::DT_INT32;
    } else if (dtypeString == "int8") {
        outputDtype = ge::DT_INT8;
    } else if (dtypeString == "float16") {
        outputDtype = ge::DT_FLOAT16;
    } else if (dtypeString == "uint8") {
        outputDtype = ge::DT_UINT8;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", dtypeStr,
                                              "dtype must be one of: float32, int32, int8, float16, uint8.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(kOutputIndexY, outputDtype);
    OP_LOGI(context->GetNodeName(), "End to do InferDataTypeForConfusionMatrix");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ConfusionMatrix).InferDataType(InferDataTypeForConfusionMatrix);
} // namespace ops
