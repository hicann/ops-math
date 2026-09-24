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
 * \file fused_mul_addn_l2loss_graph_infer.cpp
 * \brief FusedMulAddNL2loss inferDataType：y1/y2 dtype 均与 x1 一致（对齐 910b built-in infer）
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus InferDataTypeForFusedMulAddNL2loss(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataTypeForFusedMulAddNL2loss");
    const ge::DataType x1DataType = context->GetInputDataType(0);
    const ge::DataType x2DataType = context->GetInputDataType(1);
    const ge::DataType x3DataType = context->GetInputDataType(2);

    // README 约束：x1/x2/x3/y1/y2 dtype 必须一致，且仅支持 FLOAT/FLOAT16
    OP_CHECK_IF(x1DataType != ge::DT_FLOAT && x1DataType != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(), "unsupported x1 dtype %d: only FLOAT and FLOAT16 are supported",
                        static_cast<int>(x1DataType)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2DataType != x1DataType,
                OP_LOGE(context->GetNodeName(), "x2 dtype %d must equal x1 dtype %d", static_cast<int>(x2DataType),
                        static_cast<int>(x1DataType)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(x3DataType != x1DataType,
                OP_LOGE(context->GetNodeName(), "x3 dtype %d must equal x1 dtype %d", static_cast<int>(x3DataType),
                        static_cast<int>(x1DataType)),
                return ge::GRAPH_FAILED);

    context->SetOutputDataType(0, x1DataType); // y1 = x1 * x3 + x2，dtype 同 x1
    context->SetOutputDataType(1, x1DataType); // y2 = sum(x1^2 / 2)，dtype 同 x1
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedMulAddNL2loss).InferDataType(InferDataTypeForFusedMulAddNL2loss);
} // namespace ops
