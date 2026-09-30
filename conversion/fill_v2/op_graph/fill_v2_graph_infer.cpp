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
 * \file fill_v2_graph_infer.cpp
 * \brief fill_v2 operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus InferDataType4FillV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4FillV2 enter");
    // FillV2的output dtype与input dims的dtype不同(dims为INT16/INT32/INT64),无法从input推导;
    // 对齐canndev FillV2D实现,输出数据类型固定为float32。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "InferDataType4FillV2 end");
    return GRAPH_SUCCESS;
}

IMPL_OP(FillV2).InferDataType(InferDataType4FillV2);
} // namespace ops
