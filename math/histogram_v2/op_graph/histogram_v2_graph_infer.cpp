/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_graph_infer.cpp
 * \brief histogram_v2 operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "histogram_v2_proto.h"

namespace ops {

static ge::graphStatus HistogramV2InferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramV2InferDataTypeFunc");

    auto inputDtype = context->GetInputDataType(0);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());
    OP_LOGD(context, "befer set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    context->SetOutputDataType(0, ge::DT_INT32);
    OP_LOGD(context, "after set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    OP_LOGD(context, "End to do HistogramV2InferDataTypeFunc end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(HistogramV2).InferDataType(HistogramV2InferDataTypeFunc);

}; // namespace ops