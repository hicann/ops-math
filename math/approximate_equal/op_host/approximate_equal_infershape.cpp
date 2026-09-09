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
 * \file approximate_equal_infershape.cpp
 * \brief ApproximateEqual shape inference (y shape = broadcast(x1, x2))
 */

#include "util/shape_util.h"
#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ApproximateEqual(gert::InferShapeContext* context)
{
    const auto ret = Ops::Base::InferShape4Broadcast(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "broadcast shape inference failed"),
                return ret);
    const auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s", Ops::Base::ToString(*outputShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApproximateEqual).InferShape(InferShape4ApproximateEqual);

} // namespace ops
