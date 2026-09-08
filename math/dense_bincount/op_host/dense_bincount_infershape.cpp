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
#include "util/const_util.h"
#include "log/log.h"
#include "util/shape_util.h"
#include <limits>
namespace ops {
constexpr int64_t MAX_OUTPUT_ELEMENTS = std::numeric_limits<int64_t>::max() / sizeof(float);

static ge::graphStatus InferShapeDenseBincount(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    const auto* sizeTensor = context->GetInputTensor(1);
    auto* output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    if (Ops::Base::IsUnknownRank(*input)) {
        Ops::Base::SetUnknownRank(*output);
        return ge::GRAPH_SUCCESS;
    }
    if (input->GetDimNum() < 1 || input->GetDimNum() > 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input",
                                                 std::to_string(input->GetDimNum()).c_str(), "1D or 2D tensor");
        return ge::GRAPH_FAILED;
    }
    const auto& shape = *input;
    int64_t size = 0;
    if (!Ops::Base::GetConstInt(context, 1, size)) {
        output->SetDimNum(shape.GetDimNum());
        output->SetDim(0, shape.GetDim(0));
        if (shape.GetDimNum() == 1) {
            output->SetDim(0, -1);
        } else {
            output->SetDim(1, -1);
        }
        return ge::GRAPH_SUCCESS;
    }
    if (size < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", std::to_string(size),
                                              "size must be a non-negative constant");
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = shape.GetDimNum() == 1 ? 1 : shape.GetDim(0);
    if (rows >= 0 && rows > 0 && size > MAX_OUTPUT_ELEMENTS / rows) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output.numel", "overflow",
                                              "output element count exceeds the int64 FLOAT32 storage limit");
        return ge::GRAPH_FAILED;
    }
    output->SetDimNum(shape.GetDimNum());
    output->SetDim(0, shape.GetDim(0));
    if (shape.GetDimNum() == 1)
        output->SetDim(0, size);
    else
        output->SetDim(1, size);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DenseBincount).InferShape(InferShapeDenseBincount).InputsDataDependency({1});
} // namespace ops
