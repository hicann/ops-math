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
 * \file ragged_bin_count_graph_infer.cpp
 * \brief Data type inference for the RaggedBinCount operator.
 */

#include "ragged_bin_count_graph_infer_internal.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "graph/operator_reg.h"

namespace ge {
namespace {
graphStatus ReadLegacyBinCount(const Operator& op, int64_t& bins)
{
    bins = UNKNOWN_DIM;
    Tensor sizeTensor;
    if (op.GetInputConstData("size", sizeTensor) != GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    }
    const auto dtype = op.GetInputDescByName("size").GetDataType();
    if (sizeTensor.GetData() == nullptr || sizeTensor.GetTensorDesc().GetDataType() != dtype) {
        return GRAPH_FAILED;
    }
    if (dtype == DT_INT32 && sizeTensor.GetSize() == sizeof(int32_t)) {
        int32_t value = 0;
        std::memcpy(&value, sizeTensor.GetData(), sizeof(value));
        bins = value;
    } else if (dtype == DT_INT64 && sizeTensor.GetSize() == sizeof(int64_t)) {
        std::memcpy(&bins, sizeTensor.GetData(), sizeof(bins));
    } else {
        return GRAPH_FAILED;
    }
    return bins >= 0 ? GRAPH_SUCCESS : GRAPH_FAILED;
}
} // namespace

IMPLEMT_COMMON_INFERFUNC(RaggedBinCountLegacyInferShape)
{
    const auto splits = op.GetInputDescByName("splits").GetShape().GetDims();
    const auto size = op.GetInputDescByName("size").GetShape().GetDims();
    if (splits != UNKNOWN_RANK && (splits.size() != 1U || (splits[0] != UNKNOWN_DIM && splits[0] < 2))) {
        return GRAPH_FAILED;
    }
    if (size != UNKNOWN_RANK && (size.size() != 1U || (size[0] != UNKNOWN_DIM && size[0] != 1))) {
        return GRAPH_FAILED;
    }

    const int64_t rows = splits.size() == 1U && splits[0] >= 2 ? splits[0] - 1 : UNKNOWN_DIM;
    int64_t bins = UNKNOWN_DIM;
    if (ReadLegacyBinCount(op, bins) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    auto output = op.GetOutputDescByName("output");
    output.SetShape(Shape({rows, bins}));
    // Keep the legacy IR dtype surface; native 950 dtype checks remain in the
    // existing InferDataType and tiling callbacks.
    output.SetDataType(op.GetInputDescByName("weights").GetDataType());
    if (rows == UNKNOWN_DIM || bins == UNKNOWN_DIM) {
        const auto rowRange = rows == UNKNOWN_DIM ? std::make_pair(int64_t{0}, UNKNOWN_DIM) :
                                                    std::make_pair(rows, rows);
        const auto binRange = bins == UNKNOWN_DIM ? std::make_pair(int64_t{0}, UNKNOWN_DIM) :
                                                    std::make_pair(bins, bins);
        if (output.SetShapeRange({rowRange, binRange}) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
    }
    return op.UpdateOutputDesc("output", output);
}

// GEIR can use the legacy registry before the native InferShape callback.
// Its built-in implementation resizes weights to the rank of values, losing
// dimensions (and elements). Infer only the output and preserve every input.
COMMON_INFER_FUNC_REG(RaggedBinCount, RaggedBinCountLegacyInferShape);
} // namespace ge

namespace ops {
static ge::graphStatus InferDataTypeForRaggedBinCount(gert::InferDataTypeContext* context)
{
    return ragged_bin_count_graph_infer_internal::InferDataType(context);
}

IMPL_OP(RaggedBinCount).InferDataType(InferDataTypeForRaggedBinCount);
} // namespace ops
