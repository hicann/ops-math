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
 * \file dot_tiling_arch35.cc
 * \brief tiling for dot
 */

#include "dot_tiling_arch35.h"
#include <vector>
#include "../../op_kernel/arch35/dot_dag.h"
#include "../../op_kernel/arch35/dot_tiling_key.h"
#include "log/log.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
using namespace Ops::Base;
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static ge::graphStatus DoTiling(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<Dot::DotDag<float, float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<Dot::DotDag<half, float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == 1) {
        status = Tiling4ReduceOp<Dot::DotDagI8<int8_t, float>::OpDag>(context, opInput, key);
    }

    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(
            context->GetNodeName(), "ReduceOp Tiling failed, dtype shoude be in (int8/uint8/half/bf16/float/int32)"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus GenInput(gert::TilingContext* context, ReduceOpInputParam& opInput)
{
    std::vector<int64_t> shape0;
    std::vector<int64_t> shape1;
    OP_CHECK_IF((ReduceOpTmpl::GetInputShape(context, 0, shape0) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get input_x shape failed"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF((ReduceOpTmpl::GetInputShape(context, 1, shape1) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get input_y shape failed"),
                    return ge::GRAPH_FAILED);
    if (shape0.size() != shape1.size()) {
        OP_LOGE(context, "shape0 size:%zu not equeal to shape1 size:%zu", shape0.size(), shape1.size());
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < shape0.size(); i++) {
        if (shape0[i] != shape1[i]) {
            OP_LOGE(context, "shape0[%zu]:%ld not equeal to shape1[%zu]:%ld", i, shape0[i], i, shape1[i]);
            return ge::GRAPH_FAILED;
        }
    }
    opInput.shape = shape0;

    ge::DataType dtype0;
    ge::DataType dtype1;
    OP_CHECK_IF((ReduceOpTmpl::GetInputDtype(context, 0, dtype0) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get input_x dtype failed"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF((ReduceOpTmpl::GetInputDtype(context, 1, dtype1) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get input_y dtype failed"),
                    return ge::GRAPH_FAILED);
    if (dtype0 != dtype1) {
        OP_LOGE(context, "dtype0:%d not equeal to dtype1:%d", dtype0, dtype1);
        return ge::GRAPH_FAILED;
    }
    opInput.inputDtype = dtype0;
    opInput.axes.resize(opInput.shape.size());
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        opInput.axes[i] = i;
    }
    return ge::GRAPH_SUCCESS;
}

[[maybe_unused]] static ge::graphStatus TilingForDot(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const DotCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    ReduceOpInputParam opInput;
    OP_CHECK_IF((GenInput(context, opInput) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"),
                    return ge::GRAPH_FAILED);

    ReduceTilingKey key;
    OP_CHECK_IF((DoTiling(context, opInput, key) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceSum"),
                    return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(key.patternID, key.loopARCount, key.loopInnerARCount);
    OP_LOGI(context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
            key.patternID, key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

[[maybe_unused]] static ge::graphStatus TilingPrepareForDot([[maybe_unused]]gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Dot).Tiling(TilingForDot).TilingParse<DotCompileInfo>(TilingPrepareForDot);
}  // namespace optiling