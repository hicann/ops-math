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
 * \file bias_add_grad_tiling_arch35.cpp
 * \brief tiling for bias add grad ops
 */

#include <vector>
#include "bias_add_grad_tiling.h"
#include "math/bias_add_grad/op_kernel/bias_add_grad_dag.h"
#include "math/bias_add_grad/op_kernel/bias_add_grad_tiling_key.h"
#include "bias_add_grad_tiling_arch35.h"
#include "log/log.h"

namespace optiling {
static constexpr int32_t SIZE8 = 8;
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static ge::graphStatus DoTiling(
    gert::TilingContext* context, const BiasAddGradCompileInfo* compileInfo, ReduceOpInputParam& opInput,
    ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE8) {
        status = Tiling4ReduceOp<BiasAddGrad::BiasAddGradDag<int64_t, int64_t>::OpDag>(
            context, opInput, key, &compileInfo->opInfo);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<BiasAddGrad::BiasAddGradDag<float, float>::OpDag>(
            context, opInput, key, &compileInfo->opInfo);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<BiasAddGrad::BiasAddGradDag<half, float>::OpDag>(
            context, opInput, key, &compileInfo->opInfo);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(
            context->GetNodeName(), "ReduceOp Tiling failed, dtype shoude be in (bfloat16/float16/float/int32/int64)"),
        return ge::GRAPH_FAILED);
    return status;
}

ge::graphStatus Tiling4BiasAddGradForAscendC(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const BiasAddGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    ReduceOpInputParam opInput;
    ge::graphStatus status = ReduceOpTmpl::GetInputParam(context, opInput, 0);
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED), OP_LOGE(context->GetNodeName(), "BiasAddGrad get input failed"),
        return ge::GRAPH_FAILED);

    auto inPutDimNum = opInput.shape.size();
    if (inPutDimNum < 2) { // 输入维度必须大于等于2
        OP_LOGE(context->GetNodeName(), "BiasAddGrad dimension must be greater than or equal to 2");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_format);

    if (strcmp(data_format, "NHWC") == 0) { // NHWC类型排除最后一维做Reduce
        opInput.axes.resize(inPutDimNum - 1);
        for (size_t i = 0; i < inPutDimNum - 1; i++) { // 排除C轴，轴数-1
            opInput.axes[i] = i;
        }
    } else if (strcmp(data_format, "NCHW") == 0) { // NCHW类型排除第一维做Reduce
        opInput.axes.resize(inPutDimNum - 1);
        int32_t j = 0;
        for (size_t i = 0; i <= inPutDimNum - 1; i++) { // 排除C轴，轴数-1
            if (i == 1) {
                continue;
            }
            opInput.axes[j++] = i;
        }
    } else {
        OP_LOGE(context->GetNodeName(), "BiasAddGrad data_format must in (NCHW, NHWC)");
        return ge::GRAPH_FAILED;
    }

    ReduceTilingKey key;
    OP_CHECK_IF(
        (DoTiling(context, compileInfo, opInput, key) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for BiasAddGrad"), return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(key.patternID, key.loopARCount, key.loopInnerARCount);
    OP_LOGI(
        context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu", key.patternID,
        key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling