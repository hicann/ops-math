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
 * \file trunc_tiling_arch35.cpp
 * \brief
 */
#include "trunc_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/trunc/op_kernel/arch35/trunc_dag.h"
#include "math/trunc/op_kernel/arch35/trunc_struct.h"

#include <iostream>

namespace optiling
{
using namespace Ops::Base;
const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;

ge::graphStatus TruncTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "TruncTiling SetTilingData enter.");

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "TruncTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 &&
                   this->inputDtype != ge::DT_FLOAT && this->inputDtype != ge::DT_INT8 &&
                   this->inputDtype != ge::DT_UINT8 && this->inputDtype != ge::DT_INT32,
               OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "TruncTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != outputZShape,
               OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "TruncTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as input x"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "TruncTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<TruncTilingData>();
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<TruncDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<TruncDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<TruncDAG<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        dType = TPL_INT8;
        baseTilingResult = elewiseBaseTiling.DoTiling<TruncDAGInt<int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        dType = TPL_UINT8;
        baseTilingResult = elewiseBaseTiling.DoTiling<TruncDAGInt<uint8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        dType = TPL_INT32;
        baseTilingResult = elewiseBaseTiling.DoTiling<TruncDAGInt<int32_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus TilingForTrunc(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Trunc rt2.0 is running.");
    TruncTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForTrunc([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Trunc).Tiling(TilingForTrunc).TilingParse<TruncCompileInfo>(TilingPrepareForTrunc);
}  // namespace optiling