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
 * \file square_tiling_arch35.cpp
 * \brief
 */
#include "square_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/square_dag.h"
#include "log/log.h"

#include <iostream>

namespace optiling {
const uint64_t TILING_KEY_FP16 = 1;
const uint64_t TILING_KEY_BF16 = 2;
const uint64_t TILING_KEY_FP32 = 3;
const uint64_t TILING_KEY_INT32 = 4;
const uint64_t TILING_KEY_INT64 = 5;

ge::graphStatus SquareTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT &&
            this->inputDtype != ge::DT_INT32 && this->inputDtype != ge::DT_INT64,
        OP_LOGE(tilingContext->GetNodeName(), "self dtype not support"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquareTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT &&
            this->outputDtype != ge::DT_INT32 && this->outputDtype != ge::DT_INT64,
        OP_LOGE(tilingContext->GetNodeName(), "self dtype not support"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype, OP_LOGE(tilingContext->GetNodeName(), "out dtype not same as self"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquareTiling::CheckShape()
{
    auto selfStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, selfStorageShape);
    const gert::Shape& inputShape = Ops::Math::OpTiling::EnsureNotScalar(selfStorageShape->GetStorageShape());

    auto outStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outStorageShape);
    const gert::Shape& outputShape = Ops::Math::OpTiling::EnsureNotScalar(outStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputShape != outputShape, OP_LOGE(tilingContext->GetNodeName(), "input y and output z shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquareTiling::SetTilingData()
{
    if (this->outputDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        tilingKey = TILING_KEY_BF16;
    } else if (this->outputDtype == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FP32;
    } else if (this->outputDtype == ge::DT_INT32) {
        tilingKey = TILING_KEY_INT32;
    } else if (this->outputDtype == ge::DT_INT64) {
        tilingKey = TILING_KEY_INT64;
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "self dtype is only support fp16、bf16、fp32、int32、int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquareTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    // 获取tiling计算所需的参数
    ge::graphStatus status = ge::GRAPH_FAILED;
    status = CalcInputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    status = CalcOutputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    status = CheckShape();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);
    status = SetTilingData();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "SetTilingData failed"), return ge::GRAPH_FAILED);
    tiling = (tilingContext->GetTilingData<SquareTilingData>());
    if (tilingKey == TILING_KEY_FP16) {
        status = elewiseBaseTiling.DoTiling<SquareOp<half>::OpDag>(tiling->baseTiling);
    } else if (tilingKey == TILING_KEY_BF16) {
        status = elewiseBaseTiling.DoTiling<SquareOp<half>::OpDag>(tiling->baseTiling);
    } else if (tilingKey == TILING_KEY_FP32) {
        status = elewiseBaseTiling.DoTiling<SquareOp<float>::OpDag>(tiling->baseTiling);
    } else if (tilingKey == TILING_KEY_INT32) {
        status = elewiseBaseTiling.DoTiling<SquareOp<int32_t>::OpDag>(tiling->baseTiling);
    } else if (tilingKey == TILING_KEY_INT64) {
        status = elewiseBaseTiling.DoTiling<SquareOp<int64_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling DoTiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%ld.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetRawTilingData());
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Square(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling4Square rt2.0 is running.");
    SquareTiling squareTiling(tilingContext);
    return squareTiling.RunTiling();
}

static ge::graphStatus Tiling4SelectMode(gert::TilingContext* tilingContext)
{
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    return Tiling4Square(tilingContext); // ascendc
}

static ge::graphStatus TilingPrepareForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Square).Tiling(Tiling4SelectMode).TilingParse<ElewiseCompileInfo>(TilingPrepareForSquare);
} // namespace optiling
