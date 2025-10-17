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
 * \file is_inf_tiling_arch35.cpp
 * \brief
 */
#include "is_inf_tiling_arch35.h"
#include "tiling_base/tiling_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/arch35/is_inf_dag.h"

namespace optiling {
using namespace Ops::Math::OpTiling;
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const uint64_t TILING_KEY_FP16 = 101UL;
const uint64_t TILING_KEY_BF16 = 102UL;
const uint64_t TILING_KEY_FP32 = 103UL;

ge::graphStatus IsInfRegbaseTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext, "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsInfRegbaseTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_BOOL,
        OP_LOGE(
            tilingContext, "output y dtype [%s] not supported, only support [DT_BOOL]",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsInfRegbaseTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape.GetDimNum() > 8, OP_LOGE(tilingContext, "input x dim num should be no more than 8"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE(tilingContext, "input x and output y shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsInfRegbaseTiling::SetTilingData()
{
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<uint64_t>(ASCEND_WORKSPACE);

    if (this->inputDtype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(TILING_KEY_FP16);
    } else if (this->inputDtype == ge::DT_BF16) {
        tilingContext->SetTilingKey(TILING_KEY_BF16);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(TILING_KEY_FP32);
    } else {
        OP_LOGE(
            tilingContext, "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    tilingContext->SetBlockDim(tiling_->blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsInfRegbaseTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    tiling_ = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling_ == nullptr), OP_LOGE(tilingContext, "Get EleBaseTilingDataV2 from context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->inputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsInfOp::IsInfDAG<half>::OpDag>(*tiling_);
    } else if (this->inputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsInfOp::IsInfDAG<bfloat16_t>::OpDag>(*tiling_);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsInfOp::IsInfDAG<float>::OpDag>(*tiling_);
    } else {
        OP_LOGE(
            tilingContext, "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}
} // namespace optiling