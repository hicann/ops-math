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
 * \file is_nan_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "is_nan_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/util/fp16.h"
#include "log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/is_nan/op_kernel/arch35/is_nan_dag.h"

namespace optiling {
using namespace Ops::Math::OpTiling;
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const uint64_t TILING_KEY_FP16 = 101UL;
const uint64_t TILING_KEY_BF16 = 102UL;
const uint64_t TILING_KEY_FP32 = 103UL;

ge::graphStatus IsNanTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext->GetNodeName(),
            "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNanTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_BOOL,
        OP_LOGE(
            tilingContext->GetNodeName(), "output y dtype [%s] not supported, only support [DT_BOOL]",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNanTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNanTiling::SetTilingData()
{
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
            tilingContext->GetNodeName(),
            "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNanTiling::RunTiling()
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

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<IsNanTilingData>();
    if (this->inputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNanOp::IsNanDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNanOp::IsNanDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNanOp::IsNanDAG<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(
            tilingContext->GetNodeName(),
            "input x dtype [%s] not supported, only support [DT_FLOAT16, DT_BF16, DT_FLOAT]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus TilingPrepare4IsNan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4IsNanForAscendC(gert::TilingContext* context)
{
    OP_LOGD("IsNanTiling", "Enter new IsNanTiling");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (static_cast<int32_t>(coreNum) <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (static_cast<int64_t>(ubSize) <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);

    IsNanTiling IsNanTiling(context);
    return IsNanTiling.RunTiling();
}

static ge::graphStatus Tiling4IsNan(gert::TilingContext* context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto compile_info = reinterpret_cast<const IsNanCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return Tiling4IsNanForAscendC(context);
}
IMPL_OP_OPTILING(IsNan).Tiling(Tiling4IsNan).TilingParse<IsNanCompileInfo>(TilingPrepare4IsNan);
} // namespace optiling