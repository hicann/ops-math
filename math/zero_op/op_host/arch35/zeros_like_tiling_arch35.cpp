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
 * \file zeros_like_tiling_arch35.cpp
 * \brief
 */
#include "zeros_like_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_util.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "conversion/zeros_like/op_kernel/arch35/zeros_like_dag.h"
#include "conversion/zeros_like/op_kernel/arch35/zeros_like_tiling_key.h"

#include <iostream>

using namespace Ops::Math::OpTiling;
using namespace ZerosLikeNs;

namespace optiling
{
static const size_t ASCEND_WORKSPACE = 16 * 1024 * 1024;

ge::graphStatus ZerosLikeTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling SetTilingData enter.");
    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ZerosLikeTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    static const std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT,
                                                          ge::DT_BOOL, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8};
    auto inputTypeCheck = std::find(inputDtypes.begin(), inputDtypes.end(), this->inputDtype);

    OP_CHECK_IF(inputTypeCheck == inputDtypes.end(),
               OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support %d", this->inputDtype),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ZerosLikeTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputXShape.GetShapeSize() != outputYShape.GetShapeSize(),
               OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ZerosLikeTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling CalcOutputDtype enter.");

    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as input x"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ZerosLikeTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<ZerosLikeTilingData>();
    if (this->outputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT8 || this->outputDtype == ge::DT_BOOL) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int32_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int64_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<uint8_t>::OpDag>(tiling->baseTiling);
    }else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4ZerosLike(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4ZerosLike rt2.0 is running.");
    auto compileInfo = static_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    ZerosLikeTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForZerosLike(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ZerosLike).Tiling(Tiling4ZerosLike).TilingParse<ElewiseCompileInfo>(TilingPrepareForZerosLike);
}  // namespace optiling