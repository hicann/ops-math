/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reciprocal_tiling_arch35.cpp
 * \brief
 */
#include "reciprocal_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "math/reciprocal/op_kernel/arch35/reciprocal_dag.h"
#include "math/reciprocal/op_kernel/arch35/reciprocal_struct.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include <iostream>

namespace optiling
{
using namespace Ops::Base;
const int64_t ASCEND_WORKSPACE = 32;

ge::graphStatus ReciprocalTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->baseTiling.scheMode), dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
	return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReciprocalTiling::CalcInputDtype()
{
	OP_LOGD(tilingContext->GetNodeName(), "ReciprocalTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,

        OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReciprocalTiling::CheckShape()
{
	OP_LOGD(tilingContext->GetNodeName(), "ReciprocalTiling CheckShape enter.");
    auto xStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputXShape = EnsureNotScalar(xStorageShape->GetStorageShape());
 
    auto yStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(yStorageShape->GetStorageShape());

    OP_CHECK_IF(inputXShape != outputYShape,
               OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReciprocalTiling::CalcOutputDtype()
{
	OP_LOGD(tilingContext->GetNodeName(), "ReciprocalTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext, "output dtype not support"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as input y"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReciprocalTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReciprocalTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling = tilingContext->GetTilingData<ReciprocalNs::ReciprocalTilingData>();
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
		dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReciprocalDag::ReciprocalDag<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
		dType = TPL_BF16;
		baseTilingResult = elewiseBaseTiling.DoTiling<ReciprocalDag::ReciprocalDag<bfloat16_t>::OpDag>(tiling->baseTiling); 
    } else if (this->outputDtype == ge::DT_FLOAT) {
		dType = TPL_FP32;
		baseTilingResult = elewiseBaseTiling.DoTiling<ReciprocalDag::ReciprocalDag<float>::OpDag>(tiling->baseTiling);
	} else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Reciprocal(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Reciprocal rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    ReciprocalTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForReciprocal(gert::TilingParseContext *context)
{
    OP_LOGD("ElewiseTiling", "Enter TilingPrepareForReciprocal.");
    auto compileInfoPtr = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Reciprocal).Tiling(Tiling4Reciprocal).TilingParse<ElewiseCompileInfo>(TilingPrepareForReciprocal);
}  // namespace optiling
