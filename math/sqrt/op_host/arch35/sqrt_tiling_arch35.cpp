/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sqrt_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/sqrt_dag.h"
#include "../../op_kernel/arch35/sqrt_struct.h"

#include <iostream>

using namespace SqrtOp;
using namespace Ops::Base;
namespace optiling {
constexpr size_t SYS_WORKSPACE = 16777216; //16M

ge::graphStatus SqrtTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "SqrtTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqrtTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "SqrtTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Math::OpTiling::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Math::OpTiling::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != outputZShape,
               OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqrtTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "SqrtTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE(tilingContext->GetNodeName(), "output y dtype not same as input x"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqrtTiling::RunTiling()
{
    auto tiling = tilingContext->GetTilingData<EleBaseTilingData16B>();
    OP_LOGD(tilingContext->GetNodeName(), "SqrtTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<SqrtDAG<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<SqrtDAG<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<SqrtDAG<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Sqrt(gert::TilingContext *context)
{
    OP_LOGD("SqrtTiling", "Enter Tiling4Sqrt");
    if (context == nullptr) {
        OP_LOGE("SqrtTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const SqrtCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("SqrtTiling", "Enter new SqrtTiling");
    SqrtTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForSqrt(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SqrtCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sqrt).Tiling(Tiling4Sqrt)
                            .TilingParse<SqrtCompileInfo>(TilingPrepareForSqrt);

}