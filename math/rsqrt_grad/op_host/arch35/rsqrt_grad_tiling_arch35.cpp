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
 * \file rsqrt_grad_regbase_optiling.cc
 * \brief
 */
#include "rsqrt_grad_tiling_arch35.h"
#include "tiling_base/tiling_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "math/rsqrt_grad/op_kernel/arch35/rsqrt_grad_dag.h"
#include "math/rsqrt_grad/op_kernel/arch35/rsqrt_grad_struct.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/rsqrt_grad_tilingdata.h"

#include <iostream>

using namespace RsqrtGradOp;

namespace optiling
{
using namespace Ops::Math::OpTiling;
using namespace Ops::Base;

const int64_t ASCEND_WORKSPACE = static_cast<int64_t>(16) * 1024 * 1024;

class RsqrtGradTiling
{
public:
    explicit RsqrtGradTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
    ge::DataType inputDtype1;
    uint64_t dType = 0;
    RsqrtGradTilingData* tiling = nullptr;
};

ge::graphStatus RsqrtGradTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto inputDesc1 = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc1);
    this->inputDtype1 = inputDesc1->GetDataType();

    OP_CHECK_IF(this->inputDtype1 != this->inputDtype,
               OP_LOGE(tilingContext->GetNodeName(), "input dy dtype not same as input y"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtGradTiling::CheckShape()
{
    auto yStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& inputYShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    auto dyStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dyStorageShape);
    const gert::Shape& inputDYShape = EnsureNotScalar(dyStorageShape->GetStorageShape());

    auto zStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, zStorageShape);
    const gert::Shape& outputZShape = EnsureNotScalar(zStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != inputDYShape,
               OP_LOGE(tilingContext->GetNodeName(), "input y and dy shape not same"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputYShape != outputZShape,
               OP_LOGE(tilingContext->GetNodeName(), "input y and output z shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtGradTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE(tilingContext->GetNodeName(), "output z dtype not same as input y"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtGradTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "RsqrtGradTiling RunTiling enter.");

    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<RsqrtGradTilingData>();
    OP_CHECK_IF((tiling == nullptr),
                    OP_LOGE(tilingContext->GetNodeName(), "Get RsqrtGradTiling from GE context failed"),
                    return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtGradDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtGradDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtGradDAG<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        dType = TPL_INT8;
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtGradInt8<int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        dType = TPL_INT32;
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtGradWithDiv<int32_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult != ge::GRAPH_SUCCESS,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->baseTiling.scheMode), dType);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForRsqrtGrad(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<RsqrtGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4RsqrtGrad(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4RsqrtGrad rt2.0 is running.");
    auto compileInfo = static_cast<const RsqrtGradCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    RsqrtGradTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

IMPL_OP_OPTILING(RsqrtGrad).Tiling(Tiling4RsqrtGrad).TilingParse<RsqrtGradCompileInfo>(TilingPrepareForRsqrtGrad);
}  // namespace optiling