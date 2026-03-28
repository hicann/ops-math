/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file asin_tiling_arch35.cpp
 * \brief Asin tiling implementation
 */
#include "asin_tiling_arch35.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/asin/op_kernel/arch35/asin_dag.h"
#include "op_host/tiling_util.h"

namespace optiling {
using namespace ge;
using namespace AsinDag;

constexpr uint64_t WORKSPACE_RESERVE_BYTE = 0;  // Asin 无需 workspace

ge::graphStatus AsinTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_RESERVE_BYTE;
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsinTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(inputDtype != this->outputDtype,
                OP_LOGE(tilingContext, "input and output dtype is diff, check failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsinTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "AsinTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputShape = Ops::Math::OpTiling::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputShape = Ops::Math::OpTiling::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputShape != outputShape,
                OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsinTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<AsinTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<AsinOpWithCast<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<AsinOpDirect<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<AsinOpWithCast<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext, "data type check failed. dtype: %d", this->outputDtype);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext, "DoTiling failed"), return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForAsin(gert::TilingContext* context)
{
    OP_LOGD("AsinTiling", "Enter TilingForAsin");
    OP_CHECK_IF(context == nullptr,
                OP_LOGE(context, "Tiling context is null"),
                return ge::GRAPH_FAILED);

    OP_LOGD("AsinTiling", "Enter new AsinTiling");
    AsinTiling asinTiling(context);
    return asinTiling.RunTiling();
}

ge::graphStatus TilingPrepareForAsin([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Asin).Tiling(TilingForAsin).TilingParse<ElewiseCompileInfo>(TilingPrepareForAsin);
}  // namespace optiling
