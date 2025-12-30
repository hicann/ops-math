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
 * \file round_tiling_arch35.cpp
 * \brief
 */
#include "round_tiling_arch35.h"
#include "tiling_base/tiling_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "math/round/op_kernel/arch35/round_dag.h"
#include "math/round/op_kernel/arch35/round_struct.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"

#include <iostream>

using namespace RoundOp;
namespace optiling
{
using namespace Ops::Math::OpTiling;

const uint64_t ASCEND_WORKSPACE = 16777216;
const std::int32_t ATTR_ROUND_DECIMALS_POS = 0;
const int64_t DEFAULT_ZERO = 0;
const float DEFAULT_FP32_ZERO = 0.0;
const int64_t DEFAULT_TEN = 10;
const int64_t DEFAULT_THIRTY_EIGHT = 38;
const float DEFAULT_INF = INFINITY;

class RoundTiling
{
public:
    explicit RoundTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    RoundTilingData* tiling_ = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();
    ge::graphStatus DoTiling(bool decimalsNeg, bool decimalsNan);

private:
    uint64_t dType = 0;
    uint64_t schMode = 0;
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
};

ge::graphStatus RoundTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "RoundTiling SetTilingData enter.");
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);

    schMode = tiling_->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling_->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoundTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16
        && this->inputDtype != ge::DT_FLOAT && this->inputDtype != ge::DT_INT32,
        OP_LOGE(tilingContext, "input x dtype[%s] not support",
        ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoundTiling::CheckShape()
{
    auto xStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputResultShape = EnsureNotScalar(xStorageShape->GetStorageShape());

    auto yStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& outputYShape = EnsureNotScalar(yStorageShape->GetStorageShape());

    OP_CHECK_IF(inputResultShape != outputYShape,
               OP_LOGE(tilingContext, "input x and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoundTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16
        && this->outputDtype != ge::DT_FLOAT && this->outputDtype != ge::DT_INT32,
        OP_LOGE(tilingContext, "output y dtype[%s] not support",
        ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE(tilingContext, "output y dtype[%s] not same as input x [%s]",
               ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str(),
               ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoundTiling::DoTiling(bool decimalsNeg, bool decimalsNan)
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        if (tiling_->decimals == DEFAULT_FP32_ZERO) {
            dType = static_cast<uint64_t>(ROUND_TPL_ZERO);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundZero<half>::OpDag>(tiling_->baseTiling);
        } else if (decimalsNeg) {
            dType = static_cast<uint64_t>(ROUND_TPL_NEGATIVE_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundNegativeDecimals<half>::OpDag>(tiling_->baseTiling);
        } else if (decimalsNan) {
            dType = static_cast<uint64_t>(ROUND_TPL_NAN_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundNan<half>::OpDag>(tiling_->baseTiling);
        } else {
            dType = static_cast<uint64_t>(ROUND_TPL_POSITIVE_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundPositiveDecimals<half>::OpDag>(tiling_->baseTiling);
        }
    } else if (this->outputDtype == ge::DT_BF16) {
        if (tiling_->decimals == DEFAULT_FP32_ZERO) {
            dType = static_cast<uint64_t>(ROUND_TPL_ZERO);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundZero<bfloat16_t>::OpDag>(tiling_->baseTiling);
        } else if (decimalsNeg) {
            dType = static_cast<uint64_t>(ROUND_TPL_NEGATIVE_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundNegativeDecimals<bfloat16_t>::OpDag>(tiling_->baseTiling);
        } else if (decimalsNan) {
            dType = static_cast<uint64_t>(ROUND_TPL_NAN_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundNan<bfloat16_t>::OpDag>(tiling_->baseTiling);
        } else {
            dType = static_cast<uint64_t>(ROUND_TPL_POSITIVE_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundPositiveDecimals<bfloat16_t>::OpDag>(tiling_->baseTiling);
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (tiling_->decimals == DEFAULT_FP32_ZERO) {
            dType = static_cast<uint64_t>(ROUND_TPL_ZERO);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundZero<float>::OpDag>(tiling_->baseTiling);
        } else if (decimalsNeg) {
            dType = static_cast<uint64_t>(ROUND_TPL_NEGATIVE_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundNegativeDecimals<float>::OpDag>(tiling_->baseTiling);
        } else if (decimalsNan) {
            dType = static_cast<uint64_t>(ROUND_TPL_NAN_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundNan<float>::OpDag>(tiling_->baseTiling);
        } else {
            dType = static_cast<uint64_t>(ROUND_TPL_POSITIVE_DECIMALS);
            baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundPositiveDecimals<float>::OpDag>(tiling_->baseTiling);
        }
    } else if (this->outputDtype == ge::DT_INT32) {
        dType = static_cast<uint64_t>(ROUND_TPL_INT32);
        baseTilingResult = elewiseBaseTiling.DoTiling<RoundDag::RoundInt<int>::OpDag>(tiling_->baseTiling);
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoundTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "RoundTiling RunTiling enter.");
    tiling_ = tilingContext->GetTilingData<RoundTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling_);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input x dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output y dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);
    auto runtimeAttrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runtimeAttrs);
    const int64_t *decimalsPtr = runtimeAttrs->GetAttrPointer<int64_t>(ATTR_ROUND_DECIMALS_POS);
    bool decimalsNeg = false;
    bool decimalsNan = false;
    if (decimalsPtr != nullptr) {
        if (*decimalsPtr < DEFAULT_ZERO) {
            decimalsNeg = true;
        }
        if (*decimalsPtr == DEFAULT_ZERO) {
            tiling_->decimals = DEFAULT_FP32_ZERO;
        } else {
            if (llabs(*decimalsPtr) > DEFAULT_THIRTY_EIGHT) {
                tiling_->decimals = DEFAULT_INF;
                decimalsNan = true;
                decimalsNeg = false;
            } else {
                tiling_->decimals = pow(DEFAULT_TEN, llabs(*decimalsPtr));
            }
        }
    } else {
        tiling_->decimals = DEFAULT_FP32_ZERO;
    }
    DoTiling(decimalsNeg, decimalsNan);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForRound(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<RoundCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Round(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "Tiling context is null."),
        return ge::GRAPH_FAILED);
        
    OP_LOGD(context->GetNodeName(), "Tiling4Round rt2.0 is running.");

    auto compileInfo = static_cast<const RoundCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "Enter new Round.");
    RoundTiling baseOpTiling(context);
    return baseOpTiling.RunTiling();
}

IMPL_OP_OPTILING(Round).Tiling(Tiling4Round).TilingParse<RoundCompileInfo>(TilingPrepareForRound);
}  // namespace optiling