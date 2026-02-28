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
 * \file expand_tiling_arch35.cpp
 * \brief calc tiling for expand
 */
#include "expand_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/util/const_util.h"
#include "op_host/tiling_util.h"
#include "util/platform_util.h"

namespace optiling {
constexpr size_t MAX_DIM_NUM = 0x8;
constexpr size_t BRCTO_MAX_DIM_NUM = 0x8;

template <typename T>
static std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus CheckExpandRule(
    const gert::TilingContext* context, const gert::Shape& inShape, const gert::Shape& outShape)
{
    auto outDimNum = outShape.GetDimNum();
    OP_CHECK_IF(
        inShape.GetDimNum() != outDimNum,
        OP_LOGE(context->GetNodeName(), "The input shape dims are different with output's!"), return ge::GRAPH_FAILED);

    for (size_t i = 0; i < outDimNum; i++) {
        if (inShape[i] != 1 && outShape[i] != inShape[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void AdjustShapesToSameDimNum(gert::Shape& inShape, size_t outDimNum)
{
    auto inDimNum = inShape.GetDimNum();
    if (inDimNum >= outDimNum) {
        return;
    }

    gert::Shape newShape;
    size_t gapSize = outDimNum - inDimNum;
    for (size_t i = 0; i < gapSize; i++) {
        newShape.AppendDim(1);
    }
    for (size_t j = 0; j < inDimNum; j++) {
        newShape.AppendDim(inShape[j]);
    }
    inShape = newShape;
}

ge::graphStatus DeleteOneSizeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto dimNum = inShape.GetDimNum();
    OP_CHECK_IF(
        dimNum != outShape.GetDimNum(),
        OP_LOGE(context->GetNodeName(), "The input shape dims are different with output's!"), return ge::GRAPH_FAILED);

    if (dimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }

    size_t mIdx = 0;
    for (size_t oIdx = 0; oIdx < dimNum; oIdx++) {
        if (outShape[oIdx] != 1) {
            inShape[mIdx] = inShape[oIdx];
            outShape[mIdx] = outShape[oIdx];
            mIdx += size_t(1);
        }
    }

    if (mIdx == size_t(0)) {
        inShape[0] = 1;
        outShape[0] = 1;
        mIdx += size_t(1);
    }
    inShape.SetDimNum(mIdx);
    outShape.SetDimNum(mIdx);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetABFlag(const gert::TilingContext* context, const gert::Shape& inShape, const gert::Shape& outShape,
                          std::array<bool, MAX_DIM_NUM>& abInfo)
{
    auto inDimNum = inShape.GetDimNum();
    OP_CHECK_IF(
        inDimNum != outShape.GetDimNum(),
        OP_LOGE(context->GetNodeName(), "The input shape dims are different with output's!"),
        return ge::GRAPH_FAILED);

    for (size_t idx = 0; idx < inDimNum; idx++) {
        abInfo[idx] = (inShape[idx] != outShape[idx]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MergeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto dimNum = inShape.GetDimNum();
    OP_CHECK_IF(
        dimNum != outShape.GetDimNum(),
        OP_LOGE(context->GetNodeName(), "The input shape dims are different with output's!"), return ge::GRAPH_FAILED);

    if (dimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }

    std::array<bool, MAX_DIM_NUM> abInfo{};
    OP_CHECK_IF(
        GetABFlag(context, inShape, outShape, abInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to get axes info."), return ge::GRAPH_FAILED);

    size_t mIdx = 0;
    for (size_t oIdx = 1; oIdx < dimNum; oIdx++) {
        if (abInfo[oIdx] == abInfo[mIdx]) {
            inShape[mIdx] *= inShape[oIdx];
            outShape[mIdx] *= outShape[oIdx];
        } else {
            mIdx += 1;
            inShape[mIdx] = inShape[oIdx];
            outShape[mIdx] = outShape[oIdx];
            abInfo[mIdx] = abInfo[oIdx];
        }
    }
    inShape.SetDimNum(mIdx + 1);
    outShape.SetDimNum(mIdx + 1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapeInfo(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    inShape = Ops::Math::OpTiling::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    outShape = Ops::Math::OpTiling::EnsureNotScalar(yStorage->GetStorageShape());

    auto outDimNum = outShape.GetDimNum();
    OP_CHECK_IF(
        inShape.GetDimNum() > outDimNum,
        OP_LOGE(context->GetNodeName(), "The input shape has more dimensions than output shape!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outDimNum > BRCTO_MAX_DIM_NUM, OP_LOGE(context->GetNodeName(), "Not support the dim num: %lu yet!", outDimNum),
        return ge::GRAPH_FAILED);
    AdjustShapesToSameDimNum(inShape, outDimNum);
    OP_CHECK_IF(
        inShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
        OP_LOGE(context->GetNodeName(), "The input or output shape is empty!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckExpandRule(context, inShape, outShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The input and output shapes mismatch the broadcast rule!"),
        return ge::GRAPH_FAILED);
    OP_LOGI(
        context->GetNodeName(), "The input and output is: %s and %s", Shape2String(inShape).c_str(),
        Shape2String(outShape).c_str());

    OP_CHECK_IF(
        DeleteOneSizeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to delete one size axes!"), return ge::GRAPH_FAILED);
    OP_LOGI(
        context->GetNodeName(), "The reshaped input and output is: %s and %s", Shape2String(inShape).c_str(),
        Shape2String(outShape).c_str());

    OP_CHECK_IF(
        MergeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to merge axes!"), return ge::GRAPH_FAILED);
    OP_LOGI(
        context->GetNodeName(), "The merged input and output is: %s and %s", Shape2String(inShape).c_str(),
        Shape2String(outShape).c_str());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ExpandAscendC(
    gert::TilingContext* context, const gert::Shape* inShapePtr, const gert::Shape* outShapePtr)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    brcto::BroadcastToTilingAscendC brcToTiling(context, inShapePtr, outShapePtr);

    OP_CHECK_IF(
        (brcToTiling.GetHardwareInfo<ExpandCompileInfo>() != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "BroadcastToTilingAscendC failed to get hardware info."),
        return ge::GRAPH_FAILED);

    return brcToTiling.DoTiling();
}

static ge::graphStatus Tiling4Expand(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const ExpandCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    gert::Shape inShape;
    gert::Shape outShape;

    OP_CHECK_IF(
        GetShapeInfo(context, inShape, outShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get input or output shape was failed!"), return ge::GRAPH_FAILED);
    return Tiling4ExpandAscendC(context, &inShape, &outShape);
}

static ge::graphStatus TilingPrepare4Expand(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4Expand.");

    auto compileInfo = context->GetCompiledInfo<ExpandCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is negative."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    compileInfo->clSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->clSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get cache line size."),
        return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->blockSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get block size."),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        (compileInfo->vRegSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get vReg size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4Expand.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Expand op.
IMPL_OP_OPTILING(Expand).Tiling(Tiling4Expand).TilingParse<ExpandCompileInfo>(TilingPrepare4Expand);
} // namespace optiling
