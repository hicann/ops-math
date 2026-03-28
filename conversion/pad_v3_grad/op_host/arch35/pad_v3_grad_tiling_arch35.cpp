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
 * \file pad_v3_grad_tiling_arch35.cpp
 * \brief ac pad v3 grad tiling cpp
 */

#include "pad_v3_grad_tiling_arch35.h"
#include "util/platform_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace AscendC;

namespace optiling {

static constexpr uint64_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr size_t PADDINGS_IDX = 1;
static constexpr size_t PAIR = 2;
static constexpr uint8_t MAX_DIM_NUM = 8;
static constexpr uint8_t NON_CONSTANT_MAX_DIM_NUM = 5;
static constexpr uint64_t EXPANSION_FACTOR = 2;

template <typename T>
std::string PadV3GradACTiling::ToString(const T* value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ",";
    }
    r = r + "]";
    return r;
}

void PadV3GradACTiling::DoTilingWithSIMTMirror()
{
    isBigShape_ = false;
    if (inShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX || outShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX) {
        isBigShape_ = true;
    }
}
void PadV3GradACTiling::DoTilingWithSIMTEdge()
{
    isBigShape_ = false;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        isBigShape_ = true;
    }
}
void PadV3GradACTiling::DoTilingWithSIMTCircular()
{
    isBigShape_ = false;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        isBigShape_ = true;
    }
}
void PadV3GradACTiling::DoTilingWithSIMTConstant()
{
    isBigShape_ = false;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        isBigShape_ = true;
    }
}

void PadV3GradACTiling::FillsAndPrintTilingData()
{
    tilingData_->dimNum = dimNum_;
    OP_LOGI(
        context_,
        "tilingData is dimNum = %u, inShape: %s, outShape: %s, inStride: %s, outStride: %s, leftPad: %s, rightPad: %s, "
        "tilingKey: %lu, coreNum is %u dtype:%s",
        tilingData_->dimNum, ToString(tilingData_->inShape, dimNum_).c_str(),
        ToString(tilingData_->outShape, dimNum_).c_str(), ToString(tilingData_->inStride, dimNum_).c_str(),
        ToString(tilingData_->outStride, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
        ToString(tilingData_->rightPad, dimNum_).c_str(), tilingKey_, coreNum_,
        Ops::Base::ToString(paramsDtype_).c_str());
}

void PadV3GradACTiling::EmptyTensorCollapse()
{
    OP_LOGD(context_, "Start PadV3GradACTiling EmptyTensorCollapse.");
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = outShapeSize_;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
    tilingData_->rightPad[0] = 0;
}

ge::graphStatus PadV3GradACTiling::ComputeAfterPaddingsAndStrides()
{
    inShapeSize_ = 1UL;
    outShapeSize_ = 1UL;
    for (int64_t i = dimNum_ - 1; i >= 0; --i) {
        // 校验是否会出现pad为负且比inshape大的情况

        if (static_cast<int64_t>(tilingData_->inShape[i]) - tilingData_->leftPad[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) - tilingData_->rightPad[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) - tilingData_->leftPad[i] - tilingData_->rightPad[i] < 0) {
            OP_LOGE(context_, "outShape length must be non-negative.");
            return ge::GRAPH_FAILED;
        }

        // compute shape after padding
        tilingData_->outShape[i] = tilingData_->inShape[i] - tilingData_->leftPad[i] - tilingData_->rightPad[i];
        // compute src stride
        tilingData_->inStride[i] = inShapeSize_;
        inShapeSize_ *= tilingData_->inShape[i];
        // compute dst stride
        tilingData_->outStride[i] = outShapeSize_;
        outShapeSize_ *= tilingData_->outShape[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::CheckModeInputParam(int64_t inShapeV, int64_t padFront, int64_t padBack)
{
    int64_t outShapeV = inShapeV - padFront - padBack;
    int64_t leftsubin = padFront - outShapeV;
    int64_t rightsubin = padBack - outShapeV;
    if (padMode_ == TPL_MODE_REFLECT && (0 < (leftsubin + 1) || 0 < (rightsubin + 1))) {
        OP_LOGE(
            context_,
            "reflect mode padFront:%ld and padBack:%ld cannot be bigger than InferredOutShape - 1, "
            "InferredOutShape:%ld.",
            padFront, padBack, outShapeV);
        return ge::GRAPH_FAILED;
    }
    if (padMode_ == TPL_MODE_SYMMETRIC && (0 < leftsubin || 0 < rightsubin)) {
        OP_LOGE(
            context_, "symmetric mode padFront:%ld and padBack:%ld cannot be bigger than InferredOutShape:%ld.",
            padFront, padBack, outShapeV);
        return ge::GRAPH_FAILED;
    }

    if (padMode_ == TPL_MODE_CIRCULAR && (0 < leftsubin || 0 < rightsubin)) {
        OP_LOGE(
            context_, "circular mode padFront:%ld and padBack:%ld cannot be bigger than InferredOutShape:%ld.",
            padFront, padBack, outShapeV);
        return ge::GRAPH_FAILED;
    }

    if (padMode_ == TPL_MODE_EDGE && outShapeV == 0 && (padFront != 0 || padBack != 0)) {
        OP_LOGE(
            context_, "When InferredOutShape == 0, edge mode padFront:%ld and padBack:%ld must be 0.", padFront,
            padBack);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DimensionCollapseMode()
{
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(
        context_, "Before collapse paddings, padMode_:%u dimNum is %u, shape is %s, left pad is %s, right pad is %s",
        padMode_, originalRank, ToString(tilingData_->inShape, originalRank).c_str(),
        Ops::Base::ToString(paddings_.padFront).c_str(), Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
        int64_t padFront = paddings_.padFront.GetDim(fastDim);
        int64_t padBack = paddings_.padBack.GetDim(fastDim);

        OP_CHECK_IF(
            CheckModeInputParam(tilingData_->inShape[fastDim], padFront, padBack) == ge::GRAPH_FAILED,
            OP_LOGE(context_, "CheckModeInputParam failed."), return ge::GRAPH_FAILED);

        // 消除1轴
        if (tilingData_->inShape[fastDim] == 1 && (tilingData_->inShape[fastDim] + padFront + padBack) == 1) {
            fastDim++;
            dimNum_--;
            continue;
        }

        uint64_t collapsedShape = tilingData_->inShape[fastDim];
        int64_t collapsedPadFront = paddings_.padFront.GetDim(fastDim);
        int64_t collapsedPadBack = paddings_.padBack.GetDim(fastDim);
        fastDim++;
        if (0 == paddings_.padFront.GetDim(fastDim - 1) && 0 == paddings_.padBack.GetDim(fastDim - 1)) {
            while (fastDim < originalRank &&
                   (0 == paddings_.padFront.GetDim(fastDim) && 0 == paddings_.padBack.GetDim(fastDim))) {
                collapsedShape *= tilingData_->inShape[fastDim];
                collapsedPadFront *= tilingData_->inShape[fastDim];
                collapsedPadBack *= tilingData_->inShape[fastDim];
                fastDim++;
                dimNum_--;
            }
        }
        tilingData_->inShape[slowDim] = collapsedShape;
        tilingData_->leftPad[slowDim] = collapsedPadFront;
        tilingData_->rightPad[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    // shape全1且不补pad的场景,消除1轴后就为空了
    if (dimNum_ == 0) {
        tilingData_->inShape[0] = 1;
        tilingData_->leftPad[0] = 0;
        tilingData_->rightPad[0] = 0;
        dimNum_ = 1;
    }
    OP_LOGD(
        context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
        ToString(tilingData_->rightPad, dimNum_).c_str());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DimensionCollapse()
{
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(
        context_, "Before collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", originalRank,
        ToString(tilingData_->inShape, originalRank).c_str(), Ops::Base::ToString(paddings_.padFront).c_str(),
        Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
        uint64_t collapsedShape = tilingData_->inShape[fastDim];
        int64_t collapsedPadFront = paddings_.padFront.GetDim(fastDim);
        int64_t collapsedPadBack = paddings_.padBack.GetDim(fastDim);
        fastDim++;
        while (fastDim < originalRank &&
               (0 == paddings_.padFront.GetDim(fastDim) && 0 == paddings_.padBack.GetDim(fastDim))) {
            collapsedShape *= tilingData_->inShape[fastDim];
            collapsedPadFront *= tilingData_->inShape[fastDim];
            collapsedPadBack *= tilingData_->inShape[fastDim];
            fastDim++;
            dimNum_--;
        }

        tilingData_->inShape[slowDim] = collapsedShape;
        tilingData_->leftPad[slowDim] = collapsedPadFront;
        tilingData_->rightPad[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    OP_LOGD(
        context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
        ToString(tilingData_->rightPad, dimNum_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::DoTilingModeEdge()
{
    OP_CHECK_IF(
        DimensionCollapseMode() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling Edge Collapse error."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "PadV3GradACTiling Edge ComputeAfterPaddingsAndStrides error."), return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    DoTilingWithSIMTEdge();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DoTilingModeMirror()
{
    OP_CHECK_IF(
        DimensionCollapseMode() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling Mirror Collapse error."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "PadV3GradACTiling Mirror ComputeAfterPaddingsAndStrides error."), return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    DoTilingWithSIMTMirror();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DoTilingModeCircular()
{
    OP_CHECK_IF(
        DimensionCollapseMode() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling Circular Collapse error."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "PadV3GradACTiling Circular ComputeAfterPaddingsAndStrides error."), return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    DoTilingWithSIMTCircular();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DoTilingModeConstant()
{
    OP_CHECK_IF(
        DimensionCollapse() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling Constant Collapse error."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "PadV3GradACTiling Constant ComputeAfterPaddingsAndStrides error."), return ge::GRAPH_FAILED);
    // 空tensor时，合为一根轴
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    DoTilingWithSIMTConstant();
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void PadV3GradACTiling::GetPaddingsToShape(const gert::Tensor* paddingsTensor)
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo GetPaddings GetPaddingsToShape.");
    const T* paddingsValue = paddingsTensor->GetData<T>();
    const size_t paddingsNum = paddingsTensor->GetShapeSize();

    size_t inputDimNum = paddingsNum / PAIR;
    paddings_.padFront.SetDimNum(inputDimNum);
    paddings_.padBack.SetDimNum(inputDimNum);
    size_t indexCof = 1;
    size_t indexOffset = inputDimNum;
    // choose diff padding handle method by paddings_contiguous
    if (paddingContiguous_) {
        indexCof = PAIR;
        indexOffset = 1;
    }
    for (size_t i = 0; i < inputDimNum; ++i) {
        paddings_.padFront.SetDim(i, paddingsValue[i * indexCof]);
        paddings_.padBack.SetDim(i, paddingsValue[i * indexCof + indexOffset]);
    }
}

ge::graphStatus PadV3GradACTiling::GetPaddings()
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo GetPaddings.");
    const gert::Tensor* paddingsTensor = context_->GetInputTensor(PADDINGS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingsTensor);
    ge::DataType paddingsDtype = paddingsTensor->GetDataType();
    switch (paddingsDtype) {
        case ge::DT_INT32: {
            GetPaddingsToShape<int32_t>(paddingsTensor);
            return ge::GRAPH_SUCCESS;
        }
        case ge::DT_INT64: {
            GetPaddingsToShape<int64_t>(paddingsTensor);
            return ge::GRAPH_SUCCESS;
        }
        default:
            OP_LOGE(
                context_, "Paddings only support [int32, int64]. but is %s",
                Ops::Base::ToString(paddingsDtype).c_str());
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus PadV3GradACTiling::GetShapesAndDtypes()
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo GetShapesAndDtypes.");
    // get input shape & input shape's dim num
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    dimNum_ = inShapeVal.GetDimNum(); // 获取维度
    if (dimNum_ > MAX_DIM_NUM) {
        OP_LOGE(context_, "input shape dim should <= 8, please check.");
        return ge::GRAPH_FAILED;
    }
    if (padMode_ != TPL_MODE_CONSTANT && dimNum_ > NON_CONSTANT_MAX_DIM_NUM) {
        OP_LOGE(context_, "non-constant mode, the input shape dim should <= 5, please check.");
        return ge::GRAPH_FAILED;
    }
    for (uint16_t i = 0; i < dimNum_; ++i) {
        if (inShapeVal.GetDim(i) < 0) {
            OP_LOGE(context_, "Input shape should >= 0, please check");
            return ge::GRAPH_FAILED;
        }
        if (inShapeVal.GetDim(i) == 0) {
            isEmptyTensor_ = true;
        }
        tilingData_->inShape[i] = inShapeVal.GetDim(i); // 获取inShape
    }
    // 判断类型
    auto inputTensor = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputTensor);
    paramsDtype_ = inputTensor->GetDataType();
    dtypeBytes_ = GetSizeByDataType(paramsDtype_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto* mode = attrs->GetAttrPointer<char>(0);
    // padv3grad mode可选，默认reflect;
    if (mode) {
        if (!strcmp(mode, "constant")) {
            padMode_ = TPL_MODE_CONSTANT;
        } else if (!strcmp(mode, "edge")) {
            padMode_ = TPL_MODE_EDGE;
        } else if (!strcmp(mode, "symmetric")) {
            padMode_ = TPL_MODE_SYMMETRIC;
        } else if (!strcmp(mode, "circular")) {
            padMode_ = TPL_MODE_CIRCULAR;
        }
        OP_CHECK_IF(
            strcmp(mode, "constant") != 0 && strcmp(mode, "edge") != 0 && strcmp(mode, "reflect") != 0 &&
                strcmp(mode, "symmetric") != 0 && strcmp(mode, "circular") != 0,
            OP_LOGE(context_, "PadV3Grad only support constant/edge/reflect/symmetric/circular mode."),
            return ge::GRAPH_FAILED);
    }

    auto* paddingContiguous = attrs->GetAttrPointer<bool>(1);
    if (paddingContiguous) {
        paddingContiguous_ = *paddingContiguous;
    }

    OP_CHECK_IF(
        GetShapesAndDtypes() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "PadV3GradACTiling GetShapeAttrsInfo GetShapesAndDtypes error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetPaddings() == ge::GRAPH_FAILED, OP_LOGE(context_, "Get padding info failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::Init()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0U, OP_LOGE(context_, "Failed to core num."), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0U, OP_LOGE(context_, "Failed to ub size."), return ge::GRAPH_FAILED);
    blockSize_ = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    OP_CHECK_IF(blockSize_ == 0U, OP_LOGE(context_, "Failed to ub block size."), return ge::GRAPH_FAILED);
    vectorSize_ = static_cast<uint64_t>(Ops::Base::GetVRegSize(context_));
    OP_CHECK_IF(vectorSize_ == 0U, OP_LOGE(context_, "Failed to vector size."), return ge::GRAPH_FAILED);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(cacheLineSize_ == 0U, OP_LOGE(context_, "Failed to cache line size."), return ge::GRAPH_FAILED);
    OP_LOGD(
        context_,
        "platform info is coreNum_ = %u, ubSize_ = %lu, blockSize_ = %lu, vectorSize_ = %lu, cacheLineSize_ = %u",
        coreNum_, ubSize_, blockSize_, vectorSize_, cacheLineSize_);
    tilingData_ = context_->GetTilingData<PadV3GradACTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::DoTiling()
{
    OP_LOGD(context_, "Start PadV3GradACTiling DoTiling.");
    OP_CHECK_IF(
        Init() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling Init error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetShapeAttrsInfo() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling GetShapeAttrsInfo error."),
        return ge::GRAPH_FAILED);
    if (padMode_ == TPL_MODE_EDGE) {
        OP_CHECK_IF(
            DoTilingModeEdge() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeEdge failed."),
            return ge::GRAPH_FAILED);
    } else if (padMode_ == TPL_MODE_REFLECT || padMode_ == TPL_MODE_SYMMETRIC) {
        OP_CHECK_IF(
            DoTilingModeMirror() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeMirror failed."),
            return ge::GRAPH_FAILED);
    } else if (padMode_ == TPL_MODE_CIRCULAR) {
        OP_CHECK_IF(
            DoTilingModeCircular() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeCircular failed."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            DoTilingModeConstant() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeConstant failed."),
            return ge::GRAPH_FAILED);
    }

    tilingKey_ = GET_TPL_TILING_KEY(padMode_, isBigShape_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is %lu, modeName %d, isBigShape %d", tilingKey_, padMode_, isBigShape_);

    FillsAndPrintTilingData();
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORK_SPACE_SIZE;

    OP_LOGD(context_, "Exit PadV3GradACTiling DoTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PadV3GradTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "PadV3GradTiling running begin");
    const PadV3GradCompileInfo* compile_info = reinterpret_cast<const PadV3GradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Tiling4Pad dsl compile_info is Null, running AscendC tiling.");
    PadV3GradACTiling tilingObject(context);
    return tilingObject.DoTiling();
}

ge::graphStatus TilingPreparePadV3GradForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init PadV3Grad AscendC Tiling.");
    auto ci = context->GetCompiledInfo<PadV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->core_num <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ub_size <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3Grad(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<PadV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4PadV3Grad Mode success!");
    TilingPreparePadV3GradForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the PadV3Grad op.
IMPL_OP_OPTILING(PadV3Grad).Tiling(PadV3GradTiling).TilingParse<PadV3GradCompileInfo>(TilingPrepare4PadV3Grad);
} // namespace optiling
