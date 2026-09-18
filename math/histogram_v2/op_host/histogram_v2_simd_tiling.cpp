/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_simd_tiling.cpp
 * \brief Deterministic UB_FULL SIMD template (dhistv2). Selected ahead of SIMT
 *        when the output is fp32, the launch is deterministic, bins <= 127 and
 *        the local histogram fits in UB.
 */
#include "histogram_v2_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
constexpr int64_t INPUT_NUM = 3;
constexpr int64_t INPUT_IDX_X = 0;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int64_t BINS_IDX = 0;
constexpr int64_t SIZE_OF_INT32 = 4;
constexpr int64_t SIZE_OF_FLOAT32 = 4;
constexpr int64_t DEFAULT_BINS = 100;
constexpr int64_t SIMD_MAX_BINS = 127;
constexpr int64_t SIMD_SPLIT_ALIGN = 256;
constexpr int64_t SIMD_MIN_ELEMS_PER_CORE = 4096;
constexpr int64_t TILING_KEY_SIMD_UB_FULL = 2100;
constexpr int64_t OUTPUT_FP32_KEY_OFFSET = 10;
constexpr uint64_t SIMT_DCACHE_SIZE = 32 * 1024;
constexpr uint64_t SYSTEM_WORKSPACE = 16 * 1024 * 1024;

class HistogramV2SimdTiling : public HistogramV2BaseClass {
public:
    explicit HistogramV2SimdTiling(gert::TilingContext* context) : HistogramV2BaseClass(context) {};
    ~HistogramV2SimdTiling() override = default;
    void Reset(gert::TilingContext* context) override { HistogramV2BaseClass::Reset(context); }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus TilingDataForCore();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

    HistogramV2SimtTilingData tilingData;
    int64_t inputDtypeVal_ = 0;
    int64_t coreNum_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t totalLength_ = 0;
    bool isFp32Output_ = false;
    int64_t isDeterministic_ = 0;
    int64_t bins_ = 0;
    int64_t ubNumCanUse_ = 0;
    int64_t ubLoopNum_ = 0;
    int64_t needXCoreNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYTail_ = 0;
    uint64_t workspaceBytes_ = SYSTEM_WORKSPACE;
};

bool HistogramV2SimdTiling::IsCapable()
{
    if (!Ops::Base::IsRegbaseSocVersion(context_)) {
        return false;
    }
    if (isDeterministic_ != 1 || !isFp32Output_) {
        return false;
    }
    if (inputDtypeVal_ != 1 && inputDtypeVal_ != 7) {
        return false;
    }
    if (bins_ <= 0 || bins_ > SIMD_MAX_BINS) {
        return false;
    }
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize) - static_cast<int64_t>(SIMT_DCACHE_SIZE);
    if (ubSize <= 0) {
        return false;
    }
    int64_t ubNumCanUse = ubSize / SIZE_OF_FLOAT32;
    return bins_ < ubNumCanUse;
}

ge::graphStatus HistogramV2SimdTiling::GetShapeAttrsInfo()
{
    auto inputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    totalLength_ = inputShape->GetStorageShape().GetShapeSize();

    auto compileInfo = reinterpret_cast<const HistogramV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->totalCoreNum;
    if (coreNum_ <= 0) {
        auto platformInfo = context_->GetPlatformInfo();
        if (platformInfo != nullptr) {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            coreNum_ = ascendcPlatform.GetCoreNumAiv();
        }
    }
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_, "coreNum must be > 0, but got %ld.", coreNum_),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* binsPtr = attrs->GetAttrPointer<int64_t>(BINS_IDX);
    bins_ = (binsPtr == nullptr) ? DEFAULT_BINS : *binsPtr;
    OP_CHECK_IF(
        bins_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "bins", std::to_string(bins_).c_str(), "a positive integer"),
        return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputDataLength = outputShape->GetStorageShape().GetShapeSize();
    if (outputDataLength != bins_) {
        std::string sizeStr = std::to_string(outputDataLength) + " and " + std::to_string(bins_);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "y and bins", sizeStr.c_str(),
                                                   "The shape size of y should be the same as bins");
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto dType = inputDesc->GetDataType();

    for (int64_t i = 1; i < INPUT_NUM; i++) {
        auto inputMinMaxDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputMinMaxDesc);
        auto minMaxDtype = inputMinMaxDesc->GetDataType();
        if (minMaxDtype != dType) {
            std::string dtypesStr = Ops::Base::ToString(dType) + " and " + Ops::Base::ToString(minMaxDtype);
            std::string paramNames = std::string("x and ") + (i == 1 ? "min" : "max");
            std::string reason = std::string("The dtypes of ") + paramNames + " must be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramNames.c_str(), dtypesStr.c_str(),
                                                   reason.c_str());
            return ge::GRAPH_FAILED;
        }
        auto minMaxShape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, minMaxShape);
        auto minMaxLength = minMaxShape->GetStorageShape().GetShapeSize();
        if (minMaxLength != 1) {
            std::string paramName = (i == 1) ? "min" : "max";
            OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), paramName.c_str(),
                                          std::to_string(minMaxLength).c_str(), "1");
            return ge::GRAPH_FAILED;
        }
    }

    if (dType == ge::DT_FLOAT) {
        inputDtypeVal_ = 1;
    } else if (dType == ge::DT_FLOAT16) {
        inputDtypeVal_ = 7;
    } else {
        inputDtypeVal_ = 0;
    }

    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    isFp32Output_ = (outputDesc->GetDataType() == ge::DT_FLOAT);
    isDeterministic_ = (context_->GetDeterministic() == 1) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t HistogramV2SimdTiling::GetTilingKey() const { return context_->GetTilingKey(); }

ge::graphStatus HistogramV2SimdTiling::DoOpTiling()
{
    aicoreParams_.ubSize = aicoreParams_.ubSize - SIMT_DCACHE_SIZE;
    ubNumCanUse_ = static_cast<int64_t>(aicoreParams_.ubSize) / SIZE_OF_FLOAT32;
    ubLoopNum_ = Ops::Base::CeilDiv(bins_, ubNumCanUse_);
    context_->SetLocalMemorySize(aicoreParams_.ubSize);
    context_->SetTilingKey(TILING_KEY_SIMD_UB_FULL + OUTPUT_FP32_KEY_OFFSET + inputDtypeVal_);
    OP_CHECK_IF(TilingDataForCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "TilingDataForCore failed."),
                return ge::GRAPH_FAILED);
    return SetKernelTiling();
}

ge::graphStatus HistogramV2SimdTiling::TilingDataForCore()
{
    int64_t useCoreNum = coreNum_;
    int64_t wanted = Ops::Base::CeilDiv(totalLength_, SIMD_MIN_ELEMS_PER_CORE);
    useCoreNum = std::max<int64_t>(1, std::min(useCoreNum, wanted));

    formerLength_ = Ops::Base::CeilDiv(totalLength_, useCoreNum);
    formerLength_ = Ops::Base::CeilDiv(formerLength_, SIMD_SPLIT_ALIGN) * SIMD_SPLIT_ALIGN;
    needXCoreNum_ = Ops::Base::CeilDiv(totalLength_, formerLength_);
    tailLength_ = totalLength_ - (needXCoreNum_ - 1) * formerLength_;

    clearYFactor_ = Ops::Base::CeilDiv(bins_, coreNum_);
    OP_CHECK_IF(clearYFactor_ == 0, OP_LOGE(context_, "clearYFactor must not be 0."), return ge::GRAPH_FAILED);
    clearYCoreNum_ = Ops::Base::CeilDiv(bins_, clearYFactor_);
    clearYTail_ = bins_ - (clearYCoreNum_ - 1) * clearYFactor_;
    // Reduce writes y; blockDim only needs the x cores.
    needCoreNum_ = needXCoreNum_;

    const int64_t slotWidth = Ops::Base::CeilDiv(bins_, 32 / SIZE_OF_INT32) * (32 / SIZE_OF_INT32);
    uint64_t slotBytes = static_cast<uint64_t>(needXCoreNum_) * static_cast<uint64_t>(slotWidth) * sizeof(int32_t);
    workspaceBytes_ = std::max(SYSTEM_WORKSPACE, slotBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimdTiling::SetKernelTiling()
{
    context_->SetBlockDim(needCoreNum_);
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkSpace);
    currentWorkSpace[0] = workspaceBytes_;

    tilingData.set_bins(bins_);
    tilingData.set_ubNumCanUse(ubNumCanUse_);
    tilingData.set_ubLoopNum(ubLoopNum_);
    tilingData.set_needXCoreNum(needXCoreNum_);
    tilingData.set_formerLength(formerLength_);
    tilingData.set_tailLength(tailLength_);
    tilingData.set_clearYCoreNum(clearYCoreNum_);
    tilingData.set_clearYFactor(clearYFactor_);
    tilingData.set_clearYTail(clearYTail_);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimdTiling::GetWorkspaceSize()
{
    workspaceSize_ = workspaceBytes_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimdTiling::PostTiling()
{
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void HistogramV2SimdTiling::TilingDataPrint() const
{
    OP_LOGD(context_, "simd tilingKey: %lu.", context_->GetTilingKey());
    OP_LOGD(context_, "needCoreNum_: %ld.", needCoreNum_);
    OP_LOGD(context_, "totalLength_: %ld.", totalLength_);
    OP_LOGD(context_, "bins_: %ld.", bins_);
    OP_LOGD(context_, "needXCoreNum_: %ld.", needXCoreNum_);
    OP_LOGD(context_, "formerLength_: %ld.", formerLength_);
    OP_LOGD(context_, "tailLength_: %ld.", tailLength_);
}

// Tried before SIMT (30000) so capable cases land on dhistv2.
REGISTER_OPS_TILING_TEMPLATE(HistogramV2, HistogramV2SimdTiling, 20000);
} // namespace optiling
