/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include "register/op_impl_registry.h"
#include "conversion/matrix_set_diag/op_kernel/arch35/matrix_set_diag_tilingdata.h"
#include "conversion/matrix_set_diag/op_kernel/arch35/matrix_set_diag_tilingkey.h"
#include "platform/platform_ascendc.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "op_host/input_util.h"
#include "exe_graph/runtime/runtime_attrs.h"

namespace optiling {
// NCHW 常量
// BUFFER分割数量
static constexpr uint32_t BUFFER_NUM = 2;

static constexpr uint8_t MIN_INPUT_DIMNUM = 2;
static constexpr uint8_t MAX_INPUT_DIMNUM = 8;
// 用于tiling优化
static constexpr double MIN_USED_CORES_RATIO = 0.8;
static constexpr int64_t MIN_PER_UB_SIZE = 4096;
// UB内 scatter 操作的最大元素个数
static constexpr int32_t MAX_UB_SCATTER_ELEMENT_NUM = std::numeric_limits<uint16_t>::max();

// SIMT 常量
static constexpr int64_t MAX_SHAPE_SIZE_FOR_SIMT = 1024;

class MatrixSetDiagTiling {
private:
    /* data */
    // soc info
    uint64_t ubSize_{0};
    uint64_t ubBlockSize_{0};
    uint64_t bufferSize_{0};
    uint64_t vectorSize_{0};
    uint32_t coreNum_{0};
    uint32_t realCoreNum_{0};
    int32_t ubBlockElements_{0};
    int32_t cacheLineElements_{0};

    uint32_t dimNum_{1};
    uint32_t diagDimNum_{1};
    uint64_t ubFactor_{0};
    uint64_t ubPerCount_{0};
    uint64_t ubTotalCount_{0};
    uint64_t mergeDimSize_{1};
    uint64_t diagLen_{0};
    uint64_t xRowNum_{0};
    uint64_t xColNum_{0};
    uint64_t tailAxisDataSize_{0};
    uint64_t ubPerTail_{0};

    // tiling key param
    bool isSIMT_{false};
    bool isCutW_{false};

    // 输入参数
    int32_t dSize_{0};
    MatrixSetDiagTilingData* tilingData_;

    // tiling context
    gert::TilingContext* context_;

public:
    explicit MatrixSetDiagTiling(gert::TilingContext* context) : context_(context) {};
    ~MatrixSetDiagTiling();

    ge::graphStatus DoTiling();

private:
    // 参数检查，数据获取
    ge::graphStatus ParamCheck();
    ge::graphStatus GetSocInfo();

    // tiling 计算
    ge::graphStatus Tiling4MatrixSetDiag();
    ge::graphStatus Tiling4CutW();
    ge::graphStatus Tiling4NoCutW();

    // 辅助函数
    template <typename T>
    inline T AlignBlock(T elementCount);
    // NCHW
    void CalUbFactor();
    void GetOptimizeTiling();
    void GetOptimizeTilingAxis();
    uint64_t CalSizeTaken(uint64_t factor);

    // 打印
    void ShowTilingData();
    void FillsTilingData();
};

MatrixSetDiagTiling::~MatrixSetDiagTiling()
{}

ge::graphStatus MatrixSetDiagTiling::DoTiling()
{
    // 校验属性
    auto ret = ParamCheck();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling ParamCheck failed"), return ge::GRAPH_FAILED);

    // soc信息获取
    ret = GetSocInfo();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling GetSocInfo failed"), return ge::GRAPH_FAILED);

    ret = Tiling4MatrixSetDiag();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(isCutW_, isSIMT_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is %lu, isSIMT %d, isCutW %d", tilingKey, isSIMT_, isCutW_);
    tilingData_ = context_->GetTilingData<MatrixSetDiagTilingData>();
    FillsTilingData();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(realCoreNum_);
    size_t* workSpaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaceSize);
    workSpaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline T MatrixSetDiagTiling::AlignBlock(T elementCount)
{
    return Ops::Base::CeilAlign(elementCount, static_cast<T>(ubBlockElements_));
}

void MatrixSetDiagTiling::ShowTilingData()
{
    OP_LOGI(
        context_, "ubFactor %lu, ubPerCore %lu, ubTotalCount %lu, ubPerTail %lu, isCutW %d", ubFactor_, ubPerCount_,
        ubTotalCount_, ubPerTail_, isCutW_);
}

ge::graphStatus MatrixSetDiagTiling::ParamCheck()
{
    auto inputValueDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);

    auto inputDataType = inputValueDesc->GetDataType();
    dSize_ = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(dSize_ <= 0, OP_LOGE(context_, "data size should be positive"), return ge::GRAPH_FAILED);

    // 校验输入shape
    auto inputShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);

    auto inputShapeVal = inputShape->GetStorageShape();
    dimNum_ = inputShapeVal.GetDimNum();
    OP_CHECK_IF(
        dimNum_ < MIN_INPUT_DIMNUM || dimNum_ > MAX_INPUT_DIMNUM, OP_LOGE(context_, "input dim must be between [2,8]"),
        return ge::GRAPH_FAILED);

    auto diagValueDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagValueDesc);

    auto diagDataType = diagValueDesc->GetDataType();
    OP_CHECK_IF(
        inputDataType != diagDataType, OP_LOGE(context_, "input and diag should have same type"),
        return ge::GRAPH_FAILED);

    // 校验输入shape
    auto diagShape = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagShape);

    auto diagShapeVal = diagShape->GetStorageShape();
    diagDimNum_ = diagShapeVal.GetDimNum();
    OP_CHECK_IF(diagDimNum_ < 1, OP_LOGE(context_, "diag dim must >=1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dimNum_ != diagDimNum_ + 1, OP_LOGE(context_, "diag dim must equal input dim - 1"), return ge::GRAPH_FAILED);

    xColNum_ = inputShapeVal.GetDim(dimNum_ - 1);
    xRowNum_ = inputShapeVal.GetDim(dimNum_ - 2);
    tailAxisDataSize_ = xColNum_ * xRowNum_;
    diagLen_ = diagShapeVal.GetDim(diagDimNum_ - 1);
    OP_CHECK_IF(
        diagLen_ != std::min(xColNum_, xRowNum_), OP_LOGE(context_, "diagLen is invalid"), return ge::GRAPH_FAILED);
    if (diagDimNum_ > 1) {
        for (int32_t i = diagDimNum_ - 2; i >= 0; i--) {
            OP_CHECK_IF(
                diagShapeVal.GetDim(i) != inputShapeVal.GetDim(i), OP_LOGE(context_, "diagDim is invalid"),
                return ge::GRAPH_FAILED);
            mergeDimSize_ = mergeDimSize_ * static_cast<uint64_t>(diagShapeVal.GetDim(i));
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MatrixSetDiagTiling::CalUbFactor()
{
    uint64_t validBufSize = bufferSize_ - ubBlockSize_ * 2;
    if (isCutW_) {
        if (xColNum_ * dSize_ >= bufferSize_) {
            ubFactor_ = validBufSize / dSize_;
        } else {
            uint64_t diagStride = xColNum_ + 1;
            ubFactor_ = (validBufSize / dSize_ * diagStride) / (diagStride + 1);
            while (ubFactor_ > 0 &&
                   AlignBlock(ubFactor_) + AlignBlock(ubFactor_ / diagStride + 1) > bufferSize_ / dSize_) {
                ubFactor_ = ubFactor_ - 1;
            }
        }
    } else {
        uint64_t totalTailSize = (tailAxisDataSize_ + diagLen_) * dSize_;
        ubFactor_ = validBufSize >= totalTailSize ? validBufSize / totalTailSize : 1;
    }
}

void MatrixSetDiagTiling::FillsTilingData()
{
    tilingData_->coreNum = realCoreNum_;
    tilingData_->mergeDimSize = mergeDimSize_;
    tilingData_->xRowNum = xRowNum_;
    tilingData_->xColNum = xColNum_;
    tilingData_->diagLen = diagLen_;
    tilingData_->ubPerCore = ubPerCount_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->ubTotalCount = ubTotalCount_;
    tilingData_->ubPerTail = ubPerTail_;
    tilingData_->tailAxisDataSize = tailAxisDataSize_;
}

ge::graphStatus MatrixSetDiagTiling::Tiling4CutW()
{
    isCutW_ = true;
    CalUbFactor();
    OP_CHECK_IF((ubFactor_ == 0U), OP_LOGE(context_, "ubFactor is 0"), return ge::GRAPH_FAILED);
    if (dSize_ <= 2) {
        ubFactor_ = ubFactor_ < MAX_UB_SCATTER_ELEMENT_NUM ? ubFactor_ : MAX_UB_SCATTER_ELEMENT_NUM;
    }
    // 设置核数
    ubPerTail_ = Ops::Base::CeilDiv(tailAxisDataSize_, ubFactor_);
    ubFactor_ = Ops::Base::CeilDiv(tailAxisDataSize_, ubPerTail_);
    ubTotalCount_ = ubPerTail_ * mergeDimSize_;
    realCoreNum_ = ubTotalCount_ > coreNum_ ? coreNum_ : static_cast<uint32_t>(ubTotalCount_);
    ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));
    // 打印
    ShowTilingData();
    GetOptimizeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagTiling::Tiling4NoCutW()
{
    CalUbFactor();
    OP_CHECK_IF((ubFactor_ == 0U), OP_LOGE(context_, "ubFactor is 0"), return ge::GRAPH_FAILED);
    if (dSize_ <= 2) {
        ubFactor_ = ubFactor_ * tailAxisDataSize_ < MAX_UB_SCATTER_ELEMENT_NUM ?
                        ubFactor_ :
                        MAX_UB_SCATTER_ELEMENT_NUM / tailAxisDataSize_;
    }
    // 设置核数
    ubTotalCount_ = Ops::Base::CeilDiv(mergeDimSize_, ubFactor_);
    ubFactor_ = Ops::Base::CeilDiv(mergeDimSize_, ubTotalCount_);
    realCoreNum_ = ubTotalCount_ > coreNum_ ? coreNum_ : static_cast<uint32_t>(ubTotalCount_);
    ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));

    // 打印
    ShowTilingData();
    GetOptimizeTiling();
    return ge::GRAPH_SUCCESS;
}

void MatrixSetDiagTiling::GetOptimizeTiling()
{
    uint64_t curSize = CalSizeTaken(ubFactor_);
    uint64_t curFactor = ubFactor_;
    if (static_cast<double>(realCoreNum_) / static_cast<double>(coreNum_) >= MIN_USED_CORES_RATIO ||
        curSize <= MIN_PER_UB_SIZE) {
        return;
    }
    uint32_t startCoreNum = realCoreNum_ + (isCutW_ ? mergeDimSize_ : 1);
    for (uint32_t i = startCoreNum; i <= static_cast<uint32_t>(static_cast<double>(coreNum_) * MIN_USED_CORES_RATIO);) {
        curFactor =
            Ops::Base::CeilDiv(isCutW_ ? mergeDimSize_ * tailAxisDataSize_ : mergeDimSize_, static_cast<uint64_t>(i));
        if (curFactor == 1 && !isCutW_) {
            GetOptimizeTilingAxis();
            break;
        }
        if (curFactor != ubFactor_) {
            if (CalSizeTaken(curFactor) <= MIN_PER_UB_SIZE) {
                break;
            } else {
                ubFactor_ = curFactor;
                realCoreNum_ = i;
            }
        }
        i += isCutW_ ? mergeDimSize_ : 1;
    }
    if (isCutW_) {
        ubPerTail_ = Ops::Base::CeilDiv(tailAxisDataSize_, ubFactor_);
        ubTotalCount_ = ubPerTail_ * mergeDimSize_;
        ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));
    } else {
        ubTotalCount_ = Ops::Base::CeilDiv(mergeDimSize_, ubFactor_);
        ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, static_cast<uint64_t>(realCoreNum_));
    }
    ShowTilingData();
}

void MatrixSetDiagTiling::GetOptimizeTilingAxis()
{
    isCutW_ = true;
    uint64_t lastFactor_ = ubFactor_;
    ubFactor_ = tailAxisDataSize_;
    uint64_t curFactor = ubFactor_;
    for (uint32_t i = 1;
         i <= static_cast<uint32_t>(static_cast<double>(coreNum_) * MIN_USED_CORES_RATIO) / mergeDimSize_; i++) {
        curFactor = Ops::Base::CeilDiv(tailAxisDataSize_, static_cast<uint64_t>(i));
        if (curFactor != ubFactor_) {
            if (CalSizeTaken(curFactor) <= MIN_PER_UB_SIZE) {
                break;
            } else {
                ubFactor_ = curFactor;
                realCoreNum_ = i * mergeDimSize_;
            }
        }
    }
    if (ubFactor_ == tailAxisDataSize_) {
        isCutW_ = false;
        ubFactor_ = lastFactor_;
    }
}

uint64_t MatrixSetDiagTiling::CalSizeTaken(uint64_t factor)
{
    return (isCutW_ ? AlignBlock(factor) + AlignBlock(factor / (xColNum_ + 1) + 1) :
                      (AlignBlock(tailAxisDataSize_ * factor) + AlignBlock(diagLen_ * factor))) *
           dSize_;
}

ge::graphStatus MatrixSetDiagTiling::Tiling4MatrixSetDiag()
{
    ubBlockElements_ = ubBlockSize_ / dSize_;

    uint64_t shapeSize = mergeDimSize_ * tailAxisDataSize_;
    if (shapeSize <= MAX_SHAPE_SIZE_FOR_SIMT) {
        isSIMT_ = true;
        realCoreNum_ = realCoreNum_ < shapeSize ? realCoreNum_ : shapeSize;
        return ge::GRAPH_SUCCESS;
    }
    uint64_t totalTailSize = (AlignBlock(tailAxisDataSize_) + AlignBlock(diagLen_)) * dSize_;
    bufferSize_ = ubSize_ / BUFFER_NUM - vectorSize_;
    OP_LOGI(context_, "bufferSize_ %lu, totalTailSize %lu", bufferSize_, totalTailSize);
    if (totalTailSize >= bufferSize_ || (dSize_ <= 2 && tailAxisDataSize_ >= MAX_UB_SCATTER_ELEMENT_NUM)) {
        return Tiling4CutW();
    } else {
        return Tiling4NoCutW();
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus MatrixSetDiagTiling::GetSocInfo()
{
    // 获取soc信息, 如ub大小, core数等
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    realCoreNum_ = coreNum_;
    OP_CHECK_IF((coreNum_ == 0U), OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ == 0U), OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF((ubBlockSize_ == 0U), OP_LOGE(context_, "Failed to get ub block size."), return ge::GRAPH_FAILED);
    vectorSize_ = static_cast<uint64_t>(Ops::Base::GetVRegSize(context_));
    OP_CHECK_IF(vectorSize_ == 0U, OP_LOGE(context_, "Failed to vector size."), return ge::GRAPH_FAILED);
    OP_LOGI(context_, "soc info: ubSize %lu, coreNum %u, ubBlockSize %lu ", ubSize_, coreNum_, ubBlockSize_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4MatrixSetDiag(gert::TilingContext* context)
{
    // DoTiling
    MatrixSetDiagTiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMatrixSetDiag([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatrixSetDiag)
    .Tiling(Tiling4MatrixSetDiag)
    .TilingParse<MatrixSetDiagCompileInfo>(TilingPrepareForMatrixSetDiag);
} // namespace optiling
