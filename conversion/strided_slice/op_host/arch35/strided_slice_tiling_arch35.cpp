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
 * \file strided_slice_tiling.cc
 * \brief
 */
#include "strided_slice_tiling_arch35.h"
#include "common/inc/op_host/op_const_def_math.h"

namespace optiling {
static const std::string OP_NAME = "StridedSlice";
constexpr int64_t MAX_GATHER_STRIDE = 16;
constexpr int64_t MOVE_ALIGN_CACHE_LINE_FACTOR = 4;
constexpr int64_t NDDMA_CACHE_LINE_FACTOR = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UNCONST_BEGIN_VALUE = -10; // unconst场景下，用-10标识需要从kernel获取
constexpr int64_t GATHER_MAX_IDX = 65536;
constexpr int64_t SIMT_PACK_BYTES_16 = 16;
constexpr int64_t SIMT_PACK_BYTES_8 = 8;
constexpr int64_t SIMT_PACK_BYTES_4 = 4;
constexpr int64_t SIMT_PACK_BYTES_2 = 2;
// SIMT 线程数常量 (逐级折半: 2048→1024→512→256)
constexpr int64_t SIMT_THREAD_MAX = 2048;
constexpr int64_t SIMT_THREAD_HALF = 1024;
constexpr int64_t SIMT_THREAD_QUARTER = 512;
constexpr int64_t SIMT_THREAD_MIN = 256;
// SIMT正常场景维度阈值: dim1-7用全量线程, dim8用一半
constexpr int32_t SIMT_FULL_THREAD_MAX_DIM = 7;
// BigShape场景线程分配维度阈值
constexpr int32_t BIGSHAPE_DIM_FULL = 1; // dim==1 用 SIMT_THREAD_MAX
constexpr int32_t BIGSHAPE_DIM_HALF = 2; // dim==2 用 SIMT_THREAD_HALF
// BigShape场景: dim<=4用512线程, 超过用256
constexpr int32_t BIGSHAPE_HALF_THREAD_MAX_DIM = 4;
// SIMT key基础值 + 按合并字节数偏移
constexpr int64_t SIMT_KEY_OFFSET_2B = 1;
constexpr int64_t SIMT_KEY_OFFSET_4B = 2;
constexpr int64_t SIMT_KEY_OFFSET_8B = 3;
constexpr int64_t SIMT_KEY_OFFSET_16B = 4;

static const int INDEX_X = 0;
static const int INDEX_BEGIN = 1;
static const int INDEX_END = 2;
static const int INDEX_STRIDES = 3;
static const int INDEX_Y = 0;
static const size_t IDX_MASK_BEGIN = 0;
static const size_t IDX_MASK_END = 1;
static const size_t IDX_MASK_ELLIPSIS = 2;
static const size_t IDX_MASK_NEW_AXIS = 3;
static const size_t IDX_MASK_SHRINK_AXIS = 4;

ge::graphStatus StrideSliceTiling::Init(int64_t coreNum, int64_t ubSize, int64_t cacheLineSize,
                                        SliceParametersRuntime2& sliceParam, const ge::DataType& dtype)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init StrideSliceTiling.");
    coreNum_ = coreNum;
    ubSize_ = ubSize;
    cacheLineSize_ = cacheLineSize;
    if (CheckPlatformParam() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    sliceParam_ = sliceParam;
    dtype_ = dtype;
    xDtypeSize_ = GetSizeByDataType(dtype);
    ubElementNum_ = (ubSize_ - UB_RESERVE_SIZE) / xDtypeSize_ / DOUBLE_BUFFER; // 负Stride会在ub切分时重新赋值
    dimNum_ = sliceParam.inputShape.GetDimNum();
    if (dimNum_ > MAX_AXIS_NUM_FOR_STRIDESLICE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "input", std::to_string(dimNum_),
                                                 "The shape dim of input must be <= 8.");
        return ge::GRAPH_FAILED;
    }

    const auto outputShape = sliceParam.outputShape;
    const auto inputShape = sliceParam.inputShape;
    const auto strideList = sliceParam.strideList;
    int32_t shapeSize = dimNum_;
    lastOneOutputDim_ = outputShape.GetDim(shapeSize - 1);
    lastTwoOutputDim_ = shapeSize >= NUMBER_TWO ? outputShape.GetDim(shapeSize - NUMBER_TWO) : 0;
    lastThreeOutputDim_ = shapeSize >= NUMBER_THREE ? outputShape.GetDim(shapeSize - NUMBER_THREE) : 0;
    lastOneInputDim_ = inputShape.GetDim(shapeSize - 1);
    lastTwoInputDim_ = shapeSize >= NUMBER_TWO ? inputShape.GetDim(shapeSize - NUMBER_TWO) : 0;
    lastThreeInputDim_ = shapeSize >= NUMBER_THREE ? inputShape.GetDim(shapeSize - NUMBER_THREE) : 0;
    lastOneStride_ = strideList[shapeSize - 1];
    lastTwoStride_ = shapeSize >= NUMBER_TWO ? strideList[shapeSize - NUMBER_TWO] : 0;
    lastThreeStride_ = shapeSize >= NUMBER_THREE ? strideList[shapeSize - NUMBER_THREE] : 0;
    lastFourStride_ = shapeSize >= NUMBER_FOUR ? strideList[shapeSize - NUMBER_FOUR] : 0;

    totalOutputSize_ = xDtypeSize_;
    for (int32_t i = 0; i < dimNum_; i++) {
        if (outputShape.GetDim(i) <= 0 || inputShape.GetDim(i) <= 0) {
            isEmptyTensor_ = true;
            return ge::GRAPH_SUCCESS;
        }
        totalOutputSize_ = totalOutputSize_ * outputShape.GetDim(i);
        if (strideList[i] < 0) {
            isStrideNeg_ = true;
        }
    }
    CalInputOutputSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StrideSliceTiling::CheckPlatformParam()
{
    if (coreNum_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(tilingContext_->GetNodeName(), "core number", std::to_string(coreNum_),
                                               "The value of core number cannot <= 0.");
        return ge::GRAPH_FAILED;
    }
    if (ubSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(tilingContext_->GetNodeName(), "ubSize", std::to_string(ubSize_),
                                               "The value of ubSize cannot <= 0.");
        return ge::GRAPH_FAILED;
    }
    if (cacheLineSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(tilingContext_->GetNodeName(), "cacheLineSize",
                                               std::to_string(cacheLineSize_),
                                               "The value of cacheLineSize cannot <= 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void StrideSliceTiling::CalInputOutputSize()
{
    int64_t outputProd = 1;
    int64_t inputProd = 1;
    int64_t shapeSize = std::min((int64_t)dimNum_, MAX_SIMT_UB_SPLIT_AXIS_NUM);
    const auto outputShape = sliceParam_.outputShape;
    const auto inputShape = sliceParam_.inputShape;
    for (int64_t i = shapeSize - 1; i >= 0; i--) {
        outputProd *= outputShape.GetDim(i);
        outputShapeProd_[i] = outputProd;
        inputProd *= inputShape.GetDim(i);
        inputShapeProd_[i] = inputProd;
    }
    if (outputShapeProd_[0] > MAX_UINT32_NUM || inputShapeProd_[0] > MAX_UINT32_NUM) {
        isShapeExceedUint32_ = true;
    }
    if (outputShapeProd_[0] <= SIMT_MIN_OUTPUT_SIZE) {
        isShapeBelowThreshold_ = true;
    }
}

void StrideSliceTiling::SetAllInUbSplitInfo()
{
    const auto& outputShape = sliceParam_.outputShape;
    isAllInUb_ = true;
    blkIndex_ = 0;
    ubIndex_ = 0;
    blkFactor_ = outputShape.GetDim(0);
    ubFactor_ = blkFactor_;
    realCoreNum_ = 1;
    ubTailFactor_ = 0;
    ubTailTailFactor_ = 0;
    blkTailFactor_ = 0;
}

void StrideSliceTiling::SetTwoDimTilingInfo()
{
    // 计算分核 整尾块切分  moveAlignv2参数
    if (lastOneInputDim_ * xDtypeSize_ < VL_SIZE && sliceParam_.isBeginConst == 1 &&
        sliceParam_.beginList.GetDim(1) == 0) {
        outBlockLen_ = static_cast<uint32_t>(lastOneInputDim_ * xDtypeSize_); // 全量搬入
    } else {
        outBlockLen_ = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
    }

    uint32_t perCoreRowNum = ubSize_ / NUMBER_FOUR / Ops::Base::CeilAlign(outBlockLen_, (uint32_t)BLOCK_SIZE);
    uint32_t coreNumByInput = Ops::Base::CeilDiv((uint32_t)lastTwoInputDim_, perCoreRowNum);
    uint32_t coreNumByOutput = Ops::Base::CeilDiv((uint32_t)totalOutputSize_, (uint32_t)TWO_DIM_MIN_OUTPUT_SIZE);
    uint32_t calCoreNum = coreNumByInput > coreNumByOutput ? coreNumByInput : coreNumByOutput;
    realCoreNum_ = static_cast<int64_t>(calCoreNum) >= coreNum_ ? coreNum_ : static_cast<int64_t>(calCoreNum);

    blkTailFactor_ = lastTwoInputDim_ / realCoreNum_;
    mainCoreNum_ = static_cast<uint32_t>(lastTwoInputDim_ % realCoreNum_);
    if (mainCoreNum_ != 0U) {
        blkFactor_ = blkTailFactor_ + 1;
    } else {
        blkFactor_ = blkTailFactor_;
        mainCoreNum_ = static_cast<uint32_t>(realCoreNum_);
    }
}

void StrideSliceTiling::CalcBlockSplitInfo()
{
    const auto& outputShape = sliceParam_.outputShape;
    int64_t minCoreProcessSize = 0;
    if (dimNum_ >= NUMBER_SIX && maxSplitDim_ == MAX_NDDMA_UB_SPLIT_AXIS_NUM) {
        int64_t lastFourDimProd = 1L;
        for (int32_t i = 0; i < NUMBER_FOUR; i++) {
            lastFourDimProd *= outputShape.GetDim(dimNum_ - 1 - i);
        }
        // 每个核最少处理4个轴的数据
        minCoreProcessSize = std::min(lastFourDimProd * xDtypeSize_, LAST_DIM_MIN_DATA_SIZE);
    } else {
        // 每个核最少处理1KB的数据，小shape场景下减少开核个数
        minCoreProcessSize = LAST_DIM_MIN_DATA_SIZE;
    }
    int64_t rightDiv = std::min(coreNum_, Ops::Base::CeilDiv(totalOutputSize_, minCoreProcessSize));
    if (dimNum_ == NUMBER_TWO && ((lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1) * xDtypeSize_ <=
                                     LAST_DIM_MIN_DATA_SIZE * NUMBER_FOUR) {
        rightDiv = std::min(rightDiv, outputShape.GetDim(0));
    }
    for (int32_t i = 0; i < dimNum_; i++) {
        int64_t curDim = outputShape.GetDim(i);
        if (rightDiv >= 1 && rightDiv / curDim <= 1) {
            blkIndex_ = i;
            blkFactor_ = Ops::Base::CeilDiv(curDim, rightDiv);
            break;
        }
        rightDiv = rightDiv / curDim;
    }

    // 最后一维留够1KB
    if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM && blkIndex_ == static_cast<int64_t>(dimNum_ - 1) &&
        blkFactor_ * std::abs(lastOneStride_) < LAST_DIM_MIN_DATA_SIZE / xDtypeSize_) {
        blkFactor_ = std::min(LAST_DIM_MIN_DATA_SIZE / xDtypeSize_ / std::abs(lastOneStride_),
                              outputShape.GetDim(blkIndex_));
    }

    // 说明整个shape都很小，blk不切最后一根轴
    if (blkIndex_ == static_cast<int64_t>(-1)) {
        blkIndex_ = static_cast<int64_t>(dimNum_ - 1);
        blkFactor_ = outputShape.GetDim(static_cast<size_t>(dimNum_ - 1));
    }

    blkTailFactor_ = outputShape.GetDim(blkIndex_) % blkFactor_;
    realCoreNum_ = Ops::Base::CeilDiv(outputShape.GetDim(blkIndex_), blkFactor_);
    for (int32_t i = 0; i < blkIndex_; i++) {
        realCoreNum_ *= outputShape.GetDim(i);
    }
}

void StrideSliceTiling::CalcUbSplitInfo()
{
    if (isSliceGather_) { // lastInputDim < 128B  lastOutputDim < 64B
        uint64_t calOutputSize = Ops::Base::CeilAlign(lastTwoOutputDim_ * lastOneOutputDim_ * xDtypeSize_, BLOCK_SIZE);
        uint64_t calInputSize = Ops::Base::CeilAlign(lastTwoOutputDim_ * lastOneInputDim_ * xDtypeSize_, BLOCK_SIZE);
        uint64_t halfUbSize = static_cast<uint64_t>((ubSize_ - INIT_INDEX_SIZE - cacheLineSize_) / DOUBLE_BUFFER);
        if (calOutputSize + calInputSize < halfUbSize) {
            // ub切分轴至少在-3轴
            ubSizeInput_ = static_cast<int64_t>(halfUbSize * calInputSize / (calOutputSize + calInputSize) /
                                                BLOCK_SIZE * BLOCK_SIZE);
            ubSizeOutput_ = static_cast<int64_t>(
                halfUbSize * calOutputSize / (calOutputSize + calInputSize) / BLOCK_SIZE * BLOCK_SIZE + cacheLineSize_);
        } else {
            // ub切分轴在-2轴
            ubSizeInput_ = static_cast<int64_t>(halfUbSize * lastOneInputDim_ / (lastOneInputDim_ + lastOneOutputDim_) /
                                                BLOCK_SIZE * BLOCK_SIZE);
            ubSizeOutput_ = static_cast<int64_t>(halfUbSize * lastOneOutputDim_ /
                                                     (lastOneInputDim_ + lastOneOutputDim_) / BLOCK_SIZE * BLOCK_SIZE +
                                                 cacheLineSize_);
        }
        ubElementNum_ = ubSizeInput_ / xDtypeSize_;
    }

    int64_t rightProduct = 1;
    const auto outputShape = sliceParam_.outputShape;
    const auto inputShape = sliceParam_.inputShape;
    int32_t shapeSize = outputShape.GetDimNum();
    int32_t maxLeftIdx = std::max(shapeSize - maxSplitDim_, static_cast<int32_t>(blkIndex_));
    for (int32_t i = shapeSize - 1; i >= 0 && i >= maxLeftIdx; i--) {
        int64_t curDim = outputShape.GetDim(i);
        if (rightProduct * curDim >= ubElementNum_) {
            ubIndex_ = i;
            ubFactor_ = ubElementNum_ / rightProduct;
            ubTailFactor_ = curDim % ubFactor_;
            break;
        }
        if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM && i == shapeSize - NUMBER_TWO) {
            rightProduct = Ops::Base::CeilAlign(curDim * rightProduct * xDtypeSize_, BLOCK_SIZE) / xDtypeSize_;
        } else if (isSliceGather_ && i == shapeSize - 1) {
            rightProduct = inputShape.GetDim(i);
        } else if (isSliceGather_ &&
                   i == shapeSize -
                            NUMBER_TWO) { // 最后2维连续搬入，倒数2根轴可以合并，否则就是最后一维数据需要block对齐
            rightProduct = Ops::Base::CeilAlign(curDim * rightProduct * xDtypeSize_, BLOCK_SIZE) / xDtypeSize_;
        } else {
            rightProduct = curDim * rightProduct;
        }
    }

    if (ubIndex_ == -1L) {
        ubIndex_ = maxLeftIdx;
        ubFactor_ = outputShape.GetDim(ubIndex_);
        ubTailFactor_ = 0;
    }

    // ub blk split same axis
    if (ubIndex_ == blkIndex_) {
        ubFactor_ = std::min(ubFactor_, blkFactor_);
        ubTailFactor_ = blkFactor_ % ubFactor_;
        ubTailTailFactor_ = blkTailFactor_ % ubFactor_;
    }
}

void StrideSliceTiling::CalcUbSplitInfoNeg()
{
    int32_t ubInFactor = isNddma_ ? 1 : std::abs(lastOneStride_);
    ubInFactor = std::max(ubInFactor, 1);
    // 开启db： 2*UbIn + 2*UbOut = 2*UbOut*ubInFactor + 2*UbOut = ubSize_
    int32_t ubOutBase = static_cast<int32_t>((ubSize_ - UB_RESERVE_SIZE) / DOUBLE_BUFFER / (ubInFactor + 1) /
                                             cacheLineSize_ * cacheLineSize_);
    ubOutBase = std::max(ubOutBase, static_cast<int32_t>(cacheLineSize_));
    ubSizeInput_ = static_cast<int64_t>(ubOutBase * ubInFactor);
    // B8/B16使用uint16作为gather的索引，这里限制一下，保证不超过uint16的最大值。限制输入大小不超过64KB
    if (ubSizeInput_ > GATHER_MAX_IDX) {
        ubOutBase = GATHER_MAX_IDX / ubInFactor / cacheLineSize_ * cacheLineSize_;
        ubOutBase = std::max(ubOutBase, static_cast<int32_t>(cacheLineSize_));
        ubSizeInput_ = static_cast<int64_t>(ubOutBase * ubInFactor);
    }
    ubElementNum_ = ubOutBase / xDtypeSize_;
    ubSizeOutput_ = ubOutBase;

    // moveAlign和nddma、gather/ub2ub，都要求最后一维ub内按block对齐
    int64_t rightProduct = 1;
    const auto outputShape = sliceParam_.outputShape;
    int32_t shapeSize = outputShape.GetDimNum();
    int32_t maxLeftIdx = std::max(shapeSize - maxSplitDim_, static_cast<int32_t>(blkIndex_));
    for (int32_t i = shapeSize - 1; i >= 0 && i >= maxLeftIdx; i--) {
        int64_t curDim = outputShape.GetDim(i);
        if (i == shapeSize - 1) {
            // 这里按ubOut作对齐，则ubIn必然也是对齐的
            curDim = Ops::Base::CeilAlign(curDim, BLOCK_SIZE / xDtypeSize_);
        }
        if (rightProduct * curDim >= ubElementNum_) {
            ubIndex_ = i;
            ubFactor_ = ubElementNum_ / rightProduct;
            ubTailFactor_ = outputShape.GetDim(i) % ubFactor_;
            break;
        }
        rightProduct = curDim * rightProduct;
    }

    if (ubIndex_ == -1L) {
        ubIndex_ = maxLeftIdx;
        ubFactor_ = outputShape.GetDim(ubIndex_);
        ubTailFactor_ = 0;
    }

    // ub blk split same axis
    if (ubIndex_ == blkIndex_) {
        ubFactor_ = std::min(ubFactor_, blkFactor_);
        ubTailFactor_ = blkFactor_ % ubFactor_;
        ubTailTailFactor_ = blkTailFactor_ % ubFactor_;
    }
}

ge::graphStatus StrideSliceTiling::RunStrideSliceTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start running Tiling4StrideSlice.");
    // WARNING: 优先判断 isEmptyTensor_，else分支内不需要除0保护
    if (isEmptyTensor_) {
        OP_LOGI(tilingContext_->GetNodeName(), "Input or Output tensor is empty, set SIMT mode.");
        SetSIMTTilingMode();
    } else if (isStrideNeg_) {
        CalMaxSplitDimNeg();
        if (maxSplitDim_ != MAX_SIMT_UB_SPLIT_AXIS_NUM) {
            CalcBlockSplitInfo();
            CalcUbSplitInfoNeg();
        }
        SetTilingModeNeg();
    } else {
        CalMaxSplitDim();
        if (maxSplitDim_ == MAX_TWO_DIM_UB_SPLIT_AXIS_NUM) {
            SetTwoDimTilingInfo();
        } else if (totalOutputSize_ <= LAST_DIM_MIN_DATA_SIZE && dimNum_ <= maxSplitDim_) {
            SetAllInUbSplitInfo();
        } else {
            CalcBlockSplitInfo();
            if (!useGather_) {
                CalcUbSplitInfo();
            } else {
                CalcUbSplitInfoNeg();
            }
        }
        SetTilingMode();
    }

    // fill data
    FillTilingData();
    // print data
    PrintTilingData();
    // set block dim and tilingKey
    SetBlockDimAndTilingKey();

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    auto platformInfo = tilingContext_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGD(tilingContext_->GetNodeName(), "Tiling4StrideSlice success.");
    return ge::GRAPH_SUCCESS;
}

int64_t StrideSliceTiling::GetDimStartPosition(int32_t idx)
{
    if (!sliceParam_.isBeginConst) {
        if (sliceParam_.beginList[idx] >= 0) {
            return (sliceParam_.strideList[idx] > 0) ?
                       sliceParam_.beginList[idx] :
                       (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
        }

        return 0;
    }

    return (sliceParam_.strideList[idx] > 0) ?
               sliceParam_.beginList[idx] :
               (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
}

int64_t StrideSliceTiling::GetDimEndPosition(int32_t idx)
{
    if (!sliceParam_.isBeginConst) {
        if (sliceParam_.beginList[idx] >= 0) {
            return (sliceParam_.strideList[idx] < 0) ?
                       sliceParam_.beginList[idx] :
                       (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
        }

        return 1;
    }

    return (sliceParam_.strideList[idx] < 0) ?
               sliceParam_.beginList[idx] :
               (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
}

int64_t StrideSliceTiling::GetValidNumInCacheLine()
{
    int64_t cachelineNum = cacheLineSize_ / xDtypeSize_; // cacheline内的数据个数
    int64_t validNum = 1;                                // 起始位置肯定有效，计数为1

    if (lastOneInputDim_ >= cachelineNum) {
        int64_t checkNum = std::min((lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1, cachelineNum);
        validNum = Ops::Base::CeilDiv(checkNum, std::abs(lastOneStride_));
        OP_LOGI(tilingContext_->GetNodeName(), "cachelineNum:%ld validNum:%ld", cachelineNum, validNum);
        return validNum;
    }

    int64_t startOffsetNum = 0;
    int64_t inputShapeStride = 1;
    for (int32_t i = dimNum_ - 1; i >= 0; i--) {
        int64_t startV = GetDimStartPosition(i);
        startOffsetNum += startV * inputShapeStride;
        inputShapeStride *= sliceParam_.inputShape[i];
    }
    OP_LOGD(tilingContext_->GetNodeName(), "startOffsetNum:%ld inputShapeProd_[0]:%ld", startOffsetNum,
            inputShapeProd_[0]);

    int64_t totalInputCnt = std::min(inputShapeProd_[0], startOffsetNum + cachelineNum);
    for (int64_t idx = startOffsetNum + 1; idx < totalInputCnt; idx++) {
        bool curValid = true;
        int64_t shapeIdx = idx;
        for (int32_t i = dimNum_ - 1; i >= 0; i--) {
            int64_t curStart = GetDimStartPosition(i);
            int64_t curEnd = GetDimEndPosition(i);
            int64_t curDimIdx = shapeIdx % sliceParam_.inputShape[i];
            shapeIdx = shapeIdx / sliceParam_.inputShape[i];
            if (curDimIdx < curStart || curDimIdx > curEnd ||
                (curDimIdx - curStart) % std::abs(sliceParam_.strideList[i]) != 0) {
                curValid = false;
                break;
            }
        }
        if (curValid) {
            validNum++;
        }
    }

    OP_LOGI(tilingContext_->GetNodeName(), "cachelineNum:%ld validNum:%ld", cachelineNum, validNum);
    return validNum;
}

void StrideSliceTiling::CalMaxSplitDimLastStrideNegative()
{
    int64_t lastDimInLen = (lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1;
    // MoveAlignV2 + gather
    if (std::abs(lastOneStride_) <= MAX_GATHER_STRIDE && lastDimInLen * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
        useGather_ = true;
        return;
    }

    // nddma + gather
    int64_t validNum = GetValidNumInCacheLine();
    if (validNum * xDtypeSize_ >= cacheLineSize_ / NDDMA_CACHE_LINE_FACTOR &&
        lastOneOutputDim_ * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG;
        useGather_ = true;
        isNddma_ = true;
        return;
    }
    // default simt
    return;
}

void StrideSliceTiling::CalMaxSplitDimLastStrideOne()
{
    int64_t lastDimInLen = (lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1;
    if (lastDimInLen * xDtypeSize_ >= cacheLineSize_ / MOVE_ALIGN_CACHE_LINE_FACTOR) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
        useGather_ = false;
        return;
    }
    // default simt
    return;
}

void StrideSliceTiling::CalMaxSplitDimLastStridePositive()
{
    int64_t lastDimInLen = (lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1;
    if (lastOneStride_ <= MAX_GATHER_STRIDE && lastDimInLen * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
        useGather_ = true;
        return;
    }

    int64_t validNum = GetValidNumInCacheLine();
    if (validNum * xDtypeSize_ >= cacheLineSize_ / NDDMA_CACHE_LINE_FACTOR &&
        lastOneOutputDim_ * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG;
        useGather_ = false;
        isNddma_ = true;
        return;
    }
    // default simt
    return;
}

void StrideSliceTiling::CalMaxSplitDimNeg()
{
    if (totalOutputSize_ <= LAST_DIM_MIN_DATA_SIZE) {
        return;
    }

    if (lastOneStride_ < 0) {
        return CalMaxSplitDimLastStrideNegative();
    }

    if (lastOneStride_ == 1) {
        return CalMaxSplitDimLastStrideOne();
    }

    return CalMaxSplitDimLastStridePositive();
}

void StrideSliceTiling::CalMaxSplitDim()
{
    int64_t validNum = GetValidNumInCacheLine();
    if (lastOneStride_ == 1) {
        if (lastOneOutputDim_ * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
            maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
            return;
        }

        if (validNum * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
            maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
            return;
        }
    } else {
        // lastOneStride_ > 1
        int64_t lastDimInLen = (lastOneOutputDim_ - 1) * lastOneStride_ + 1;
        if (lastOneStride_ <= MAX_GATHER_STRIDE && lastDimInLen * xDtypeSize_ >= cacheLineSize_ &&
            totalOutputSize_ > LAST_DIM_MIN_DATA_SIZE) {
            maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
            useGather_ = true;
            return;
        }

        if (validNum * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
            maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
            return;
        }
    }

    if (isShapeExceedUint32_ || isShapeBelowThreshold_) {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
        return;
    }
}

void StrideSliceTiling::SetTilingMode()
{
    if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM) {
        SetMovAlignV2TilingMode();
    } else if (maxSplitDim_ == MAX_NDDMA_UB_SPLIT_AXIS_NUM) {
        SetNddmaTilingMode();
    } else {
        SetSIMTTilingMode();
    }
}

void StrideSliceTiling::SetTilingModeNeg()
{
    if (maxSplitDim_ == MAX_SIMT_UB_SPLIT_AXIS_NUM) {
        SetSIMTTilingMode();
    } else {
        if (!isNddma_) {
            SetMovAlignV2TilingModeNeg();
        } else {
            SetNddmaTilingModeNeg();
        }
    }
}

void StrideSliceTiling::SetMovAlignV2TilingModeNeg()
{
    tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB;

    // ubfactor切-1轴
    if (ubIndex_ == dimNum_ - 1L) {
        MovAlignV2UbSplitLastOneDim();
        return;
    }

    // ubfactor切-2轴
    if (ubIndex_ == dimNum_ - NUMBER_TWO) {
        return MovAlignV2UbSplitLastTwoDim();
    }

    // ubfactor切-3轴
    if (ubIndex_ == dimNum_ - NUMBER_THREE) {
        return MovAlignV2UbSplitLastThreeDim();
    }

    // ubfactor切-4轴
    if (ubIndex_ == dimNum_ - NUMBER_FOUR) {
        return MovAlignV2UbSplitLastFourDim();
    }
}

void StrideSliceTiling::SetNddmaTilingModeNeg()
{
    tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_NDDMA_GATHER : STRIDED_SLICE_KEY_NDDMA_UB2UB;

    // ubfactor切倒数第i轴
    int ubSplitDimNumber = static_cast<int>(dimNum_ - ubIndex_);
    for (int32_t j = 1; j <= ubSplitDimNumber; j++) {
        if (ubSplitDimNumber == j) {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = ubFactor_;
        } else {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = sliceParam_.outputShape.GetDim(dimNum_ - j);
        }
        nddmaTotalNum_ = nddmaTotalNum_ * nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j];
        nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = std::abs(sliceParam_.strideList[dimNum_ - j]);
        nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = 1;
        for (int32_t k = 1; k < j; k++) {
            nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] *= sliceParam_.inputShape.GetDim(dimNum_ - k);
            // 尾轴对齐到block
            if (k == 1) {
                nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] *= Ops::Base::CeilAlign(
                    sliceParam_.outputShape.GetDim(dimNum_ - k), BLOCK_SIZE / xDtypeSize_);
            } else {
                nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] *= sliceParam_.outputShape.GetDim(dimNum_ - k);
            }
        }
    }

    return;
}

/*
input
(S, A, B, C, D)
outputShape:
(s, a, b, c, d)

模板2, NDDMA场景
loop0Size
loop0srcStride
loop0DstStride
单位都是element number
loop0size = loopSizeList[NUMBER_FOUR]
loop0srcStirde = loopsrcStirdeList[NUMBER_FOUR]
loop0dstStride = loopdstStrideList[NUMBER_FOUR]

场景1: ub切d
(a, b, c, d.o, d.i)
loop0Size = d.i
loop0SrcStride = stride_d
loop0dstStride = 1

loop0TailSize = d % d.i

场景2:
ub切c
(a, b, c.o, c.i, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c.i
loop1SrcStride = D * stride_c
loop1dstStride = d

loop1TailSize = c % c.i

场景3:
ub切b
(a, b.o, b.i, c, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c
loop1SrcStride = D * stride_c
loop1dstStride = d

loop2Size = b.i
loop2SrcStride = D * C * stride_b
loop2dstStride = d * c

loop2TailSize = b % b.i

场景4:
ub切a
(a.o, a.i, b, c, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c
loop1SrcStride = D * stride_c
loop1dstStride = d

loop2Size = b
loop2SrcStride = D * C * stride_b
loop2dstStride = d * c

loop3Size = a.i
loop3SrcStride = D * C * B * stride_a
loop3dstStride = d * c * b

loop3TailSize = a % a.i

场景5:
ub切s
(s.o, s.i, a, b, c, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c
loop1SrcStride = D * stride_c
loop1dstStride = d

loop2Size = b
loop2SrcStride = D * C * stride_b
loop2dstStride = d * c

loop3Size = a
loop3SrcStride = D * C * B * stride_a
loop3dstStride = d * c * b

loop4Size = s.i
loop4SrcStride = D * C * B * A * stride_s
loop4dstStride = d * c * b * a

loop4TailSize = s % s.i
*/
void StrideSliceTiling::SetNddmaTilingMode()
{
    int32_t shapeSize = dimNum_;
    tilingKey_ = STRIDED_SLICE_KEY_NDDMA;
    // ubfactor切倒数第i轴
    int ubSplitDimNumber = static_cast<int>(shapeSize - ubIndex_);
    if (ubSplitDimNumber == 1) {
        tilingKey_ = STRIDED_SLICE_KEY_NDDMA_LAST_DIM;
    }
    for (int32_t j = 1; j <= ubSplitDimNumber; j++) {
        if (ubSplitDimNumber == j) {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = ubFactor_;
        } else {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = sliceParam_.outputShape.GetDim(dimNum_ - j);
        }
        nddmaTotalNum_ = nddmaTotalNum_ * nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j];
        nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = sliceParam_.strideList[dimNum_ - j];
        nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = 1;
        for (int32_t k = 1; k < j; k++) {
            nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] *= sliceParam_.inputShape.GetDim(dimNum_ - k);
            nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] *= sliceParam_.outputShape.GetDim(dimNum_ - k);
        }
    }
}

// 线程数可选值: SIMT_THREAD_MAX→HALF→QUARTER→MIN，每级折半
static int64_t PickThread(int64_t maxThread, int64_t blockSize)
{
    const int64_t candidates[] = {SIMT_THREAD_MAX, SIMT_THREAD_HALF, SIMT_THREAD_QUARTER, SIMT_THREAD_MIN};
    for (auto t : candidates) {
        if (t <= maxThread && t <= blockSize) {
            return t;
        }
    }
    return SIMT_THREAD_MIN;
}

// 根据 dim 数和是否 BigShape 返回最大线程数
//   SIMT:         dim1-7→SIMT_THREAD_MAX  dim8→SIMT_THREAD_HALF
//   BigShape:     dim1→SIMT_THREAD_MAX  dim2→SIMT_THREAD_HALF
//                 dim3-4→SIMT_THREAD_QUARTER  dim5-8→SIMT_THREAD_MIN
static int64_t GetMaxThread(bool isBigShape, int32_t dimNum)
{
    if (!isBigShape) {
        return (dimNum <= SIMT_FULL_THREAD_MAX_DIM) ? SIMT_THREAD_MAX : SIMT_THREAD_HALF;
    }
    if (dimNum == BIGSHAPE_DIM_FULL) {
        return SIMT_THREAD_MAX;
    }
    if (dimNum == BIGSHAPE_DIM_HALF) {
        return SIMT_THREAD_HALF;
    }
    if (dimNum <= BIGSHAPE_HALF_THREAD_MAX_DIM) {
        return SIMT_THREAD_QUARTER;
    }
    return SIMT_THREAD_MIN;
}

void StrideSliceTiling::CalcThreadNum()
{
    int64_t maxThread = GetMaxThread(isShapeExceedUint32_, dimNum_);
    threadNum_ = static_cast<uint16_t>(PickThread(maxThread, mainBlockFactor_));
}

static bool CheckPackAlignment(SliceParametersRuntime2& sliceParam, const int64_t* inputShapeProd, int32_t dimNum,
                               int64_t packFactor)
{
    int32_t lastAxis = dimNum - 1;
    if (sliceParam.beginList.GetDim(lastAxis) % packFactor != 0) {
        return false;
    }
    for (int32_t i = 0; i < lastAxis; i++) {
        if ((sliceParam.beginList.GetDim(i) * inputShapeProd[i + 1]) % packFactor != 0) {
            return false;
        }
    }
    return true;
}

void StrideSliceTiling::CalSimtPackFactor()
{
    simtPackFactor_ = 1;
    if (isStrideNeg_ || lastOneStride_ != 1) {
        return;
    }

    int64_t maxThread = GetMaxThread(isShapeExceedUint32_, dimNum_);
    mainBlockFactor_ = Ops::Base::CeilDiv(outputShapeProd_[0], coreNum_);

    // 收集所有有效合并级别: b16→b32→b64→b128
    const int64_t targets[] = {SIMT_PACK_BYTES_2, SIMT_PACK_BYTES_4, SIMT_PACK_BYTES_8, SIMT_PACK_BYTES_16};
    int64_t lastDimBytes = lastOneOutputDim_ * xDtypeSize_;
    struct Level {
        int64_t packFactor;
        int64_t target;
        int64_t mergedBlock;
    };
    Level levels[4];
    int levelCnt = 0;

    for (auto target : targets) {
        if (xDtypeSize_ >= target) {
            continue;
        }
        if (lastDimBytes % target != 0) {
            continue;
        }
        int64_t p = target / xDtypeSize_;
        if (!CheckPackAlignment(sliceParam_, inputShapeProd_, dimNum_, p)) {
            continue;
        }
        int64_t block = Ops::Base::CeilDiv(outputShapeProd_[0] / p, coreNum_);
        levels[levelCnt] = {p, target, block};
        levelCnt++;
    }

    if (levelCnt == 0) {
        return;
    }

    // L0 不合并判定
    if (levels[0].mergedBlock <= 1) {
        return;
    }

    // 只有 int8 才有 b16 这个中间层
    bool hasB16 = (levels[0].target == SIMT_PACK_BYTES_2);

    if (hasB16) {
        // int8: L0=b16, L1=b32(无条件 base), L2+=升级需 block>=maxThread
        if (levels[1].mergedBlock <= 1) {
            simtPackFactor_ = levels[0].packFactor;
            return;
        }
        simtPackFactor_ = levels[1].packFactor; // b32 base
        for (int i = 2; i < levelCnt; i++) {
            if (levels[i].mergedBlock >= maxThread) {
                simtPackFactor_ = levels[i].packFactor;
            }
        }
    } else {
        // int16/int32/int64: L0=base, L1+=升级需 block>=maxThread
        simtPackFactor_ = levels[0].packFactor;
        for (int i = 1; i < levelCnt; i++) {
            if (levels[i].mergedBlock >= maxThread) {
                simtPackFactor_ = levels[i].packFactor;
            }
        }
    }
}

void StrideSliceTiling::SetSIMTTilingMode()
{
    realCoreNum_ = coreNum_;
    // key 映射: base(A6 编号，所有芯片一致) + 合并字节数偏移
    int64_t base = isShapeExceedUint32_ ? STRIDED_SLICE_KEY_SIMT_BIG_SHAPE : STRIDED_SLICE_KEY_SIMT;

    // simt pack 仅在 960DT(NpuArch::DAV_9201) 启用, 其余平台只用 base key
    auto platformInfo = tilingContext_->GetPlatformInfo();
    bool isSimtPackSupport = (platformInfo != nullptr) &&
                             (platform_ascendc::PlatformAscendC(platformInfo).GetCurNpuArch() == NpuArch::DAV_9201);
    if (!isSimtPackSupport) {
        tilingKey_ = base;
        return;
    }

    CalSimtPackFactor();

    // 重新计算合并后的 mainBlockFactor_ 和 threadNum_
    int64_t finalElements = outputShapeProd_[0] / simtPackFactor_;
    mainBlockFactor_ = Ops::Base::CeilDiv(finalElements, coreNum_);
    CalcThreadNum();
    OP_LOGD(tilingContext_->GetNodeName(), "SIMT pack: p=%ld, threadNum=%u, finalElements=%ld", simtPackFactor_,
            threadNum_, finalElements);

    if (simtPackFactor_ == 1) {
        tilingKey_ = base;
        return;
    }
    int64_t mergedBytes = simtPackFactor_ * xDtypeSize_;
    int64_t offset = 0;
    switch (mergedBytes) {
        case SIMT_PACK_BYTES_2:
            offset = SIMT_KEY_OFFSET_2B;
            break;
        case SIMT_PACK_BYTES_4:
            offset = SIMT_KEY_OFFSET_4B;
            break;
        case SIMT_PACK_BYTES_8:
            offset = SIMT_KEY_OFFSET_8B;
            break;
        case SIMT_PACK_BYTES_16:
            offset = SIMT_KEY_OFFSET_16B;
            break;
        default:
            break;
    }
    tilingKey_ = base + offset;
}

void StrideSliceTiling::MovAlignV2UbSplitLastOneDim()
{
    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(1);
        // 首尾都是有效的、能取到的数
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(
            (ubFactor_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_);
        mainMoveAlignV2Info_.srcStride = 0U;
        mainMoveAlignV2Info_.dstStride = 0U;
    }

    if (!isStrideNeg_) {
        tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM;
    }
}

void StrideSliceTiling::MovAlignV2UbSplitLastTwoDim()
{
    int64_t srcStride = std::abs(lastTwoStride_ * lastOneInputDim_ * xDtypeSize_);
    if (srcStride > MAX_UINT32_NUM || ubFactor_ > MAX_UINT16_NUM) {
        ResetMovAlignV2Para(lastOneOutputDim_);
        MovAlignV2UbSplitLastOneDim();
        return;
    }

    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(ubFactor_);
        // 首尾都是有效的、能取到的数
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(
            (lastOneOutputDim_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_);
        mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
        mainMoveAlignV2Info_.dstStride = 0U;
    }

    if (!isStrideNeg_) {
        if (!useGather_) {
            // 整块参数
            mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(ubFactor_);
            mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
            mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
            mainMoveAlignV2Info_.dstStride = mainMoveAlignV2Info_.blockLen;

            // 二维shape，且blk和ub都切0轴时，使用特殊模板
            tilingKey_ = (dimNum_ == NUMBER_TWO) ? STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM : STRIDED_SLICE_KEY_MOVE_ALIGN;
        } else {
            tilingKey_ = STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER;
        }
    }
}

void StrideSliceTiling::MovAlignV2UbSplitLastThreeDim()
{
    int64_t srcStride = std::abs(lastTwoStride_ * lastOneInputDim_ * xDtypeSize_);
    int64_t loop1SrcStride = std::abs(lastThreeStride_ * lastOneInputDim_ * lastTwoInputDim_ * xDtypeSize_);
    // compact 模式，最后两根轴一起对齐到block
    int64_t loop1DstStride = Ops::Base::CeilAlign(lastOneOutputDim_ * lastTwoOutputDim_ * xDtypeSize_, BLOCK_SIZE);
    int64_t burstLenNeg = 0;
    if (isStrideNeg_ || useGather_) {
        // 首尾都是有效的、能取到的数
        burstLenNeg = (lastOneOutputDim_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_;
        // 非compact 模式，最后一根轴对齐到block
        loop1DstStride = lastTwoOutputDim_ * Ops::Base::CeilAlign(burstLenNeg, BLOCK_SIZE);
    }
    if (loop1SrcStride > MAX_UINT32_NUM || srcStride > MAX_UINT32_NUM || lastTwoOutputDim_ > MAX_UINT16_NUM ||
        loop1DstStride > MAX_UINT16_NUM || ubFactor_ > MAX_UINT16_NUM) {
        ResetMovAlignV2Para(lastTwoOutputDim_);
        MovAlignV2UbSplitLastTwoDim();
        return;
    }

    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(burstLenNeg);
        mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
        mainMoveAlignV2Info_.dstStride = 0;
        mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(ubFactor_);
        mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
        mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
    }

    if (!isStrideNeg_) {
        tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN;
        if (!useGather_) {
            // 整块参数
            mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
            mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
            mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
            mainMoveAlignV2Info_.dstStride = mainMoveAlignV2Info_.blockLen;
            mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(ubFactor_);
            mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
            mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
        }
    }
}

void StrideSliceTiling::MovAlignV2UbSplitLastFourDim()
{
    int64_t srcStride = std::abs(lastTwoStride_ * lastOneInputDim_ * xDtypeSize_);
    int64_t loop1SrcStride = std::abs(lastThreeStride_ * lastOneInputDim_ * lastTwoInputDim_ * xDtypeSize_);
    int64_t loop2SrcStride = std::abs(lastFourStride_ * lastOneInputDim_ * lastTwoInputDim_ * lastThreeInputDim_ *
                                      xDtypeSize_);
    int64_t loop1DstStride = Ops::Base::CeilAlign(lastOneOutputDim_ * lastTwoOutputDim_ * xDtypeSize_, BLOCK_SIZE);
    int64_t burstLenNeg = 0;
    if (isStrideNeg_ || useGather_) {
        // 首尾都是有效的、能取到的数
        burstLenNeg = (lastOneOutputDim_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_;
        // 非compact 模式，最后一根轴对齐到block
        loop1DstStride = lastTwoOutputDim_ * Ops::Base::CeilAlign(burstLenNeg, BLOCK_SIZE);
    }
    int64_t loop2DstStride = lastThreeOutputDim_ * loop1DstStride;
    if (loop2SrcStride > MAX_UINT32_NUM || loop1SrcStride > MAX_UINT32_NUM || srcStride > MAX_UINT32_NUM ||
        lastTwoOutputDim_ > MAX_UINT16_NUM || lastThreeOutputDim_ > MAX_UINT16_NUM || ubFactor_ > MAX_UINT16_NUM ||
        loop1DstStride > MAX_UINT16_NUM || loop2DstStride > MAX_UINT16_NUM) {
        ResetMovAlignV2Para(lastThreeOutputDim_);
        MovAlignV2UbSplitLastThreeDim();
        return;
    }

    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(burstLenNeg);
        mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
        mainMoveAlignV2Info_.dstStride = 0U;
        mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(lastThreeOutputDim_);
        mainMoveAlignV2Info_.loop2Size = static_cast<uint16_t>(ubFactor_);
        mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
        mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
        mainMoveAlignV2Info_.loop2SrcStride = static_cast<uint32_t>(loop2SrcStride);
        mainMoveAlignV2Info_.loop2DstStride = static_cast<uint16_t>(loop2DstStride);
    }

    if (!isStrideNeg_) {
        tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN;
        if (!useGather_) {
            // 整块参数
            mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
            mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
            mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
            mainMoveAlignV2Info_.dstStride = mainMoveAlignV2Info_.blockLen;
            mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(lastThreeOutputDim_);
            mainMoveAlignV2Info_.loop2Size = static_cast<uint16_t>(ubFactor_);
            mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
            mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
            mainMoveAlignV2Info_.loop2SrcStride = static_cast<uint32_t>(loop2SrcStride);
            mainMoveAlignV2Info_.loop2DstStride = static_cast<uint16_t>(loop2DstStride);
        }
    }
}

void StrideSliceTiling::ResetMovAlignV2Para(int64_t newUbfactor)
{
    ubFactor_ = newUbfactor;
    ubTailFactor_ = 0;
    ubTailTailFactor_ = 0;
    ubIndex_ = ubIndex_ + 1;
    isAllInUb_ = false;
}

/*
input
(A, B, C, D)
outputShape:
(a, b, c, d)

模板1, last_dim>=128B
场景1: ub切d
(a, b, c, d.o, d.i)
整块:
d.i = ubFactor_
step = stride_d * d.i
for i = 0; i * step + d.begin < d.end; i++: // i < d.o
mov2_align_v2(dst, src, nburst = 1, burstlen = d.i * dype, burst_src_stride = d.i * dype,
burst_dst_stride = d.i * dype)

if d % di != 0
尾块：
mov2_align_v2(nburst = 1, burstlen = (d - di * do) * dype, burst_src_stride = 0, burst_dst_stride = 0)

block切a
(a.o, a.i, b, c, d.o, d.i)
for i = 0; i * c.stride + c.begin < c.end; i++:

for i = 0; i * b.stride +b.begin < b.end; i++:

整核:
for i = 0; i < a.i; i++:

尾核:
for i = 0; i < a - a.i * a.o; i++:

场景2:
ub切c
(a, b, c.o, c.i, d)
burstlen = d * dype
nburst = c.i
burst_src_stride = stridec * d * dype

整块:
step = strided * d.i
for i = 0; i < c.o; i++:
mov2_align_v2(dst, src, nburst = c.i, burstlen = d * dype, burst_src_stride = stridec * d * dype,
burst_dst_stride = d * dype)

if c % ci != 0
尾块：
mov2_align_v2(nburst = c - c.o * c.i, burstlen = d * dype, burst_src_stride = stridec * d * dype,
burst_dst_stride = d * dype)

block切a
(a.o, a.i, b, c.o, c.i, d)

for i = 0; i * strideb +b.begin < b.end; i++:

整核:
for i = 0; i < a.i; i++:

尾核:
for i = 0; i < a - a.i * a.o; i++:

场景3:
ub切b
(a, b.o, b.i, c, d)
nburst = c
burstlen = d * dype
burst_src_stride = stridec * d * dype
burst_dst_stride = d * dype

整块:
for i = 0; i < b.o; i++:
loop1_size = b.i
loop1_size_src_stide = strideb * D * C * dtype
loop1_size_dst_stide = d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

尾块：
if b % b.i != 0
loop1_size = b - b.i * b.o
loop1_size_src_stide = strideb * D * C * dtype
loop1_size_dst_stide = d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

场景4:
ub切a
(a.o, a.i, b, c, d)
nburst = c
burstlen = d * dype
burst_src_stride = stridec * D * dype
burst_dst_stride = d * dype
loop1_size = b
loop1_size_src_stide = strideb * D * C * dtype
loop1_size_dst_stide = d * c * dtype

整块:
for i = 0; i < a.o; i++:
loop2_size = a.i
loop2_size_src_stide = stridea * B * D * C * dtype
loop2_size_dst_stide = b * d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

尾块：
if a % a.i != 0
loop2_size = a - a.i * a.o
loop2_size_src_stide = stridea * B * D * C * dtype
loop2_size_dst_stide = b * d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

*/
void StrideSliceTiling::SetMovAlignV2TilingMode()
{
    int32_t shapeSize = dimNum_;
    // ubfactor切-1轴
    if (ubIndex_ == shapeSize - 1L) {
        MovAlignV2UbSplitLastOneDim();
        return;
    }

    // ubfactor切-2轴
    if (ubIndex_ == shapeSize - NUMBER_TWO) {
        MovAlignV2UbSplitLastTwoDim();
        return;
    }

    // ubfactor切-3轴
    if (ubIndex_ == shapeSize - NUMBER_THREE) {
        MovAlignV2UbSplitLastThreeDim();
        return;
    }

    // ubfactor切-4轴
    if (ubIndex_ == shapeSize - NUMBER_FOUR) {
        MovAlignV2UbSplitLastFourDim();
        return;
    }
}

void StrideSliceTiling::SetMoveAlignParams(StridedSliceMoveAlignParams& params, const MoveAlignV2Info& actInfo)
{
    params.blockCount = actInfo.blockCount;
    params.blockLen = actInfo.blockLen;
    params.srcStride = actInfo.srcStride;
    params.dstStride = actInfo.dstStride;
    params.loop1Size = actInfo.loop1Size;
    params.loop2Size = actInfo.loop2Size;
    params.loop1SrcStride = actInfo.loop1SrcStride;
    params.loop1DstStride = actInfo.loop1DstStride;
    params.loop2SrcStride = actInfo.loop2SrcStride;
    params.loop2DstStride = actInfo.loop2DstStride;
}

void StrideSliceTiling::SetShortMoveAlignParams(StridedSliceShortMoveAlignParams& params,
                                                const MoveAlignV2Info& actInfo)
{
    params.blockCount = actInfo.blockCount;
    params.blockLen = actInfo.blockLen;
    params.srcStride = actInfo.srcStride;
    params.dstStride = actInfo.dstStride;
}

void StrideSliceTiling::SetRowsStepsParams(StridedSliceTilingData& tilingData)
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - NUMBER_TWO; i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    int64_t inLoopSteps = 1;
    int64_t outLoopSteps = 1;
    // ub非最后一根轴
    if (ubIndex_ != dimNum_ - 1) {
        inLoopSteps = ubFactor_ * strides_[ubIndex_] * inputSteps_[ubIndex_ + 1];
        outLoopSteps = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps = ubFactor_ * strides_[ubIndex_];
        outLoopSteps = ubFactor_;
    }
    tilingData.ubInLoopSteps = inLoopSteps;
    tilingData.ubOutLoopSteps = outLoopSteps;

    std::copy(rowsOffsetSteps_, rowsOffsetSteps_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.rowsOffsetSteps);
    std::copy(inputSteps_, inputSteps_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.inputSteps);
}

void StrideSliceTiling::CalStridedSliceRowsStepsParams()
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - NUMBER_TWO; i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    // ub非最后一根轴
    if (ubIndex_ != dimNum_ - 1) {
        inLoopSteps_ = ubFactor_ * strides_[ubIndex_] * inputSteps_[ubIndex_ + 1];
        outLoopSteps_ = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps_ = ubFactor_ * strides_[ubIndex_];
        outLoopSteps_ = ubFactor_;
    }
}

void StrideSliceTiling::FillStridedSliceBaseTilingData(StridedSliceBaseTilingData& tilingData)
{
    if (isStrideNeg_ || useGather_) {
        tilingData.ubSize = ubSizeOutput_;
    } else {
        tilingData.ubSize = ubSize_;
    }
    tilingData.ubIndex = ubIndex_;
    tilingData.ubFactor = ubFactor_;
    tilingData.ubTailFactor = ubTailFactor_;
    tilingData.ubTailTailFactor = ubTailTailFactor_;
    tilingData.realCoreNum = realCoreNum_;
    tilingData.inputDims = dimNum_;
    tilingData.blkIndex = blkIndex_;
    tilingData.blkFactor = blkFactor_;
    tilingData.blkTailFactor = blkTailFactor_;
    std::copy(begin_, begin_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.begin);
    std::copy(strides_, strides_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.strides);
    std::copy(outputShape_, outputShape_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.outputShape);

    CalStridedSliceRowsStepsParams();
    tilingData.ubInLoopSteps = inLoopSteps_;
    tilingData.ubOutLoopSteps = outLoopSteps_;
    std::copy(inputSteps_, inputSteps_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.inputSteps);
    std::copy(rowsOffsetSteps_, rowsOffsetSteps_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData.rowsOffsetSteps);
}

void StrideSliceTiling::FillStridedSliceTilingData100()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData100.");
    FillStridedSliceBaseTilingData(maTilingData_.stridedSliceBaseTilingData);

    SetMoveAlignParams(maTilingData_.moveAlignParams, mainMoveAlignV2Info_);
}

void StrideSliceTiling::FillStridedSliceTilingData101()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData101.");
    FillStridedSliceBaseTilingData(maLastDimTilingData_.stridedSliceBaseTilingData);
}

void StrideSliceTiling::FillStridedSliceTilingDataNDDMA()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingDataNDDMA.");
    FillStridedSliceBaseTilingData(nddmaTilingData_.stridedSliceBaseTilingData);
    nddmaTilingData_.ubSizeInput = ubSizeInput_;
    nddmaTilingData_.nddmaTotalNum = nddmaTotalNum_;
    std::copy(nddmaLoopSize_, nddmaLoopSize_ + MAX_NDDMA_UB_SPLIT_AXIS_NUM, nddmaTilingData_.nddmaLoopSize);
    std::copy(nddmaLoopSrcStride_, nddmaLoopSrcStride_ + MAX_NDDMA_UB_SPLIT_AXIS_NUM,
              nddmaTilingData_.nddmaLoopSrcStride);
    std::copy(nddmaLoopDstStride_, nddmaLoopDstStride_ + MAX_NDDMA_UB_SPLIT_AXIS_NUM,
              nddmaTilingData_.nddmaLoopDstStride);
}

void StrideSliceTiling::SetRowsStepsParamsFor150(StridedSliceMALast2DimTilingData& tilingData)
{
    CalStridedSliceRowsStepsParams();
    tilingData.ubInLoopSteps = inLoopSteps_;
    tilingData.ubOutLoopSteps = outLoopSteps_;
    std::copy(inputSteps_, inputSteps_ + NUMBER_TWO, tilingData.inputSteps);
}

void StrideSliceTiling::FillStridedSliceTilingData150()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData150.");
    int64_t begin[NUMBER_TWO];
    int64_t strides[NUMBER_TWO];
    int64_t outputShape[NUMBER_TWO];
    for (size_t i = 0; i < NUMBER_TWO; i++) {
        begin[i] = begin_[i];
        strides[i] = strides_[i];
        outputShape[i] = outputShape_[i];
    }
    maLast2DimTilingData_.ubSize = ubSize_;
    maLast2DimTilingData_.blkFactor = blkFactor_;
    maLast2DimTilingData_.blkTailFactor = blkTailFactor_;
    maLast2DimTilingData_.ubFactor = ubFactor_;
    maLast2DimTilingData_.ubTailFactor = ubTailFactor_;
    maLast2DimTilingData_.ubTailTailFactor = ubTailTailFactor_;
    maLast2DimTilingData_.realCoreNum = realCoreNum_;

    SetShortMoveAlignParams(maLast2DimTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    std::copy(begin, begin + NUMBER_TWO, maLast2DimTilingData_.begin);
    std::copy(strides, strides + NUMBER_TWO, maLast2DimTilingData_.strides);
    std::copy(outputShape, outputShape + NUMBER_TWO, maLast2DimTilingData_.outputShape);
    SetRowsStepsParamsFor150(maLast2DimTilingData_);
}

void StrideSliceTiling::FillStridedSliceTilingDataSIMT()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingDataSIMT.");
    simtTilingData_.isEmptyTensor = isEmptyTensor_;
    simtTilingData_.inputDims = dimNum_;
    simtTilingData_.threadNum = threadNum_;

    if (simtPackFactor_ > 1) {
        int64_t p = simtPackFactor_;
        int32_t lastAxis = dimNum_ - 1;
        int64_t packedBegin[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
        int64_t packedStrides[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
        int64_t packedOutputProd[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
        int64_t packedInputProd[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
        for (int32_t i = 0; i < dimNum_; i++) {
            packedOutputProd[i] = outputShapeProd_[i] / p;
            packedInputProd[i] = inputShapeProd_[i] / p;
            packedBegin[i] = begin_[i];
            packedStrides[i] = strides_[i];
        }
        packedBegin[lastAxis] = begin_[lastAxis] / p;
        std::copy(packedBegin, packedBegin + MAX_AXIS_NUM_FOR_STRIDESLICE, simtTilingData_.begin);
        std::copy(packedStrides, packedStrides + MAX_AXIS_NUM_FOR_STRIDESLICE, simtTilingData_.strides);
        std::copy(packedOutputProd, packedOutputProd + MAX_SIMT_UB_SPLIT_AXIS_NUM, simtTilingData_.outputShapeProd);
        std::copy(packedInputProd, packedInputProd + MAX_SIMT_UB_SPLIT_AXIS_NUM, simtTilingData_.inputShapeProd);
    } else {
        std::copy(begin_, begin_ + MAX_AXIS_NUM_FOR_STRIDESLICE, simtTilingData_.begin);
        std::copy(strides_, strides_ + MAX_AXIS_NUM_FOR_STRIDESLICE, simtTilingData_.strides);
        std::copy(outputShapeProd_, outputShapeProd_ + MAX_SIMT_UB_SPLIT_AXIS_NUM, simtTilingData_.outputShapeProd);
        std::copy(inputShapeProd_, inputShapeProd_ + MAX_SIMT_UB_SPLIT_AXIS_NUM, simtTilingData_.inputShapeProd);
    }
}

void StrideSliceTiling::FillStridedSliceTilingData300()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData300.");
    FillStridedSliceBaseTilingData(maGatherTilingData_.stridedSliceBaseTilingData);
    maGatherTilingData_.ubSizeInput = ubSizeInput_;
    SetMoveAlignParams(maGatherTilingData_.moveAlignParams, mainMoveAlignV2Info_);
}

void StrideSliceTiling::FillStridedSliceTilingData301()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData301.");
    FillStridedSliceBaseTilingData(maUB2UBTilingData_.stridedSliceBaseTilingData);
    maUB2UBTilingData_.ubSizeInput = ubSizeInput_;
    SetMoveAlignParams(maUB2UBTilingData_.moveAlignParams, mainMoveAlignV2Info_);
}

void StrideSliceTiling::FillStridedSliceTilingDataOther()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering other FillTilingData.");
    if (isStrideNeg_ || useGather_) {
        tilingData_.ubSize = ubSizeOutput_;
        tilingData_.ubSizeInput = ubSizeInput_;
    } else {
        tilingData_.ubSize = ubSize_;
    }
    tilingData_.coreNum = coreNum_;
    tilingData_.ubIndex = ubIndex_;
    tilingData_.ubFactor = ubFactor_;
    tilingData_.ubTailFactor = ubTailFactor_;
    tilingData_.ubTailTailFactor = ubTailTailFactor_;
    tilingData_.realCoreNum = realCoreNum_;
    tilingData_.inputDims = dimNum_;
    tilingData_.blkIndex = blkIndex_;
    tilingData_.blkFactor = blkFactor_;
    tilingData_.blkTailFactor = blkTailFactor_;
    tilingData_.xDtypeSize = xDtypeSize_;
    tilingData_.tilingKey = tilingKey_;
    tilingData_.nddmaTotalNum = nddmaTotalNum_;
    std::copy(nddmaLoopSize_, nddmaLoopSize_ + MAX_NDDMA_UB_SPLIT_AXIS_NUM, tilingData_.nddmaLoopSize);
    std::copy(nddmaLoopSrcStride_, nddmaLoopSrcStride_ + MAX_NDDMA_UB_SPLIT_AXIS_NUM, tilingData_.nddmaLoopSrcStride);
    std::copy(nddmaLoopDstStride_, nddmaLoopDstStride_ + MAX_NDDMA_UB_SPLIT_AXIS_NUM, tilingData_.nddmaLoopDstStride);
    std::copy(outputShapeProd_, outputShapeProd_ + MAX_SIMT_UB_SPLIT_AXIS_NUM, tilingData_.outputShapeProd);
    std::copy(inputShapeProd_, inputShapeProd_ + MAX_SIMT_UB_SPLIT_AXIS_NUM, tilingData_.inputShapeProd);
    tilingData_.isShapeExceedUint32 = isShapeExceedUint32_;
    tilingData_.isEmptyTensor = isEmptyTensor_;

    SetMoveAlignParams(tilingData_.moveAlignParams, mainMoveAlignV2Info_);

    std::copy(begin_, begin_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData_.begin);
    std::copy(strides_, strides_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData_.strides);
    std::copy(outputShape_, outputShape_ + MAX_AXIS_NUM_FOR_STRIDESLICE, tilingData_.outputShape);

    SetRowsStepsParams(tilingData_);
}

void StrideSliceTiling::FillTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData.");
    for (size_t i = 0; i < sliceParam_.inputShape.GetDimNum(); i++) {
        begin_[i] = sliceParam_.beginList.GetDim(i);
        end_[i] = sliceParam_.endList.GetDim(i);
        strides_[i] = sliceParam_.strideList.GetDim(i);
        inputShape_[i] = sliceParam_.inputShape.GetDim(i);
        outputShape_[i] = sliceParam_.outputShape.GetDim(i);
    }
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            FillStridedSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            FillStridedSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
        case STRIDED_SLICE_KEY_NDDMA_GATHER:
        case STRIDED_SLICE_KEY_NDDMA_UB2UB:
            FillStridedSliceTilingDataNDDMA();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            FillStridedSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_SIMT:
        case STRIDED_SLICE_KEY_SIMT_PACK_2B:
        case STRIDED_SLICE_KEY_SIMT_PACK_4B:
        case STRIDED_SLICE_KEY_SIMT_PACK_8B:
        case STRIDED_SLICE_KEY_SIMT_PACK_16B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_2B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_4B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_8B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_16B:
            FillStridedSliceTilingDataSIMT();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
            FillStridedSliceTilingData300();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB:
            FillStridedSliceTilingData301();
            break;
        default:
            FillStridedSliceTilingDataOther();
            break;
    }
}

void StrideSliceTiling::PrintTilingData()
{
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            PrintStridedSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            PrintStridedSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
        case STRIDED_SLICE_KEY_NDDMA_GATHER:
        case STRIDED_SLICE_KEY_NDDMA_UB2UB:
            PrintStridedSliceTilingDataNDDMA();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            PrintStridedSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_SIMT:
        case STRIDED_SLICE_KEY_SIMT_PACK_2B:
        case STRIDED_SLICE_KEY_SIMT_PACK_4B:
        case STRIDED_SLICE_KEY_SIMT_PACK_8B:
        case STRIDED_SLICE_KEY_SIMT_PACK_16B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_2B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_4B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_8B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_16B:
            PrintStridedSliceTilingDataSIMT();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
            PrintStridedSliceTilingData300();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB:
            PrintStridedSliceTilingData301();
            break;
        default:
            PrintStridedSliceTilingDataOther();
            break;
    }
}

void StrideSliceTiling::PrintStridedSliceBaseTilingData(StridedSliceBaseTilingData& tilingData)
{
    OP_LOGI(tilingContext_->GetNodeName(), "StridedSliceBaseTilingData is isStrideNeg_:%d, ubSize:%d, realCoreNum:%d, \
            ubIndex:%d, ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkIndex:%d, blkFactor:%ld, blkTailFactor:%ld, \
            begin:%s, end:%s, stride:%s, inputShape:%s, outputShape:%s, \
            rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld, ubOutLoopSteps:%ld",
            isStrideNeg_, tilingData.ubSize, tilingData.realCoreNum, tilingData.ubIndex, tilingData.ubFactor,
            tilingData.ubTailFactor, tilingData.ubTailTailFactor, tilingData.blkIndex, tilingData.blkFactor,
            tilingData.blkTailFactor, ArrayToStr(tilingData.begin, dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
            ArrayToStr(tilingData.strides, dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
            ArrayToStr(tilingData.outputShape, dimNum_).c_str(),
            ArrayToStr(tilingData.rowsOffsetSteps, dimNum_).c_str(), ArrayToStr(tilingData.inputSteps, dimNum_).c_str(),
            ArrayToStr(outputSteps_, dimNum_).c_str(), tilingData.ubInLoopSteps, tilingData.ubOutLoopSteps);
}

void StrideSliceTiling::PrintStridedSliceTilingData100()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMATilingData:");
    PrintStridedSliceBaseTilingData(maTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(tilingContext_->GetNodeName(), "StridedSliceMATilingData is \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
            mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
            mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
            mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
            mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingData101()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMALastDimTilingData:");
    PrintStridedSliceBaseTilingData(maLastDimTilingData_.stridedSliceBaseTilingData);
}

void StrideSliceTiling::PrintStridedSliceTilingDataNDDMA()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceNDDMATilingData:");
    PrintStridedSliceBaseTilingData(nddmaTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(tilingContext_->GetNodeName(), "StridedSliceNDDMATilingData is  ubSizeInput:%d\
    nddmaTotalNum:%ld, nddmaLoopSize:%s, nddmaLoopSrcStride: %s, nddmaLoopDstStride: %s",
            nddmaTilingData_.ubSizeInput, nddmaTilingData_.nddmaTotalNum,
            ArrayToStr(nddmaTilingData_.nddmaLoopSize, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
            ArrayToStr(nddmaTilingData_.nddmaLoopSrcStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
            ArrayToStr(nddmaTilingData_.nddmaLoopDstStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str());
}

void StrideSliceTiling::PrintStridedSliceTilingData150()
{
    OP_LOGI(tilingContext_->GetNodeName(), "StridedSliceMALast2DimTilingData is ubSize:%d, realCoreNum:%d, \
            ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkFactor:%ld, blkTailFactor:%ld, \
            begin:%s, end:%s, inputShape:%s, outputShape:%s, \
            inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld, ubOutLoopSteps:%ld, \
            moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u",
            maLast2DimTilingData_.ubSize, maLast2DimTilingData_.realCoreNum, maLast2DimTilingData_.ubFactor,
            maLast2DimTilingData_.ubTailFactor, maLast2DimTilingData_.ubTailTailFactor, maLast2DimTilingData_.blkFactor,
            maLast2DimTilingData_.blkTailFactor, ArrayToStr(maLast2DimTilingData_.begin, dimNum_).c_str(),
            ArrayToStr(end_, dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
            ArrayToStr(maLast2DimTilingData_.outputShape, dimNum_).c_str(),
            ArrayToStr(maLast2DimTilingData_.inputSteps, dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str(),
            maLast2DimTilingData_.ubInLoopSteps, maLast2DimTilingData_.ubOutLoopSteps, mainMoveAlignV2Info_.blockCount,
            mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingDataSIMT()
{
    OP_LOGI(tilingContext_->GetNodeName(),
            "StridedSliceSIMTTilingData is isStrideNeg_:%d, isEmptyTensor:%d, begin:%s, stride:%s \
            outputShapeProd:%s, inputShapeProd:%s",
            isStrideNeg_, simtTilingData_.isEmptyTensor, ArrayToStr(simtTilingData_.begin, dimNum_).c_str(),
            ArrayToStr(simtTilingData_.strides, dimNum_).c_str(),
            ArrayToStr(simtTilingData_.outputShapeProd, MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str(),
            ArrayToStr(simtTilingData_.inputShapeProd, MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str());
}

void StrideSliceTiling::PrintStridedSliceTilingData300()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMAGatherTilingData:");
    PrintStridedSliceBaseTilingData(maGatherTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(tilingContext_->GetNodeName(), "StridedSliceMAGatherTilingData is ubSizeInput:%d \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
            maGatherTilingData_.ubSizeInput, mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen,
            mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size,
            mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
            mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingData301()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMAUB2UBTilingData:");
    PrintStridedSliceBaseTilingData(maUB2UBTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(tilingContext_->GetNodeName(), "StridedSliceMAUB2UBTilingData is ubSizeInput:%d \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
            maUB2UBTilingData_.ubSizeInput, mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen,
            mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size,
            mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
            mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingDataOther()
{
    OP_LOGI(tilingContext_->GetNodeName(),
            "tilingData is isStrideNeg_:%d ubSize:%ld ubSizeInput:%ld, coreNum:%ld, realCoreNum:%ld, \
          ubIndex:%ld, ubFactor:%ld, ubTailFactor:%ld, ubTailTailFactor:%ld, \
          blkIndex:%ld, blkFactor:%ld, blkTailFactor:%ld, xDtypeSize:%ld, tilingKey:%ld, isShapeExceedUint32:%d, \
          isEmptyTensor:%d, begin:%s, end:%s, stride:%s, inputShape:%s, outputShape:%s, \
          rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s",
            isStrideNeg_, tilingData_.ubSize, ubSizeInput_, coreNum_, tilingData_.realCoreNum, tilingData_.ubIndex,
            tilingData_.ubFactor, tilingData_.ubTailFactor, tilingData_.ubTailTailFactor, tilingData_.blkIndex,
            tilingData_.blkFactor, tilingData_.blkTailFactor, tilingData_.xDtypeSize, tilingData_.tilingKey,
            tilingData_.isShapeExceedUint32, tilingData_.isEmptyTensor, ArrayToStr(tilingData_.begin, dimNum_).c_str(),
            ArrayToStr(end_, dimNum_).c_str(), ArrayToStr(tilingData_.strides, dimNum_).c_str(),
            ArrayToStr(inputShape_, dimNum_).c_str(), ArrayToStr(tilingData_.outputShape, dimNum_).c_str(),
            ArrayToStr(tilingData_.rowsOffsetSteps, dimNum_).c_str(),
            ArrayToStr(tilingData_.inputSteps, dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str());

    OP_LOGI(tilingContext_->GetNodeName(), "tilingData is nddmaTotalNum:%ld nddmaLoopSize:%s, nddmaLoopSrcStride: %s, \
          nddmaLoopDstStride: %s, moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
          loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u outputShapeProd: %s \
          inputShapeProd: %s Tiling4StrideSlice ends.",
            tilingData_.nddmaTotalNum, ArrayToStr(tilingData_.nddmaLoopSize, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
            ArrayToStr(tilingData_.nddmaLoopSrcStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
            ArrayToStr(tilingData_.nddmaLoopDstStride, MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
            mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
            mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
            mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
            mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride,
            ArrayToStr(tilingData_.outputShapeProd, MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str(),
            ArrayToStr(tilingData_.inputShapeProd, MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str());
}

void StrideSliceTiling::SetBlockDimAndTilingKey()
{
    tilingContext_->SetBlockDim(realCoreNum_);
    tilingContext_->SetTilingKey(tilingKey_);

    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN: {
            auto* data = tilingContext_->GetTilingData<StridedSliceMATilingData>();
            *data = maTilingData_;
            break;
        }
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM: {
            auto* data = tilingContext_->GetTilingData<StridedSliceMALastDimTilingData>();
            *data = maLastDimTilingData_;
            break;
        }
        case STRIDED_SLICE_KEY_NDDMA:
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
        case STRIDED_SLICE_KEY_NDDMA_GATHER:
        case STRIDED_SLICE_KEY_NDDMA_UB2UB: {
            auto* data = tilingContext_->GetTilingData<StridedSliceNDDMATilingData>();
            *data = nddmaTilingData_;
            break;
        }
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM: {
            auto* data = tilingContext_->GetTilingData<StridedSliceMALast2DimTilingData>();
            *data = maLast2DimTilingData_;
            break;
        }
        case STRIDED_SLICE_KEY_SIMT:
        case STRIDED_SLICE_KEY_SIMT_PACK_2B:
        case STRIDED_SLICE_KEY_SIMT_PACK_4B:
        case STRIDED_SLICE_KEY_SIMT_PACK_8B:
        case STRIDED_SLICE_KEY_SIMT_PACK_16B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_2B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_4B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_8B:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_16B: {
            auto* data = tilingContext_->GetTilingData<StridedSliceSIMTTilingData>();
            *data = simtTilingData_;
            break;
        }
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER: {
            auto* data = tilingContext_->GetTilingData<StridedSliceMAGatherTilingData>();
            *data = maGatherTilingData_;
            break;
        }
        case STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB: {
            auto* data = tilingContext_->GetTilingData<StridedSliceMAUB2UBTilingData>();
            *data = maUB2UBTilingData_;
            break;
        }
        default: {
            auto* data = tilingContext_->GetTilingData<StridedSliceTilingData>();
            *data = tilingData_;
            break;
        }
    }
}

std::string StrideSliceTiling::ArrayToStr(const int64_t* arr, size_t aSize) const
{
    std::string result;
    for (size_t i = 0; i < aSize; i++) {
        result = result + std::to_string(arr[i]) + " ";
    }
    return result;
}

template <typename T>
static bool AssignInputValueConst(const gert::Tensor* tensor, ops::QuickVector& sliceList, bool isAscendc, bool depend,
                                  bool& isConst)
{
    int32_t dimNum = tensor->GetShapeSize();
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF((!isAscendc || depend) && data == nullptr,
                OP_LOGE(OP_NAME, "get const value fail, check input is const or not."), return false);
    if (!data) {
        isConst = false;
    } else {
        isConst = true;
        sliceList.SetDimNum(dimNum);
        for (int32_t i = 0; i < dimNum; i++) {
            sliceList[i] = static_cast<int64_t>(data[i]);
        }
    }
    return true;
}

static bool ConstructSliceList(const gert::Tensor* tensor, ops::QuickVector& sliceList, bool isAscendc, bool depend,
                               bool& isConst)
{
    if (tensor->GetDataType() == ge::DT_INT32) {
        return AssignInputValueConst<int32_t>(tensor, sliceList, isAscendc, depend, isConst);
    } else if (tensor->GetDataType() == ge::DT_INT64) {
        return AssignInputValueConst<int64_t>(tensor, sliceList, isAscendc, depend, isConst);
    } else {
        OP_LOGD(OP_NAME, "data type is invalid %d", tensor->GetDataType());
        return false;
    }
    return true;
}

static void ConstructSliceShape(const gert::StorageShape* storage, gert::Shape& param)
{
    const gert::Shape& shape = Ops::Base::EnsureNotScalar(storage->GetStorageShape());
    int32_t dimNum = static_cast<int32_t>(shape.GetDimNum());
    param.SetDimNum(dimNum);
    for (int32_t i = 0; i < dimNum; i++) {
        param[i] = shape.GetDim(i);
    }
}

static bool CheckStride(ops::QuickVector& stride, const gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const StridedSliceCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    for (size_t i = 0; i < stride.GetDimNum(); i++) {
        if (stride[i] == 0) {
            OP_LOGD(OP_NAME, "stride is %ld, it must be non-zero.", stride[i]);
            return false;
        }
        if (!compileInfo->isAscendc && stride[i] < 0) {
            OP_LOGD(OP_NAME, "stride is %ld, if not ascendc, it must be greater than zero.", stride[i]);
            return false;
        }
    }
    return true;
}

static ge::graphStatus ConstructSliceParam(const gert::TilingContext* context, SliceParametersRuntime2& sliceParam,
                                           bool isAscendc)
{
    // construct slice_param.input, slice_param.output_shape
    const gert::StorageShape* xStorage = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::StorageShape* yStorage = context->GetOutputShape(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    ConstructSliceShape(xStorage, sliceParam.inputShape);
    ConstructSliceShape(yStorage, sliceParam.outputShape);

    // construct slice_param.begin_list, slice_param.end_list, slice_param.stride_list
    const gert::Tensor* tensorBegin = context->GetInputTensor(INDEX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorBegin);
    const gert::Tensor* tensorEnd = context->GetInputTensor(INDEX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorEnd);
    const gert::Tensor* tensorStrides = context->GetInputTensor(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorStrides);
    if (!ConstructSliceList(tensorBegin, sliceParam.beginList, isAscendc, false, sliceParam.isBeginConst)) {
        return ge::GRAPH_FAILED;
    }
    if (!ConstructSliceList(tensorEnd, sliceParam.endList, isAscendc, false, sliceParam.isEndConst)) {
        return ge::GRAPH_FAILED;
    }
    bool isStridesConst = true;
    if (!ConstructSliceList(tensorStrides, sliceParam.strideList, isAscendc, true, isStridesConst)) {
        return ge::GRAPH_FAILED;
    }

    // check slice_param.stride_list valid, only support value==1
    if (!CheckStride(sliceParam.strideList, context)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ProcBeginEndUnconst(const gert::TilingContext* context, SliceParametersRuntime2& sliceParam)
{
    if (sliceParam.isBeginConst && sliceParam.isEndConst) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::Tensor* tensorBegin = context->GetInputTensor(INDEX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorBegin);
    const gert::Tensor* tensorEnd = context->GetInputTensor(INDEX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorEnd);

    int64_t beginLen = -1;
    beginLen = std::max(tensorBegin->GetShapeSize(), beginLen);
    beginLen = std::max(tensorEnd->GetShapeSize(), beginLen);
    beginLen = std::max(static_cast<int64_t>(sliceParam.strideList.GetDimNum()), beginLen);

    OP_CHECK_IF(beginLen == -1, OP_LOGE(OP_NAME, "beginLen invalid while nonconst"), return ge::GRAPH_FAILED);

    // begin or end被mask掉的话，infershape会依据strides的正负重新给begin/end赋值
    // 此处只需要考虑strides为正
    if (!sliceParam.isBeginConst) {
        sliceParam.beginList.SetDimNum(beginLen);
        for (int32_t i = 0; i < beginLen; i++) {
            sliceParam.beginList[i] = 0;
        }
    }

    // 参考rt1.0 infershape
    if (!sliceParam.isEndConst) {
        sliceParam.endList.SetDimNum(beginLen);
        int64_t inputDims = sliceParam.inputShape.GetDimNum();
        int64_t minLen = std::min(beginLen, inputDims);
        for (int32_t i = 0; i < minLen; i++) {
            sliceParam.endList[i] = sliceParam.inputShape.GetDim(i);
        }
        for (int32_t i = minLen; i < beginLen; i++) {
            sliceParam.endList[i] = sliceParam.inputShape.GetDim(inputDims - 1);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static void ReconstructSliceParamByInferShape(ops::StridedSliceParams& inputParams, gert::Shape& shapeOutput,
                                              SliceParametersRuntime2& sliceParam)
{
    sliceParam.beginList = inputParams.begin;
    sliceParam.endList = inputParams.end;
    sliceParam.strideList = inputParams.strides;
    sliceParam.inputShape.SetDimNum(0);
    for (size_t i = 0; i < inputParams.input_shape.GetDimNum(); i++) {
        sliceParam.inputShape.AppendDim(inputParams.input_shape.GetDim(i));
    }
    for (size_t i = 0; i < shapeOutput.GetDimNum(); i++) {
        sliceParam.outputShape.AppendDim(shapeOutput.GetDim(i));
    }
}

static void MakePerformanceParamsNeg(SliceParametersRuntime2& param)
{
    if (!param.isBeginConst || !param.isEndConst) {
        return;
    }
    OP_LOGI("", "before handle negative perf slice params: %s", param.to_string().c_str());
    SliceParametersRuntime2 perfParams;
    perfParams.isBeginConst = param.isBeginConst;
    perfParams.isEndConst = param.isEndConst;
    size_t perfSize = 0;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        const auto inputShapeI = param.inputShape[i];
        const auto outputShapeI = param.outputShape[i];
        const auto beginI = param.beginList[i];
        const auto endI = param.endList[i];
        const auto stride_i = endI > beginI ? std::min(param.strideList[i], endI - beginI) :
                                              std::max(param.strideList[i], endI - beginI);
        if (inputShapeI == 1 && outputShapeI == 1 && i != 0) {
            continue;
        }
        // Continuous stride=1 axis fused already. Continuous stride=-1 can fuse too.
        if (i == 0 || inputShapeI != outputShapeI || stride_i != -1 || perfParams.strideList[perfSize - 1] != -1) {
            perfParams.inputShape.AppendDim(inputShapeI);
            perfParams.outputShape.AppendDim(outputShapeI);
            perfParams.beginList.AppendDim(beginI);
            perfParams.endList.AppendDim(endI);
            perfParams.strideList.AppendDim(stride_i);
            perfSize++;
            continue;
        }

        const auto perfIndex = static_cast<size_t>(perfSize - 1);
        perfParams.inputShape[perfIndex] *= inputShapeI;
        perfParams.outputShape[perfIndex] *= outputShapeI;
        perfParams.beginList[perfIndex] = perfParams.beginList[perfIndex] * inputShapeI + inputShapeI - 1;
        perfParams.endList[perfIndex] = perfParams.endList[perfIndex] * inputShapeI + inputShapeI - 1;
        perfParams.strideList[perfIndex] = -1;
    }

    param = perfParams;

    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        if (param.outputShape[i] == 1 && param.strideList[i] < 0) {
            param.strideList[i] = 1;
            param.endList[i] = param.beginList[i] + 1;
        }
    }
}

static void MakePerformanceParams(SliceParametersRuntime2& param, bool isAdjustLastStride)
{
    if (param.outputShape.GetDimNum() == 0) {
        return;
    }

    bool hasNegStride = false;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        if (param.strideList[i] < 0) {
            hasNegStride = true;
            break;
        }
    }

    // Adjust params while input[-1] can be divided by stride[-1]
    if (isAdjustLastStride && !hasNegStride && param.isBeginConst && param.isEndConst) {
        size_t th = param.inputShape.GetDimNum();
        const auto inputLastDim = param.inputShape[th - 1U];
        const auto beginLastDim = param.beginList[th - 1U];
        const auto strideLastDim = param.strideList[th - 1U];
        const auto outputLastDim = param.outputShape[th - 1U];
        if (strideLastDim > 1 && inputLastDim % strideLastDim == 0 && beginLastDim / strideLastDim == 0 &&
            outputLastDim == inputLastDim / strideLastDim) {
            param.inputShape[th - 1] = inputLastDim / strideLastDim;
            param.inputShape.AppendDim(strideLastDim);

            param.beginList[th - 1] = beginLastDim / strideLastDim;
            param.beginList.AppendDim(beginLastDim % strideLastDim);

            param.strideList[th - 1U] = 1U;
            param.strideList.AppendDim(1);

            param.outputShape.AppendDim(1);

            param.endList[th - 1] = param.beginList[th - 1] + outputLastDim;
            param.endList.AppendDim(param.beginList[th] + 1);
        }
    }

    SliceParametersRuntime2 perfParams;
    perfParams.isBeginConst = param.isBeginConst;
    perfParams.isEndConst = param.isEndConst;
    size_t perfSize = 0;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        const auto inputShapeI = param.inputShape[i];
        const auto outputShapeI = param.outputShape[i];
        const auto beginI = param.beginList[i];
        const auto endI = param.endList[i];
        const auto stride_i = endI > beginI ? std::min(param.strideList[i], endI - beginI) : param.strideList[i];
        if (i == 0 || inputShapeI != outputShapeI || stride_i != 1 || perfParams.strideList[perfSize - 1] != 1 ||
            (!param.isBeginConst && (beginI < 0 || (i > 0 && param.beginList[i - 1] < 0)))) {
            int64_t realBeginValue = (!param.isBeginConst && beginI < 0) ?
                                         (UNCONST_BEGIN_VALUE - static_cast<int64_t>(i)) :
                                         beginI;
            perfParams.inputShape.AppendDim(inputShapeI);
            perfParams.outputShape.AppendDim(outputShapeI);
            perfParams.beginList.AppendDim(realBeginValue);
            perfParams.endList.AppendDim(endI);
            perfParams.strideList.AppendDim(stride_i);
            perfSize++;
            continue;
        }

        const auto perfIndex = perfSize - static_cast<size_t>(1);
        perfParams.inputShape[perfIndex] *= inputShapeI;
        perfParams.outputShape[perfIndex] *= outputShapeI;
        perfParams.beginList[perfIndex] *= inputShapeI;
        perfParams.endList[perfIndex] *= inputShapeI;
        perfParams.strideList[perfIndex] = 1;
    }

    param = perfParams;

    if (hasNegStride) {
        MakePerformanceParamsNeg(param);
    }
}

static void MakeSameDims(SliceParametersRuntime2* parametersPtr)
{
    auto& parameters = *parametersPtr;
    bool sameSize = parameters.inputShape.GetDimNum() == parameters.beginList.GetDimNum() &&
                    parameters.inputShape.GetDimNum() == parameters.endList.GetDimNum() &&
                    parameters.inputShape.GetDimNum() == parameters.strideList.GetDimNum();
    if (!sameSize) {
        return;
    }

    parameters.outputShape.SetDimNum(0);
    for (size_t i = 0; i < parameters.inputShape.GetDimNum(); i++) {
        auto interval = parameters.endList[i] - parameters.beginList[i];
        auto strideI = parameters.strideList[i];
        if (strideI == 0) {
            strideI = 1;
        }
        int64_t outputSize = interval / strideI + (interval % strideI != 0 ? 1 : 0);
        parameters.outputShape.AppendDim(outputSize);
    }
}

static ge::graphStatus CheckInputForFp4(gert::TilingContext* context, SliceParametersRuntime2& sliceParam)
{
    int32_t inDimNum = sliceParam.inputShape.GetDimNum();
    int32_t outDimNum = sliceParam.outputShape.GetDimNum();
    if (inDimNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "dim num of input", std::to_string(inDimNum),
                                               "Dim num of input must be greater than 0 for fp4 input.");
        return ge::GRAPH_FAILED;
    }
    if (outDimNum < inDimNum || static_cast<int32_t>(sliceParam.beginList.GetDimNum()) < inDimNum ||
        static_cast<int32_t>(sliceParam.endList.GetDimNum()) < inDimNum ||
        static_cast<int32_t>(sliceParam.strideList.GetDimNum()) < inDimNum) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "dim num of output, begin, end, strides",
            std::to_string(outDimNum) + ", " + std::to_string(sliceParam.beginList.GetDimNum()) + ", " +
                std::to_string(sliceParam.endList.GetDimNum()) + ", " +
                std::to_string(sliceParam.strideList.GetDimNum()),
            "Dim num of output/begin/end/strides must be greater than or equal to that of input for fp4 input.");
        return ge::GRAPH_FAILED;
    }

    if (sliceParam.inputShape[inDimNum - 1] & 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "last dimension of input",
                                               Ops::Base::ToString(sliceParam.inputShape).c_str(),
                                               "Expected last dimension of input to be even for fp4 input.");
        return ge::GRAPH_FAILED;
    }
    sliceParam.inputShape[inDimNum - 1] /= NUMBER_TWO;

    if (sliceParam.outputShape[outDimNum - 1] & 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "last dimension of output",
                                               Ops::Base::ToString(sliceParam.outputShape).c_str(),
                                               "Expected last dimension of output to be even for fp4 input.");
        return ge::GRAPH_FAILED;
    }
    sliceParam.outputShape[outDimNum - 1] /= NUMBER_TWO;

    if (sliceParam.beginList[inDimNum - 1] & 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "last dimension of begin",
                                               Ops::Base::ToString(sliceParam.beginList).c_str(),
                                               "Expected last dimension of begin to be even for fp4 input.");
        return ge::GRAPH_FAILED;
    }
    sliceParam.beginList[inDimNum - 1] /= NUMBER_TWO;

    if (sliceParam.endList[inDimNum - 1] & 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "last dimension of end",
                                               Ops::Base::ToString(sliceParam.endList).c_str(),
                                               "Expected last dimension of end to be even for fp4 input.");
        return ge::GRAPH_FAILED;
    }
    sliceParam.endList[inDimNum - 1] /= NUMBER_TWO;

    if (sliceParam.strideList[inDimNum - 1] != 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "last dimension of strides",
                                               Ops::Base::ToString(sliceParam.strideList).c_str(),
                                               "Expected last dimension of strides to be 1 for fp4 input.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StrideSliceTilingForAscendC(gert::TilingContext* tilingContext, int64_t coreNum, int64_t ubSize,
                                            int64_t cachelineSize, SliceParametersRuntime2& sliceParam,
                                            const ge::DataType& dtype)
{
    StrideSliceTiling tilingObject(tilingContext);
    if (tilingObject.Init(coreNum, ubSize, cachelineSize, sliceParam, dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunStrideSliceTiling();
}

ge::graphStatus TilingPrepare4StridedSlice(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init TransposeNddmaTiling.");
    auto ci = context->GetCompiledInfo<StridedSliceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    ci->isAscendc = true;
    OP_CHECK_IF((ci->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core number."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    ci->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF((ci->cacheLineSize == 0), OP_LOGE(context->GetNodeName(), "Failed to get cacheLineSize."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4StridedSlice(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const StridedSliceCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "compile info: %s.", compileInfo->to_string().c_str());

    // construct slice param
    SliceParametersRuntime2 sliceParam;
    if (ConstructSliceParam(context, sliceParam, compileInfo->isAscendc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // get mask attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* maskBegin = attrs->GetAttrPointer<int64_t>(IDX_MASK_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskBegin);
    const int64_t* maskEnd = attrs->GetAttrPointer<int64_t>(IDX_MASK_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskEnd);
    const int64_t* maskEllipsis = attrs->GetAttrPointer<int64_t>(IDX_MASK_ELLIPSIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskEllipsis);
    const int64_t* maskNewAxis = attrs->GetAttrPointer<int64_t>(IDX_MASK_NEW_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskNewAxis);
    const int64_t* maskShrinkAxis = attrs->GetAttrPointer<int64_t>(IDX_MASK_SHRINK_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShrinkAxis);

    if (ProcBeginEndUnconst(context, sliceParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // infer shape
    const gert::StorageShape* xStorage = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::Shape& shapeInput = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    ops::StridedSliceParams inputParams = {shapeInput,
                                           sliceParam.beginList,
                                           sliceParam.endList,
                                           sliceParam.strideList,
                                           static_cast<uint64_t>(*maskBegin),
                                           static_cast<uint64_t>(*maskEnd),
                                           static_cast<uint64_t>(*maskEllipsis),
                                           static_cast<uint64_t>(*maskNewAxis),
                                           static_cast<uint64_t>(*maskShrinkAxis),
                                           true,
                                           true,
                                           true,
                                           sliceParam.isBeginConst,
                                           sliceParam.isEndConst,
                                           shapeInput};
    gert::Shape shapeOutput;
    if (!ops::InferShape(inputParams, &shapeOutput)) {
        return ge::GRAPH_FAILED;
    }

    // reconstruct slice_param by infer shape
    ReconstructSliceParamByInferShape(inputParams, shapeOutput, sliceParam);

    // align slice param dims.
    MakeSameDims(&sliceParam);
    OP_LOGI(context->GetNodeName(), "align slice params: %s", sliceParam.to_string().c_str());

    // infer tiling mode
    auto xDesc = context->GetInputDesc(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType inputDtype = xDesc->GetDataType();

    // For fp4 input, reinterpret as int8 by halving the last packed dim.
    // Must be done before MakePerformanceParams, otherwise perf may merge/reverse
    // the packed dim and strideList, which would break the halving logic.
    if (inputDtype == ge::DT_FLOAT4_E2M1 || inputDtype == ge::DT_FLOAT4_E1M2 || inputDtype == Ops::Math::DT_HIFLOAT4) {
        if (!compileInfo->isAscendc) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputDtype).c_str(),
                                                  "fp4 input is only supported on ascendc path.");
            return ge::GRAPH_FAILED;
        }
        if (CheckInputForFp4(context, sliceParam) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        inputDtype = ge::DT_INT8;
    }

    // optimize formance slice param
    MakePerformanceParams(sliceParam, true);
    OP_LOGI(context->GetNodeName(), "perf slice params: %s", sliceParam.to_string().c_str());

    if (compileInfo->isAscendc) {
        return StrideSliceTilingForAscendC(context, compileInfo->coreNum, compileInfo->ubSize,
                                           compileInfo->cacheLineSize, sliceParam, inputDtype);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StridedSlice)
    .Tiling(Tiling4StridedSlice)
    .TilingParse<StridedSliceCompileInfo>(TilingPrepare4StridedSlice);

} // namespace optiling
