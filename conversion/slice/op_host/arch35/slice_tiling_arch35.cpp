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
 * \file slice_tiling_arch35.cpp
 * \brief
 */
#include "slice_tiling_arch35.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {

void SliceTiling::CalMaxSplitDim()
{
    // 受movealignv2的参数限制,blockcout:uint16, srcstride:uint32
    // 输入维度越高，索引生产的scalar越重
    bool isUbinnerGather = lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_ >= MIN_MOVE_ALIGN_LEN &&
                           dimNum_ <= NUMBER_FOUR &&
                           (lastTwoInputDim_ - lastTwoOutputDim_) * lastOneInputDim_ * xDtypeSize_ <= MAX_UINT32_NUM &&
                           lastOneInputDim_ * lastTwoInputDim_ * lastThreeInputDim_ * xDtypeSize_ <= MAX_UINT32_NUM;
    if (dimNum_ == NUMBER_TWO && lastOneOutputDim_ * xDtypeSize_ < VL_SIZE && lastTwoOutputDim_ == lastTwoInputDim_ &&
        VL_SIZE * lastTwoInputDim_ < coreNum_ * ubSize_ / NUMBER_FOUR &&
        (lastOneOutputDim_ != 1 || lastTwoOutputDim_ <= LAST_DIM_MIN_DATA_SIZE)) {
        maxSplitDim_ = MAX_TWO_DIM_UB_SPLIT_AXIS_NUM;
    } else if (
        (lastOneInputDim_ * xDtypeSize_) % BLOCK_SIZE != 0 && lastOneInputDim_ * xDtypeSize_ <= VL_SIZE &&
        (lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_ >= VL_SIZE ||
         lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_ % BLOCK_SIZE == 0) &&
        lastOneOutputDim_ * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE && totalOutputSize_ > MIN_OUTPUT_SIZE &&
        lastOneInputDim_ / lastOneOutputDim_ < DATA_COPY_SPARSITY_THRESHOLD && isUbinnerGather) {
        // 输入的尾轴非block对齐
        // 输入尾轴<= 256B && 64B <= 输出尾轴 <= 256B
        // 当连续搬入的burstlen小于256B且非block对齐时，性能比compact模式更差
        isUnalignCopy_ = true;
        maxSplitDim_ = MAX_SLICE_GATHER_UB_SPLIT_AXIS_NUM;
        isSliceGather_ = true;
    } else if (lastOneOutputDim_ * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
    } else if (
        lastOneInputDim_ * xDtypeSize_ < RESERVE_LAST_DIM_SIZE &&
        lastOneInputDim_ / lastOneOutputDim_ < DATA_SPARSITY_THRESHOLD &&
        totalOutputSize_ / lastOneOutputDim_ * lastOneInputDim_ >
            ubSize_ * lastOneInputDim_ / (lastOneOutputDim_ + lastOneInputDim_) * coreNum_ &&
        xDtypeSize_ != 1 && isUbinnerGather) {
        // b8 性能不稳定
        // 数据稀疏度越低，搬入的有效数据越少
        // 数据太小，一次搬入就处理完，vf计算完全不能被MTE掩掉的场景
        // 上述2个条件保证了totalOutputSize_ > LAST_DIM_MIN_DATA_SIZE
        maxSplitDim_ = MAX_SLICE_GATHER_UB_SPLIT_AXIS_NUM;
        isSliceGather_ = true;
    } else {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
    }
}

void SliceTiling::SetTilingMode()
{
    if (maxSplitDim_ == MAX_TWO_DIM_UB_SPLIT_AXIS_NUM) {
        SetTwoDimSmallShapeTilingMode();
    } else if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM) {
        SetMovAlignV2TilingMode();
    } else if (isSliceGather_) {
        SetSliceGatherTilingMode();
    } else {
        SetNddmaTilingMode();
    }
}

void SliceTiling::SetTwoDimSmallShapeTilingMode()
{
    tilingKey_ = SLICE_KEY_TWO_DIM_SMALL_SHAPE;
}

/*
input
(A, B, C, D，E)
outputShape:
(a, b, c, d，e)

(a, b, c, d，E)
搬入d*E的数据
场景1：ub切d
(a, b, c, d.o, d.i, E)
burstlen = d.i * E * dype
nburst = 1

场景2：ub切c
(a, b, c.o, c.i, d，E)
burstlen = d * E * dype
nburst = c.i

场景3：ub切b
(a, b.o, b.i, c, d，E)
burstlen = d * E * dype
nburst = c
loop1size = b.i

场景4：ub切a
(a.o, a.i, b, c, d，E)
burstlen = d * E * dype
nburst = c
loop1size = b
loop2size = a.i

维度太高，scalar计算也会很重，可以考虑限制, inputdim <= 4

*/
void SliceTiling::SetSliceGatherTilingMode()
{
    int32_t shapeSize = dimNum_;
    // SliceGather场景的准入条件是尾轴小于64B，保证了尾轴不会被切UB
    if (isUnalignCopy_) {
        tilingKey_ = SLICE_KEY_MOVE_UNALIGN_GATHER;
    } else {
        tilingKey_ = STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER;
    }
    // ubfactor切-2轴
    if (ubIndex_ == shapeSize - NUMBER_TWO) {
        SliceGatherUbSplitLastTwoDim();
        return;
    }

    // ubfactor切-3轴
    if (ubIndex_ == shapeSize - NUMBER_THREE) {
        SliceGatherUbSplitLastThreeDim();
        return;
    }

    // ubfactor切-4轴
    if (ubIndex_ == shapeSize - NUMBER_FOUR) {
        SliceGatherUbSplitLastFourDim();
        return;
    }
}

/*
场景1：ub切d
(a, b, c, d.o, d.i, E)
burstlen = d.i * E * dype
nburst = 1
*/
void SliceTiling::SliceGatherUbSplitLastTwoDim()
{
    mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(1);
    mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(ubFactor_ * lastOneInputDim_ * xDtypeSize_);
    mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(0);
    mainMoveAlignV2Info_.dstStride = static_cast<uint32_t>(0);
    outBlockLen_ = static_cast<uint32_t>(ubFactor_ * lastOneOutputDim_ * xDtypeSize_);
}

/*
场景2：ub切c
(a, b, c.o, c.i, d，E)
burstlen = d * E * dype
nburst = c.i
*/
void SliceTiling::SliceGatherUbSplitLastThreeDim()
{
    mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(ubFactor_);
    mainMoveAlignV2Info_.blockLen =
        static_cast<uint32_t>(lastTwoOutputDim_ * lastOneInputDim_ * xDtypeSize_); // 每个blockLen之间会补pad
    mainMoveAlignV2Info_.srcStride =
        static_cast<uint32_t>((lastTwoInputDim_ - lastTwoOutputDim_) * lastOneInputDim_ * xDtypeSize_);
    mainMoveAlignV2Info_.dstStride = static_cast<uint32_t>(0);

    // out
    outBlockLen_ = static_cast<uint32_t>(lastTwoOutputDim_ * lastOneOutputDim_ * xDtypeSize_);
}

/*
场景3：ub切b
(a, b.o, b.i, c, d，E)
burstlen = d * E * dype
nburst = c
loop1size = b.i
*/
void SliceTiling::SliceGatherUbSplitLastFourDim()
{
    mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastThreeOutputDim_);
    mainMoveAlignV2Info_.blockLen =
        static_cast<uint32_t>(lastTwoOutputDim_ * lastOneInputDim_ * xDtypeSize_); // 每个blockLen之间会补pad
    mainMoveAlignV2Info_.srcStride =
        static_cast<uint32_t>((lastTwoInputDim_ - lastTwoOutputDim_) * lastOneInputDim_ * xDtypeSize_);
    mainMoveAlignV2Info_.dstStride = static_cast<uint32_t>(0);
    outBlockLen_ = static_cast<uint32_t>(lastTwoOutputDim_ * lastOneOutputDim_ * xDtypeSize_);

    int64_t loop1SrcStride = lastOneInputDim_ * lastTwoInputDim_ * lastThreeInputDim_ * xDtypeSize_;
    int64_t loop1DstStride =
        Ops::Base::CeilAlign(lastOneInputDim_ * lastTwoOutputDim_ * xDtypeSize_, BLOCK_SIZE) * lastThreeOutputDim_;
    mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(ubFactor_);
    mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride); // repeat stride是头和头之间的间隔
    mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
}

void SliceTiling::CalSliceRowsStepsParams()
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - static_cast<int32_t>(NUMBER_TWO); i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    // ub非最后一根轴
    if (ubIndex_ != static_cast<int64_t>(dimNum_) - static_cast<int64_t>(1)) {
        inLoopSteps_ = ubFactor_ * inputSteps_[ubIndex_ + 1];
        outLoopSteps_ = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps_ = ubFactor_ * strides_[ubIndex_];
        outLoopSteps_ = ubFactor_;
    }
}

void SliceTiling::SetRowsStepsParamsFor150(SliceMoveAlignLast2DimTilingData& tilingData)
{
    CalSliceRowsStepsParams();
    tilingData.set_ubInLoopSteps(inLoopSteps_);
    tilingData.set_ubOutLoopSteps(outLoopSteps_);
    int64_t inputSteps[NUMBER_TWO];
    for (size_t i = 0; i < static_cast<size_t>(NUMBER_TWO); i++) {
        inputSteps[i] = inputSteps_[i];
    }
    tilingData.set_inputSteps(inputSteps);
}

void SliceTiling::SetShortMoveAlignParams(SliceMoveAlignParams& params, const MoveAlignV2Info& actInfo)
{
    params.set_blockCount(actInfo.blockCount);
    params.set_blockLen(actInfo.blockLen);
    params.set_srcStride(actInfo.srcStride);
    params.set_dstStride(actInfo.dstStride);
}

void SliceTiling::FillSliceTilingData150()
{
    int64_t begin[NUMBER_TWO];
    int64_t outputShape[NUMBER_TWO];
    for (size_t i = 0; i < static_cast<size_t>(NUMBER_TWO); i++) {
        begin[i] = begin_[i];
        outputShape[i] = outputShape_[i];
    }
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData150.");
    sliceMoveAlignLast2DimTilingData_.set_ubSize(ubSize_);
    sliceMoveAlignLast2DimTilingData_.set_blkFactor(blkFactor_);
    sliceMoveAlignLast2DimTilingData_.set_blkTailFactor(blkTailFactor_);
    sliceMoveAlignLast2DimTilingData_.set_ubFactor(ubFactor_);
    sliceMoveAlignLast2DimTilingData_.set_ubTailFactor(ubTailFactor_);
    sliceMoveAlignLast2DimTilingData_.set_ubTailTailFactor(ubTailTailFactor_);
    sliceMoveAlignLast2DimTilingData_.set_realCoreNum(realCoreNum_);
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    sliceMoveAlignLast2DimTilingData_.set_isBeginConst(isConstBegin);

    SetShortMoveAlignParams(sliceMoveAlignLast2DimTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    sliceMoveAlignLast2DimTilingData_.set_begin(begin);
    sliceMoveAlignLast2DimTilingData_.set_outputShape(outputShape);
    SetRowsStepsParamsFor150(sliceMoveAlignLast2DimTilingData_);
    sliceMoveAlignLast2DimTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceMoveAlignLast2DimTilingData_.GetDataSize());
}

void SliceTiling::FillSliceBaseTilingData(SliceBaseTilingData& tilingData)
{
    tilingData.set_ubSize(ubSize_);
    tilingData.set_ubIndex(ubIndex_);
    tilingData.set_ubFactor(ubFactor_);
    tilingData.set_ubTailFactor(ubTailFactor_);
    tilingData.set_ubTailTailFactor(ubTailTailFactor_);
    tilingData.set_realCoreNum(realCoreNum_);
    tilingData.set_inputDims(dimNum_);
    tilingData.set_blkIndex(blkIndex_);
    tilingData.set_blkFactor(blkFactor_);
    tilingData.set_blkTailFactor(blkTailFactor_);
    tilingData.set_begin(begin_);
    tilingData.set_outputShape(outputShape_);
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    tilingData.set_isBeginConst(isConstBegin);

    CalSliceRowsStepsParams();
    tilingData.set_ubInLoopSteps(inLoopSteps_);
    tilingData.set_inputSteps(inputSteps_);
    tilingData.set_rowsOffsetSteps(rowsOffsetSteps_);
}

void SliceTiling::FillSliceTilingData100()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData100.");
    FillSliceBaseTilingData(sliceMoveAlignTilingData_.sliceBaseTilingData);

    sliceMoveAlignTilingData_.set_ubOutLoopSteps(outLoopSteps_);
    SetMoveAlignParams(sliceMoveAlignTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    sliceMoveAlignTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceMoveAlignTilingData_.GetDataSize());
}

void SliceTiling::FillSliceTilingData101()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData101.");
    FillSliceBaseTilingData(sliceMoveAlignLastDimTilingData_.sliceBaseTilingData);
    sliceMoveAlignLastDimTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceMoveAlignLastDimTilingData_.GetDataSize());
}

void SliceTiling::FillSliceTilingData102()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData102.");
    FillSliceBaseTilingData(sliceNDDMATilingData_.sliceBaseTilingData);

    sliceNDDMATilingData_.set_ubOutLoopSteps(outLoopSteps_);
    sliceNDDMATilingData_.set_nddmaTotalNum(nddmaTotalNum_);
    sliceNDDMATilingData_.set_nddmaLoopSize(nddmaLoopSize_);
    sliceNDDMATilingData_.set_nddmaLoopSrcStride(nddmaLoopSrcStride_);
    sliceNDDMATilingData_.set_nddmaLoopDstStride(nddmaLoopDstStride_);
    sliceNDDMATilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceNDDMATilingData_.GetDataSize());
}

void SliceTiling::FillSliceTilingData103()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData103.");
    FillSliceBaseTilingData(sliceNDDMALastDimTilingData_.sliceBaseTilingData);

    sliceNDDMALastDimTilingData_.set_nddmaLoopSrcStride(nddmaLoopSrcStride_);
    sliceNDDMALastDimTilingData_.set_nddmaLoopDstStride(nddmaLoopDstStride_);
    sliceNDDMALastDimTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceNDDMALastDimTilingData_.GetDataSize());
}

void SliceTiling::FillSliceTilingData300()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData300.");
    FillSliceBaseTilingData(sliceMoveAlignGatherTilingData_.sliceBaseTilingData);
    sliceMoveAlignGatherTilingData_.sliceBaseTilingData.set_ubSize(ubSizeOutput_);
    sliceMoveAlignGatherTilingData_.set_ubSizeInput(ubSizeInput_);
    sliceMoveAlignGatherTilingData_.set_lastOneInputDim(lastOneInputDim_);
    sliceMoveAlignGatherTilingData_.set_outBlockLen(outBlockLen_);
    sliceMoveAlignGatherTilingData_.set_ubOutLoopSteps(outLoopSteps_);

    SetMoveAlignParams(sliceMoveAlignGatherTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    sliceMoveAlignGatherTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceMoveAlignGatherTilingData_.GetDataSize());
}

void SliceTiling::FillSliceTilingData400()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData400.");
    sliceTwoDimSmallSapeTilingData_.set_ubSize((uint32_t)ubSize_);
    sliceTwoDimSmallSapeTilingData_.set_realCoreNum(realCoreNum_);
    sliceTwoDimSmallSapeTilingData_.set_mainCoreNum(mainCoreNum_);
    sliceTwoDimSmallSapeTilingData_.set_blockLen(outBlockLen_);
    sliceTwoDimSmallSapeTilingData_.set_blkFactor(blkFactor_);
    sliceTwoDimSmallSapeTilingData_.set_lastOneInputDim(lastOneInputDim_);
    sliceTwoDimSmallSapeTilingData_.set_lastOneOutputDim(lastOneOutputDim_);
    sliceTwoDimSmallSapeTilingData_.set_lastOneDimOffset(begin_[1]);
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    sliceTwoDimSmallSapeTilingData_.set_isBeginConst(isConstBegin);

    sliceTwoDimSmallSapeTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceTwoDimSmallSapeTilingData_.GetDataSize());
}

void SliceTiling::FillSliceTilingDataOther()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData.");
    sliceTilingData_.stridedSliceTilingData.set_ubSize(ubSize_);
    sliceTilingData_.stridedSliceTilingData.set_coreNum(coreNum_);
    sliceTilingData_.stridedSliceTilingData.set_ubIndex(ubIndex_);
    sliceTilingData_.stridedSliceTilingData.set_ubFactor(ubFactor_);
    sliceTilingData_.stridedSliceTilingData.set_ubTailFactor(ubTailFactor_);
    sliceTilingData_.stridedSliceTilingData.set_ubTailTailFactor(ubTailTailFactor_);
    sliceTilingData_.stridedSliceTilingData.set_realCoreNum(realCoreNum_);
    sliceTilingData_.stridedSliceTilingData.set_inputDims(dimNum_);
    sliceTilingData_.stridedSliceTilingData.set_blkIndex(blkIndex_);
    sliceTilingData_.stridedSliceTilingData.set_blkFactor(blkFactor_);
    sliceTilingData_.stridedSliceTilingData.set_blkTailFactor(blkTailFactor_);
    sliceTilingData_.stridedSliceTilingData.set_xDtypeSize(xDtypeSize_);
    sliceTilingData_.stridedSliceTilingData.set_tilingKey(tilingKey_);
    sliceTilingData_.stridedSliceTilingData.set_nddmaTotalNum(nddmaTotalNum_);
    sliceTilingData_.stridedSliceTilingData.set_nddmaLoopSize(nddmaLoopSize_);
    sliceTilingData_.stridedSliceTilingData.set_nddmaLoopSrcStride(nddmaLoopSrcStride_);
    sliceTilingData_.stridedSliceTilingData.set_nddmaLoopDstStride(nddmaLoopDstStride_);
    sliceTilingData_.stridedSliceTilingData.set_outputShapeProd(outputShapeProd_);
    sliceTilingData_.stridedSliceTilingData.set_inputShapeProd(inputShapeProd_);
    int8_t isConstBegin = sliceParam_.isBeginConst ? 1 : 0;
    sliceTilingData_.stridedSliceTilingData.set_isBeginConst(isConstBegin);

    SetMoveAlignParams(sliceTilingData_.stridedSliceTilingData.moveAlignParams, mainMoveAlignV2Info_);

    for (size_t i = 0; i < sliceParam_.inputShape.GetDimNum(); i++) {
        begin_[i] = sliceParam_.beginList.GetDim(i);
        end_[i] = sliceParam_.endList.GetDim(i);
        strides_[i] = sliceParam_.strideList.GetDim(i);
        inputShape_[i] = sliceParam_.inputShape.GetDim(i);
        outputShape_[i] = sliceParam_.outputShape.GetDim(i);
    }
    sliceTilingData_.stridedSliceTilingData.set_begin(begin_);
    sliceTilingData_.stridedSliceTilingData.set_strides(strides_);
    sliceTilingData_.stridedSliceTilingData.set_outputShape(outputShape_);

    SetRowsStepsParams(sliceTilingData_.stridedSliceTilingData);
    sliceTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(sliceTilingData_.GetDataSize());
}

void SliceTiling::FillTilingData()
{
    for (size_t i = 0; i < sliceParam_.inputShape.GetDimNum(); i++) {
        begin_[i] = sliceParam_.beginList.GetDim(i);
        end_[i] = sliceParam_.endList.GetDim(i);
        inputShape_[i] = sliceParam_.inputShape.GetDim(i);
        outputShape_[i] = sliceParam_.outputShape.GetDim(i);
    }
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            FillSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            FillSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
            FillSliceTilingData102();
            break;
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
            FillSliceTilingData103();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            FillSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
        case SLICE_KEY_MOVE_UNALIGN_GATHER:
            FillSliceTilingData300();
            break;
        case SLICE_KEY_TWO_DIM_SMALL_SHAPE:
            FillSliceTilingData400();
            break;
        default:
            FillSliceTilingDataOther();
            break;
    }
}

void SliceTiling::PrintSliceTilingDataOther()
{
    StridedSliceTilingData& tilingData = sliceTilingData_.stridedSliceTilingData;
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is ubSize:%ld, coreNum:%ld, realCoreNum:%ld, \
            ubIndex:%ld, ubFactor:%ld, ubTailFactor:%ld, ubTailTailFactor:%ld, \
            blkIndex:%ld, blkFactor:%ld, blkTailFactor:%ld, xDtypeSize:%ld, tilingKey:%ld, \
            isBeginConst:%d, begin:%s, end:%s, stride:%s, inputShape:%s, outputShape:%s, \
            rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s",
        tilingData.get_ubSize(), tilingData.get_coreNum(), tilingData.get_realCoreNum(), tilingData.get_ubIndex(),
        tilingData.get_ubFactor(), tilingData.get_ubTailFactor(), tilingData.get_ubTailTailFactor(),
        tilingData.get_blkIndex(), tilingData.get_blkFactor(), tilingData.get_blkTailFactor(),
        tilingData.get_xDtypeSize(), tilingData.get_tilingKey(), tilingData.get_isBeginConst(),
        ArrayToStr(tilingData.get_begin(), dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(tilingData.get_strides(), dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(tilingData.get_outputShape(), dimNum_).c_str(),
        ArrayToStr(tilingData.get_rowsOffsetSteps(), dimNum_).c_str(),
        ArrayToStr(tilingData.get_inputSteps(), dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str());

    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is nddmaTotalNum:%ld nddmaLoopSize:%s, nddmaLoopSrcStride: %s, \
            nddmaLoopDstStride: %s, moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
            loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u outputShapeProd: %s \
            inputShapeProd: %s Tiling4StrideSlice ends.",
        tilingData.get_nddmaTotalNum(), ArrayToStr(tilingData.get_nddmaLoopSize(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData.get_nddmaLoopSrcStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData.get_nddmaLoopDstStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
        mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
        mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride, mainMoveAlignV2Info_.loop2SrcStride,
        mainMoveAlignV2Info_.loop2DstStride,
        ArrayToStr(tilingData.get_outputShapeProd(), MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData.get_inputShapeProd(), MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str());
}

void SliceTiling::PrintTilingData()
{
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            PrintSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            PrintSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
            PrintSliceTilingData102();
            break;
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
            PrintSliceTilingData103();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            PrintSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
            PrintSliceTilingData300();
            break;
        case SLICE_KEY_TWO_DIM_SMALL_SHAPE:
            PrintSliceTilingData400();
            break;
        default:
            PrintSliceTilingDataOther();
            break;
    }
}

void SliceTiling::PrintSliceBaseTilingData(SliceBaseTilingData& tilingData)
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceBaseTilingData is ubSize:%d, realCoreNum:%d, \
            ubIndex:%d, ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkIndex:%d, blkFactor:%ld, blkTailFactor:%ld, \
            isBeginConst:%d, begin:%s, end:%s, inputShape:%s, outputShape:%s, \
            rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld",
        tilingData.get_ubSize(), tilingData.get_realCoreNum(), tilingData.get_ubIndex(), tilingData.get_ubFactor(),
        tilingData.get_ubTailFactor(), tilingData.get_ubTailTailFactor(), tilingData.get_blkIndex(),
        tilingData.get_blkFactor(), tilingData.get_blkTailFactor(), tilingData.get_isBeginConst(),
        ArrayToStr(tilingData.get_begin(), dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(inputShape_, dimNum_).c_str(), ArrayToStr(tilingData.get_outputShape(), dimNum_).c_str(),
        ArrayToStr(tilingData.get_rowsOffsetSteps(), dimNum_).c_str(),
        ArrayToStr(tilingData.get_inputSteps(), dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str(),
        tilingData.get_ubInLoopSteps());
}

void SliceTiling::PrintSliceTilingData150()
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceMoveAlignLast2DimTilingData is ubSize:%d, realCoreNum:%d, \
            ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkFactor:%ld, blkTailFactor:%ld, \
            isBeginConst:%d, begin:%s, end:%s, inputShape:%s, outputShape:%s, \
            inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld, ubOutLoopSteps:%ld, \
            moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u",
        sliceMoveAlignLast2DimTilingData_.get_ubSize(), sliceMoveAlignLast2DimTilingData_.get_realCoreNum(),
        sliceMoveAlignLast2DimTilingData_.get_ubFactor(), sliceMoveAlignLast2DimTilingData_.get_ubTailFactor(),
        sliceMoveAlignLast2DimTilingData_.get_ubTailTailFactor(), sliceMoveAlignLast2DimTilingData_.get_blkFactor(),
        sliceMoveAlignLast2DimTilingData_.get_blkTailFactor(), sliceMoveAlignLast2DimTilingData_.get_isBeginConst(),
        ArrayToStr(sliceMoveAlignLast2DimTilingData_.get_begin(), dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(sliceMoveAlignLast2DimTilingData_.get_outputShape(), dimNum_).c_str(),
        ArrayToStr(sliceMoveAlignLast2DimTilingData_.get_inputSteps(), dimNum_).c_str(),
        ArrayToStr(outputSteps_, dimNum_).c_str(), sliceMoveAlignLast2DimTilingData_.get_ubInLoopSteps(),
        sliceMoveAlignLast2DimTilingData_.get_ubOutLoopSteps(), mainMoveAlignV2Info_.blockCount,
        mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride);
}

void SliceTiling::PrintSliceTilingData101()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceMoveAlignLastDimTilingData:");
    PrintSliceBaseTilingData(sliceMoveAlignLastDimTilingData_.sliceBaseTilingData);
}

void SliceTiling::PrintSliceTilingData100()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceMoveAlignTilingData:");
    PrintSliceBaseTilingData(sliceMoveAlignTilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceMoveAlignTilingData is ubOutLoopSteps:%ld, \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        sliceMoveAlignTilingData_.get_ubOutLoopSteps(), mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen,
        mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size,
        mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
        mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void SliceTiling::PrintSliceTilingData102()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceNDDMATilingData:");
    PrintSliceBaseTilingData(sliceNDDMATilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceNDDMATilingData is ubOutLoopSteps:%ld, \
    nddmaTotalNum:%ld, nddmaLoopSize:%s, nddmaLoopSrcStride: %s, nddmaLoopDstStride: %s",
        sliceNDDMATilingData_.get_ubOutLoopSteps(), sliceNDDMATilingData_.get_nddmaTotalNum(),
        ArrayToStr(sliceNDDMATilingData_.get_nddmaLoopSize(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(sliceNDDMATilingData_.get_nddmaLoopSrcStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(sliceNDDMATilingData_.get_nddmaLoopDstStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str());
}

void SliceTiling::PrintSliceTilingData103()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceNDDMALastDimTilingData:");
    PrintSliceBaseTilingData(sliceNDDMALastDimTilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(), "SliceNDDMALastDimTilingData is nddmaLoopSrcStride: %s, nddmaLoopDstStride: %s",
        ArrayToStr(sliceNDDMALastDimTilingData_.get_nddmaLoopSrcStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(sliceNDDMALastDimTilingData_.get_nddmaLoopDstStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str());
}

void SliceTiling::PrintSliceTilingData300()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceMoveAlignGatherTilingData:");
    PrintSliceBaseTilingData(sliceMoveAlignGatherTilingData_.sliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceMoveAlignGatherTilingData is ubOutLoopSteps:%ld, \
    ubSizeInput:%d, lastOneInputDim:%u, outBlockLen:%u, \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        sliceMoveAlignGatherTilingData_.get_ubOutLoopSteps(), sliceMoveAlignGatherTilingData_.get_ubSizeInput(),
        sliceMoveAlignGatherTilingData_.get_lastOneInputDim(), sliceMoveAlignGatherTilingData_.get_outBlockLen(),
        mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
        mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
        mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride, mainMoveAlignV2Info_.loop2SrcStride,
        mainMoveAlignV2Info_.loop2DstStride);
}

void SliceTiling::PrintSliceTilingData400()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing SliceTwoDimSmallSapeTilingData:");
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "SliceTwoDimSmallSapeTilingData is realCoreNum:%d, \
    mainCoreNum:%d, outBlockLen:%d, blkFactor:%d, lastOneInputDim:%ld, lastOneOutputDim:%ld, ubSize:%d, \
    lastOneDimOffset:%ld, isBeginConst:%d",
        sliceTwoDimSmallSapeTilingData_.get_realCoreNum(), sliceTwoDimSmallSapeTilingData_.get_mainCoreNum(),
        sliceTwoDimSmallSapeTilingData_.get_blockLen(), sliceTwoDimSmallSapeTilingData_.get_blkFactor(),
        sliceTwoDimSmallSapeTilingData_.get_lastOneInputDim(), sliceTwoDimSmallSapeTilingData_.get_lastOneOutputDim(),
        sliceTwoDimSmallSapeTilingData_.get_ubSize(), sliceTwoDimSmallSapeTilingData_.get_lastOneDimOffset(),
        sliceTwoDimSmallSapeTilingData_.get_isBeginConst());
}

void SliceTiling::SetBlockDimAndTilingKey()
{
    tilingContext_->SetBlockDim(realCoreNum_);
    tilingContext_->SetTilingKey(tilingKey_);
}

ge::graphStatus SliceTilingForAscendC(
    gert::TilingContext* context, int64_t coreNum, int64_t ubSize, int64_t cacheLineSize, SliceParasRuntime2& param,
    const ge::DataType dtype)
{
    OP_LOGD(context->GetNodeName(), "Enter SliceTilingForAscendC.");

    SliceTiling tilingImpl(context);
    SliceParametersRuntime2 sliceParam;
    sliceParam.inputShape = param.input;
    sliceParam.outputShape = param.output_shape;
    sliceParam.beginList = param.begin_list;
    sliceParam.endList = param.end_list;
    sliceParam.strideList = param.stride_list;
    sliceParam.tilingMode = param.tiling_mode;
    sliceParam.coreNum = param.core_num;
    sliceParam.isBeginConst = param.is_begin_const;
    if (tilingImpl.Init(coreNum, ubSize, cacheLineSize, sliceParam, dtype) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "SliceTilingForAscendC init failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingImpl.RunStrideSliceTiling();
}

template <typename T>
static ge::graphStatus AssignInputValueOpt(
    gert::TilingContext* context, size_t size, const gert::Tensor* tensor, gert::Shape& list_vector,
    bool isAscendc, bool& isConst)
{
    list_vector.SetDimNum(size);
    const T* value = tensor->GetData<T>();
    if (value == nullptr && isAscendc) {
        OP_LOGI(context->GetNodeName(), "ascendc get const data nullptr");
        isConst = false;
        for (size_t i = 0; i < size; i++) {
            list_vector[i] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        value == nullptr,
        OP_LOGE(context->GetNodeName(), "get const value fail, check input is const or not."),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < size; i++) {
        list_vector[i] = value[i];
    }
    isConst = true;
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus AssignInputValue(
    gert::TilingContext* context, size_t size, const gert::Tensor* tensor, gert::Shape& list_vector)
{
    int32_t dim_num = tensor->GetShapeSize();
    list_vector.SetDimNum(dim_num);
    const T* value = tensor->GetData<T>();
    OP_CHECK_IF(
        value == nullptr,
        OP_LOGE(context->GetNodeName(), "get const value fail, check input is const or not."),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < size; i++) {
        list_vector[i] = value[i];
    }
    return ge::GRAPH_SUCCESS;
}

static bool CalcEndAndBeginList(
    gert::Shape& list_end_vector, gert::Shape& list_begin_vector, const gert::Shape& shape_input, size_t size,
    bool flag, bool is_begin_const)
{
    if (flag) {
        for (size_t index = 0; index < size; index++) {
            if (list_end_vector[index] == -1) {
                OP_CHECK_IF(
                    is_begin_const == false,
                    OP_LOGE("Slice", "end cannot be -1 while begin is not const"),
                    return false);
                list_end_vector[index] = shape_input.GetDim(index) - list_begin_vector[index];
            }
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            if (list_begin_vector[i] < 0 || list_begin_vector[i] + list_end_vector[i] < list_begin_vector[i] ||
                list_begin_vector[i] + list_end_vector[i] > shape_input.GetDim(i)) {
                OP_LOGE(
                    "Slice", "Requirements: 0<=offsets[i]<= offsets[i]+size[i]<=input_shape[i]");
                return false;
            }
        }
    }
    return true;
}

static void MakePerformanceParams(SliceParasRuntime2& parameters)
{
    if (!parameters.is_begin_const) {
        return;
    }
    SliceParasRuntime2 perf_params;
    perf_params.is_begin_const = parameters.is_begin_const;
    bool last_same = false;
    size_t perf_size = 0;
    for (size_t i = 0; i < parameters.input.GetDimNum(); i++) {
        const auto input_shape_i = parameters.input[i];
        const auto output_shape_i = parameters.output_shape[i];
        const auto begin_i = parameters.begin_list[i];
        const auto end_i = parameters.end_list[i];
        const auto stride_i = parameters.stride_list[i];
        if (input_shape_i != output_shape_i || !last_same) {
            perf_params.input.AppendDim(input_shape_i);
            perf_params.output_shape.AppendDim(output_shape_i);
            perf_params.begin_list.AppendDim(begin_i);
            perf_params.end_list.AppendDim(end_i);
            perf_params.stride_list.AppendDim(stride_i);
            last_same = (input_shape_i == output_shape_i);
            perf_size++;
            continue;
        }

        const auto perf_index = perf_size - static_cast<size_t>(1);
        perf_params.input[perf_index] *= input_shape_i;
        perf_params.output_shape[perf_index] *= output_shape_i;
        perf_params.begin_list[perf_index] = 0;
        perf_params.end_list[perf_index] = perf_params.input[perf_index];
        perf_params.stride_list[perf_index] = 1;
    }
    const size_t last_second = 2;
    if (perf_params.input.GetDimNum() > 1 &&
        perf_params.input.GetDim(perf_params.input.GetDimNum() - 1) ==
            perf_params.output_shape.GetDim(perf_params.output_shape.GetDimNum() - 1) &&
        perf_params.stride_list[perf_params.input.GetDimNum() - last_second] == 1) {
        const auto last_second_index = perf_params.input.GetDimNum() - last_second;
        perf_params.input[last_second_index] *= perf_params.input.GetDim(perf_params.input.GetDimNum() - 1);
        perf_params.output_shape[last_second_index] *=
            perf_params.output_shape.GetDim(perf_params.output_shape.GetDimNum() - 1);
        perf_params.begin_list[last_second_index] *= perf_params.input.GetDim(perf_params.input.GetDimNum() - 1);
        perf_params.end_list[last_second_index] *= perf_params.input.GetDim(perf_params.input.GetDimNum() - 1);
        perf_params.stride_list[last_second_index] = 1;

        perf_params.input.SetDimNum(perf_params.input.GetDimNum() - 1);
        perf_params.output_shape.SetDimNum(perf_params.output_shape.GetDimNum() - 1);
        perf_params.begin_list.SetDimNum(perf_params.begin_list.GetDimNum() - 1);
        perf_params.end_list.SetDimNum(perf_params.end_list.GetDimNum() - 1);
        perf_params.stride_list.SetDimNum(perf_params.stride_list.GetDimNum() - 1);
    }

    parameters = perf_params;
}

static ge::graphStatus Tiling4Slice(gert::TilingContext* context)
{
    auto compile_info = reinterpret_cast<const SliceCompileParam*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    const gert::Shape& in_shape = Ops::Base::EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    OP_CHECK_IF(m
        compile_info->block_dim == 0,
        OP_LOGE(context->GetNodeName(), "core num = 0 is not support"),
        return ge::GRAPH_FAILED);
    // instantiate param
    SliceParasRuntime2 sliceparam;
    sliceparam.input = in_shape;
    // get input desc
    auto shape_tensor_x = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor_x);
    // get offsets const value
    auto shape_tensor_offsets = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor_offsets);
    auto shape_size_offsets = static_cast<size_t>(shape_tensor_offsets->GetShapeSize());
    // get size const value
    auto shape_tensor_size = context->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor_size);
    auto shape_size_size = static_cast<size_t>(shape_tensor_size->GetShapeSize());
    // size must be equal
    OP_CHECK_IF(
        shape_size_offsets != shape_size_size,
        OP_LOGE(
            context->GetNodeName(), "length of input_shape, offsets and size must be equal."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        shape_size_offsets != in_shape.GetDimNum(),
        OP_LOGE(
            context->GetNodeName(), "length of input_shape, offsets and size must be equal."),
        return ge::GRAPH_FAILED);

    ge::DataType offset_dtype = shape_tensor_offsets->GetDataType();
    if (offset_dtype == ge::DT_INT32) {
        // Get offset const val
        OP_CHECK_IF(
            AssignInputValueOpt<int32_t>(
                context, shape_size_offsets, shape_tensor_offsets, sliceparam.begin_list, compile_info->isAscendc,
                sliceparam.is_begin_const) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get offset fail, check input is const or not."),
            return ge::GRAPH_FAILED);
    } else {
        // Get offset const val
        OP_CHECK_IF(
            AssignInputValueOpt<int64_t>(
                context, shape_size_offsets, shape_tensor_offsets, sliceparam.begin_list, compile_info->isAscendc,
                sliceparam.is_begin_const) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get offset fail, check input is const or not."),
            return ge::GRAPH_FAILED);
    }
    ge::DataType sizeDtype = shape_tensor_size->GetDataType();
    if (sizeDtype == ge::DT_INT32) {
        // Get size const val
        OP_CHECK_IF(
            AssignInputValue<int32_t>(context, shape_size_size, shape_tensor_size, sliceparam.end_list) !=
                ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get size fail, check input is const or not."),
            return ge::GRAPH_FAILED);
    } else {
        // Get size const val
        OP_CHECK_IF(
            AssignInputValue<int64_t>(context, shape_size_size, shape_tensor_size, sliceparam.end_list) !=
                ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "get size fail, check input is const or not."),
            return ge::GRAPH_FAILED);
    }
    // calc endlist
    bool end_list_flag = true;
    bool isEndValid = CalcEndAndBeginList(
        sliceparam.end_list, sliceparam.begin_list, in_shape, shape_size_offsets, end_list_flag,
        sliceparam.is_begin_const);
    bool begin_list_flag = false;
    bool isBeginValid = CalcEndAndBeginList(
        sliceparam.end_list, sliceparam.begin_list, in_shape, shape_size_size, begin_list_flag,
        sliceparam.is_begin_const);
    OP_CHECK_IF(
        (compile_info->isAscendc && (!isEndValid || !isBeginValid)),
        OP_LOGE("Slice", "CalcEndAndBeginList failed"), return ge::GRAPH_FAILED);

    // in slice end_list is size values
    sliceparam.output_shape = sliceparam.end_list;
    // calc end list
    for (size_t i = 0; i < shape_size_offsets; i++) {
        sliceparam.end_list[i] += sliceparam.begin_list[i];
    }
    for (size_t i = 0; i < shape_size_size; i++) {
        sliceparam.stride_list[i] = 1;
    }
    MakePerformanceParams(sliceparam);

    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    return SliceTilingForAscendC(
        context, compile_info->block_dim, compile_info->ub_size, compile_info->cacheLineSize, sliceparam,
        xDesc->GetDataType());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Slice(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4Slice.");
    auto compileInfo = context->GetCompiledInfo<SliceCompileParam>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->block_dim = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->block_dim <= 0), OP_LOGE(context->GetNodeName(), "block_dim invalid."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ub_size <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."),
        return ge::GRAPH_FAILED);

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->cacheLineSize == static_cast<uint32_t>(0)),
        OP_LOGE(context->GetNodeName(), "Failed to get cacheLineSize."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Slice).Tiling(Tiling4Slice).TilingParse<SliceCompileParam>(TilingPrepare4Slice);
} // namespace optiling