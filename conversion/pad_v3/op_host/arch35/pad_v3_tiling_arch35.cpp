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
 * \file pad_v3_tiling_arch35.cpp
 * \brief ac pad v3 tiling cpp
 */

#include "pad_v3_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace AscendC;

namespace optiling {
static constexpr uint64_t CONSTANT_SLICE_BRANCH = 10000;
static constexpr uint64_t CONSTANT_SIMT_BRANCH = 20000;
static constexpr uint64_t CONSTANT_SIMT_BIG_SIZE_BRANCH = 20001;
static constexpr uint64_t CONSTANT_CUT_LAST_DIM_BRANCH = 30000;
static constexpr uint64_t CONSTANT_BIG_LAST_DIM_BRANCH = 30001;
static constexpr uint64_t CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH = 30002;
static constexpr uint64_t CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH = 30003;
static constexpr uint64_t EDGE_SIMT_BRANCH = 23000;
static constexpr uint64_t EDGE_SIMT_BIG_SIZE_BRANCH = 23001;
static constexpr uint64_t EDGE_CUT_LAST_DIM_BRANCH = 33000;
static constexpr uint64_t EDGE_BIG_LAST_DIM_BRANCH = 33001;
static constexpr uint64_t EDGE_SMALL_LAST_DIM_GATHER_BRANCH = 33002;
static constexpr uint64_t EDGE_SMALL_LAST_DIM_SCATTER_BRANCH = 33003;

static constexpr uint64_t REFLECT_SIMT_BRANCH = 21000;
static constexpr uint64_t REFLECT_SIMT_BIG_SIZE_BRANCH = 21001;
static constexpr uint64_t REFLECT_CUT_LAST_DIM_BRANCH = 31000;
static constexpr uint64_t REFLECT_BIG_LAST_DIM_BRANCH = 31001;
static constexpr uint64_t REFLECT_SMALL_LAST_DIM_GATHER_BRANCH = 31002;

static constexpr uint64_t SYMMETRIC_SIMT_BRANCH = 22000;
static constexpr uint64_t SYMMETRIC_SIMT_BIG_SIZE_BRANCH = 22001;
static constexpr uint64_t SYMMETRIC_CUT_LAST_DIM_BRANCH = 32000;
static constexpr uint64_t SYMMETRIC_BIG_LAST_DIM_BRANCH = 32001;
static constexpr uint64_t SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH = 32002;

static constexpr uint64_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr size_t PADDINGS_IDX = 1;
static constexpr size_t PAIR = 2;
static constexpr uint64_t SIMT_BRANCH_SIZE = 4 * 1024;
static constexpr uint64_t UB_MAX_DATA_SIZE_PER_BUFFER = 64 * 1024;
static constexpr uint8_t MAX_DIM_NUM = 8;

static constexpr uint8_t PAD_DIM_INDEX_SECOND = 2;
static constexpr uint8_t PAD_DIM_INDEX_THIRD = 3;
static constexpr uint8_t PAD_DIM_INDEX_FOURTH = 4;
static constexpr uint64_t EXPANSION_FACTOR = 2;
static constexpr uint64_t HALF_FACTOR = 2;
static constexpr uint64_t DIM_OFFSET_SCALE = 10;
static constexpr uint8_t UB_DIVIDER = 6;
static constexpr uint8_t CONST5 = 5;

static const std::unordered_map<ge::DataType, uint16_t> DATA_TYPE_TO_BYTES{
    {ge::DataType::DT_INT8, 1},   {ge::DataType::DT_UINT8, 1},   {ge::DataType::DT_INT16, 2},
    {ge::DataType::DT_UINT16, 2}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2},
    {ge::DataType::DT_FLOAT, 4},  {ge::DataType::DT_INT32, 4},   {ge::DataType::DT_INT64, 8},
    {ge::DataType::DT_UINT32, 4}, {ge::DataType::DT_UINT64, 8},  {ge::DataType::DT_DOUBLE, 8},
    {ge::DataType::DT_BOOL, 1}};

template <typename T>
std::string PadACTiling::ToString(const T* value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ",";
    }
    r = r + "]";
    return r;
}

uint64_t PadACTiling::GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize)
{
    if (alignBlockSize == 0) {
        return 0;
    }
    return (inputSize + alignBlockSize - 1) / alignBlockSize * alignBlockSize;
}

void PadACTiling::DoFindSplitAxisReflect(bool isBigLastDim)
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey DoFindSplitAxis.");
    uint64_t dimSizeInUb = dtypeBytes_;
    uint64_t dimSizeInLast4Axis = dimSizeInUb;
    // 找到切分轴
    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        if (isBigLastDim && i == static_cast<int64_t>(dimNum_ - 1)) {
            dimSizeInUb = GetSizeOfBlockAlign(dimSizeInUb * tilingData_->outShape[i], blockSize_);
        } else {
            dimSizeInUb *= tilingData_->outShape[i];
        }
        // 切分超过4根轴时只切最后4根轴，记录下最后4根轴的大小
        if (i == dimNum_ - PAD_DIM_INDEX_FOURTH) {
            dimSizeInLast4Axis = dimSizeInUb;
        }
        if (dimSizeInUb >= bufferSize_) {
            ubAxis_ = i;
            break;
        }
    }
    // 维度超过4，满载后4个轴
    if (dimNum_ - ubAxis_ > PAD_DIM_INDEX_FOURTH) {
        ubAxis_ = dimNum_ - PAD_DIM_INDEX_FOURTH;
        ubFactor_ = tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        dimSizeInUb = dimSizeInLast4Axis / tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_FOURTH] *
                      tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    } else if (dimSizeInUb > bufferSize_) {
        dimSizeInUb /= tilingData_->outShape[ubAxis_];
        ubFactor_ = bufferSize_ / dimSizeInUb;
        if (ubFactor_ > tilingData_->inShape[ubAxis_]) {
            ubFactor_ = tilingData_->inShape[ubAxis_];
        }
        dimSizeInUb *= ubFactor_;
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    } else {
        ubFactor_ = tilingData_->inShape[ubAxis_];
        dimSizeInUb = dimSizeInUb / tilingData_->outShape[ubAxis_] * tilingData_->inShape[ubAxis_];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    }
}

void PadACTiling::DoFindSplitAxis(bool isBigLastDim)
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey DoFindSplitAxis.");
    uint64_t dimSizeInUb = dtypeBytes_;
    uint64_t dimSizeInLast4Axis = dimSizeInUb;
    // 找到切分轴
    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        if (isBigLastDim && i == static_cast<int64_t>(dimNum_ - 1)) {
            dimSizeInUb = GetSizeOfBlockAlign(dimSizeInUb * tilingData_->outShape[i], blockSize_);
        } else {
            dimSizeInUb *= tilingData_->outShape[i];
        }
        // 切分超过4根轴时只切最后4根轴，记录下最后4根轴的大小
        if (i == dimNum_ - PAD_DIM_INDEX_FOURTH) {
            dimSizeInLast4Axis = dimSizeInUb;
        }
        if (dimSizeInUb >= bufferSize_) {
            ubAxis_ = i;
            break;
        }
    }

    // 维度超过4，满载后4个轴
    if (dimNum_ - ubAxis_ > PAD_DIM_INDEX_FOURTH) {
        ubAxis_ = dimNum_ - PAD_DIM_INDEX_FOURTH;
        ubFactor_ = tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInLast4Axis, blockSize_);
        return;
    }

    if (dimSizeInUb > bufferSize_) {
        dimSizeInUb /= tilingData_->outShape[ubAxis_];
        if (dimSizeInUb != 0) {
            ubFactor_ = bufferSize_ / dimSizeInUb;
        }
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb * ubFactor_, blockSize_);
        return;
    }

    ubFactor_ = tilingData_->outShape[ubAxis_];
    outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
}

void PadACTiling::CalculateGatherOrScatterReflect()
{
    tilingKey_ = REFLECT_SMALL_LAST_DIM_GATHER_BRANCH;
    if (padMode_ == ModeNum::SYMMETRIC) {
        tilingKey_ = SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH;
    }
}

void PadACTiling::CalculateGatherOrScatterEdge()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey CalculateGatherOrScatterEdge.");

    tilingKey_ = EDGE_SMALL_LAST_DIM_GATHER_BRANCH;
}

void PadACTiling::CalculateGatherOrScatter()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey CalculateGatherOrScatter.");
    uint8_t vlAxis = dimNum_ - 1;
    uint64_t dimSizeInVL = dtypeBytes_;

    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        dimSizeInVL *= tilingData_->outShape[i];
        if (dimSizeInVL >= vectorSize_) {
            vlAxis = i;
            break;
        }
    }

    if (dimNum_ - 1 - vlAxis == 1 &&
        tilingData_->inShape[dimNum_ - 1] * EXPANSION_FACTOR < tilingData_->outShape[dimNum_ - 1]) {
        tilingKey_ = CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH;
        return;
    }

    if (dimNum_ - 1 - vlAxis == PAD_DIM_INDEX_SECOND &&
        tilingData_->inShape[dimNum_ - 1] * tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_SECOND] * EXPANSION_FACTOR <
            tilingData_->outShape[dimNum_ - 1] * tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_SECOND]) {
        tilingKey_ = CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH;
        return;
    }
    tilingKey_ = CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH;
}

void PadACTiling::CalculateTilingKeyReflect()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKeyReflect.");
    if (outShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > CONST5) {
        tilingKey_ = REFLECT_SIMT_BRANCH;
        if (padMode_ == ModeNum::SYMMETRIC) {
            tilingKey_ = SYMMETRIC_SIMT_BRANCH;
        }
        return;
    }
    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / UB_DIVIDER - vectorSize_, vectorSize_);
    additionTileSize_ = vectorSize_ * EXPANSION_FACTOR;
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER;
    }
    if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        tilingKey_ = REFLECT_CUT_LAST_DIM_BRANCH;
        if (padMode_ == ModeNum::SYMMETRIC) {
            tilingKey_ = SYMMETRIC_CUT_LAST_DIM_BRANCH;
        }
        outTileSize_ = bufferSize_;
        return;
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) * EXPANSION_FACTOR >
            bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->outShape[dimNum_ - 1];
        tilingKey_ = REFLECT_CUT_LAST_DIM_BRANCH;
        if (padMode_ == ModeNum::SYMMETRIC) {
            tilingKey_ = SYMMETRIC_CUT_LAST_DIM_BRANCH;
        }
        outTileSize_ = bufferSize_;
        return;
    }
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        if (dtypeBytes_ == 1) {
            additionTileSize_ =
                GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ * EXPANSION_FACTOR, blockSize_);
        }
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - additionTileSize_) / UB_DIVIDER - vectorSize_, vectorSize_);
        DoFindSplitAxisReflect(true);
        if (tilingKey_ == REFLECT_SIMT_BRANCH || tilingKey_ == SYMMETRIC_SIMT_BRANCH) {
            return;
        }
        tilingKey_ = REFLECT_BIG_LAST_DIM_BRANCH;
        if (padMode_ == ModeNum::SYMMETRIC) {
            tilingKey_ = SYMMETRIC_BIG_LAST_DIM_BRANCH;
        }
        if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) * EXPANSION_FACTOR >
            outTileSize_) {
            bufferSize_ = GetSizeOfBlockAlign(ubSize_ / UB_DIVIDER - vectorSize_, vectorSize_);
            ubAxis_ = dimNum_ - 1;
            ubFactor_ = tilingData_->outShape[dimNum_ - 1];
            tilingKey_ = REFLECT_CUT_LAST_DIM_BRANCH;
            if (padMode_ == ModeNum::SYMMETRIC) {
                tilingKey_ = SYMMETRIC_CUT_LAST_DIM_BRANCH;
            }
            outTileSize_ = bufferSize_;
            return;
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign(
            (ubSize_ - vectorSize_ * EXPANSION_FACTOR - blockSize_ * PAD_DIM_INDEX_FOURTH) / UB_DIVIDER - vectorSize_,
            vectorSize_);
        DoFindSplitAxisReflect(false);
        if (tilingKey_ == REFLECT_SIMT_BRANCH || tilingKey_ == SYMMETRIC_SIMT_BRANCH) {
            return;
        }
        // 按vl切分判断是否走scatter
        CalculateGatherOrScatterReflect();
        additionTileSize_ = vectorSize_;
    }
}

void PadACTiling::CalculateTilingKeyEdge()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKeyEdge.");
    // MovealignV2指令限制，轴较多时走SIMT
    if (outShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > PAD_DIM_INDEX_FOURTH ||
        (dimNum_ == PAD_DIM_INDEX_FOURTH && (tilingData_->leftPad[0] != 0 || rightPad_[0] != 0))) {
        tilingKey_ = EDGE_SIMT_BRANCH;
        return;
    }
    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / PAIR - vectorSize_, vectorSize_);
    additionTileSize_ = vectorSize_;
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER;
    }
    if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        tilingKey_ = EDGE_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return;
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) * EXPANSION_FACTOR >
            bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->outShape[dimNum_ - 1];
        tilingKey_ = EDGE_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return;
    }
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        DoFindSplitAxis(true);
        if (tilingKey_ == EDGE_SIMT_BRANCH) {
            return;
        }
        tilingKey_ = EDGE_BIG_LAST_DIM_BRANCH;
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        // 如果切完后会爆ub，走切W分支
        if ((outTileSize_ + additionTileSize_) * EXPANSION_FACTOR > ubSize_) {
            ubAxis_ = dimNum_ - 1;
            ubFactor_ = tilingData_->outShape[dimNum_ - 1] / dtypeBytes_;
            tilingKey_ = EDGE_CUT_LAST_DIM_BRANCH;
            outTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
            additionTileSize_ = vectorSize_;
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - vectorSize_ * PAIR) / PAIR / PAIR, vectorSize_);
        DoFindSplitAxis(false);
        if (tilingKey_ == EDGE_SIMT_BRANCH) {
            return;
        }
        // 按vl切分判断是否走scatter
        CalculateGatherOrScatterEdge();
        additionTileSize_ = outTileSize_ + vectorSize_;
    }
}

void PadACTiling::CalculateTilingKey()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey.");
    if (outShapeSize_ <= SIMT_BRANCH_SIZE) {
        tilingKey_ = CONSTANT_SIMT_BRANCH;
        return;
    }
    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / PAIR - vectorSize_, vectorSize_);
    additionTileSize_ = vectorSize_;
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER;
    }
    if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        tilingKey_ = CONSTANT_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return;
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_) * EXPANSION_FACTOR >
            bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->outShape[dimNum_ - 1];
        tilingKey_ = CONSTANT_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return;
    }
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        DoFindSplitAxis(true);
        if (tilingKey_ == CONSTANT_SIMT_BRANCH) {
            return;
        }
        tilingKey_ = CONSTANT_BIG_LAST_DIM_BRANCH;
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        // 如果切完后会爆ub，走切W分支
        if ((outTileSize_ + additionTileSize_) * EXPANSION_FACTOR > ubSize_) {
            ubAxis_ = dimNum_ - 1;
            ubFactor_ = tilingData_->outShape[dimNum_ - 1] / dtypeBytes_;
            tilingKey_ = CONSTANT_CUT_LAST_DIM_BRANCH;
            outTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
            additionTileSize_ = vectorSize_;
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - vectorSize_ * PAIR) / PAIR / PAIR, vectorSize_);
        DoFindSplitAxis(false);
        if (tilingKey_ == CONSTANT_SIMT_BRANCH) {
            return;
        }
        // 按vl切分判断是否走scatter
        CalculateGatherOrScatter();
        additionTileSize_ = outTileSize_ + vectorSize_;
    }
}
void PadACTiling::DoTilingWithReflect()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithReflect.");
    CalculateTilingKeyReflect();
    if (tilingKey_ == REFLECT_SIMT_BRANCH || tilingKey_ == SYMMETRIC_SIMT_BRANCH) {
        DoTilingWithSIMTReflect();
        return;
    }
    tilingKey_ = tilingKey_ + (dimNum_ - ubAxis_) * DIM_OFFSET_SCALE;
    uint64_t factorCount = Ops::Base::CeilDiv(tilingData_->inShape[ubAxis_], static_cast<uint64_t>(ubFactor_));
    ubTotalCount_ = factorCount;
    for (uint64_t i = 0; i < ubAxis_; i++) {
        ubTotalCount_ *= tilingData_->inShape[i];
    }
    if (ubTotalCount_ > coreNum_) {
        ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, coreNum_);
        coreNum_ = Ops::Base::CeilDiv(ubTotalCount_, ubPerCount_);
    } else {
        ubPerCount_ = 1;
        coreNum_ = ubTotalCount_;
    }
}

void PadACTiling::DoTilingWithEdge()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithEdge.");
    CalculateTilingKeyEdge();
    if (tilingKey_ == EDGE_SIMT_BRANCH) {
        DoTilingWithSIMTEdge();
        return;
    }
    CaculateTilingParams();
}

void PadACTiling::CaculateTilingParams()
{
    tilingKey_ = tilingKey_ + (dimNum_ - ubAxis_) * DIM_OFFSET_SCALE;
    uint64_t factorCount = Ops::Base::CeilDiv(tilingData_->outShape[ubAxis_], static_cast<uint64_t>(ubFactor_));
    ubTotalCount_ = factorCount;
    for (uint64_t i = 0; i < ubAxis_; i++) {
        ubTotalCount_ *= tilingData_->outShape[i];
    }
    if (ubTotalCount_ > coreNum_) {
        ubPerCount_ = Ops::Base::CeilDiv(ubTotalCount_, coreNum_);
    } else {
        ubPerCount_ = 1;
        coreNum_ = ubTotalCount_;
    }
}

void PadACTiling::DoTilingWithConstant()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithConstant.");
    CalculateTilingKey();
    if (tilingKey_ == CONSTANT_SIMT_BRANCH) {
        DoTilingWithSIMT();
        return;
    }
    CaculateTilingParams();
}

void PadACTiling::DoTilingWithSIMTReflect()
{
    tilingKey_ = REFLECT_SIMT_BRANCH;
    if (padMode_ == ModeNum::SYMMETRIC) {
        tilingKey_ = SYMMETRIC_SIMT_BRANCH;
    }
    OP_LOGD(context_, "Start PadACTiling DoTilingWithSIMTReflect.");
    if (inShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX || outShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX) {
        tilingKey_ = REFLECT_SIMT_BIG_SIZE_BRANCH;
        if (padMode_ == ModeNum::SYMMETRIC) {
            tilingKey_ = SYMMETRIC_SIMT_BIG_SIZE_BRANCH;
        }
    }
}

void PadACTiling::DoTilingWithSliceOp()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithSliceOp.");
    optiling::SliceParasRuntime2 param;
    for (size_t i = 0; i < dimNum_; i++) {
        const auto input_shape_i = tilingData_->inShape[i];
        const auto output_shape_i = tilingData_->outShape[i];
        const auto begin_i = -tilingData_->leftPad[i];
        const auto end_i = output_shape_i + begin_i;
        const auto stride_i = 1;
        param.input.AppendDim(input_shape_i);
        param.output_shape.AppendDim(output_shape_i);
        param.begin_list.AppendDim(begin_i);
        param.end_list.AppendDim(end_i);
        param.stride_list.AppendDim(stride_i);
    }
    OP_LOGI(
        context_, "input_shape_ = %s, output_shape_ = %s, begin_ = %s, end_ = %s, stride_ = %s",
        Ops::Base::ToString(param.input).c_str(), Ops::Base::ToString(param.output_shape).c_str(),
        Ops::Base::ToString(param.begin_list).c_str(), Ops::Base::ToString(param.end_list).c_str(),
        Ops::Base::ToString(param.stride_list).c_str());
    ge::graphStatus res = SliceTilingForAscendC(context_, coreNum_, ubSize_, cacheLineSize_, param, paramsDtype_);
    tilingKey_ += context_->GetTilingKey();
    if (res != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SliceTilingForAscendc failed.");
    }
}

void PadACTiling::DoTilingWithSIMTEdge()
{
    tilingKey_ = EDGE_SIMT_BRANCH;
    OP_LOGD(context_, "Start PadACTiling DoTilingWithSIMTEdge.");
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        tilingKey_ = EDGE_SIMT_BIG_SIZE_BRANCH;
    }
}

void PadACTiling::DoTilingWithSIMT()
{
    tilingKey_ = CONSTANT_SIMT_BRANCH;
    OP_LOGD(context_, "Start PadACTiling DoTilingWithSIMT.");
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        tilingKey_ = CONSTANT_SIMT_BIG_SIZE_BRANCH;
    }
}

void PadACTiling::EmptyTensorCollapse()
{
    OP_LOGD(context_, "Start PadACTiling EmptyTensorCollapse.");
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = outShapeSize_;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
}

void PadACTiling::EmptyTensorCollapseEdge()
{
    OP_LOGD(context_, "Start PadACTiling EmptyTensorCollapse.");
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = 0;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
}

void PadACTiling::EmptyTensorCollapseReflect()
{
    OP_LOGD(context_, "Start PadACTiling EmptyTensorCollapse.");
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = 0;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
}

ge::graphStatus PadACTiling::ComputeAfterPaddingsAndStrides()
{
    OP_LOGD(context_, "Start PadACTiling ComputeAfterPaddingsAndStrides.");

    inShapeSize_ = 1UL;
    outShapeSize_ = 1UL;
    for (int64_t i = dimNum_ - 1; i >= 0; --i) {
        // 校验是否会出现pad为负且比inshape大的情况，为0时会走simt

        if (static_cast<int64_t>(tilingData_->inShape[i]) + tilingData_->leftPad[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) + rightPad_[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) + tilingData_->leftPad[i] + rightPad_[i] < 0) {
            OP_LOGE(context_, "outShape length must be non-negative.");
            return ge::GRAPH_FAILED;
        }
        // compute shape after padding
        tilingData_->outShape[i] = tilingData_->inShape[i] + tilingData_->leftPad[i] + rightPad_[i];
        // compute src stride
        tilingData_->inStride[i] = inShapeSize_;
        inShapeSize_ *= tilingData_->inShape[i];
        // compute dst stride
        tilingData_->outStride[i] = outShapeSize_;
        outShapeSize_ *= tilingData_->outShape[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::ReflectDimensionCollapse()
{
    OP_LOGD(context_, "Start PadACTiling EdgeDimensionCollapse.");
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(
        context_, "Before collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", originalRank,
        ToString(tilingData_->inShape, originalRank).c_str(), Ops::Base::ToString(paddings_.padFront).c_str(),
        Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
        int64_t padFront = paddings_.padFront.GetDim(fastDim);
        int64_t padBack = paddings_.padBack.GetDim(fastDim);
        if (padFront < 0 || padBack < 0) {
            isPadAllPositive_ = false;
        }
        if (padFront > 0 || padBack > 0) {
            isPadAllNegative_ = false;
        }
        int64_t leftsubin = padFront - static_cast<int64_t>(tilingData_->inShape[fastDim]);
        int64_t rightsubin = padBack - static_cast<int64_t>(tilingData_->inShape[fastDim]);
        if (padMode_ == ModeNum::REFLECT && (0 < (leftsubin + 1) || 0 < (rightsubin + 1))) {
            OP_LOGE(context_, "padFront and padBack bigger than inShape - 1.");
            return ge::GRAPH_FAILED;
        }
        if (padMode_ == ModeNum::SYMMETRIC && (0 < leftsubin || 0 < rightsubin)) {
            OP_LOGE(context_, "padFront and padBack bigger than inShape.");
            return ge::GRAPH_FAILED;
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
        rightPad_[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    OP_LOGD(
        context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, originalRank).c_str(),
        ToString(rightPad_, originalRank).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::EdgeDimensionCollapse()
{
    OP_LOGD(context_, "Start PadACTiling EdgeDimensionCollapse.");
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(
        context_, "Before collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", originalRank,
        ToString(tilingData_->inShape, originalRank).c_str(), Ops::Base::ToString(paddings_.padFront).c_str(),
        Ops::Base::ToString(paddings_.padBack).c_str());

    while (fastDim < originalRank) {
        int64_t padFront = paddings_.padFront.GetDim(fastDim);
        int64_t padBack = paddings_.padBack.GetDim(fastDim);
        if (padFront < 0 || padBack < 0) {
            isPadAllPositive_ = false;
        }
        if (padFront > 0 || padBack > 0) {
            isPadAllNegative_ = false;
        }

        if (tilingData_->inShape[fastDim] == 0 && (padFront != 0 || padBack != 0)) {
            OP_LOGE(context_, "If inShape == 0 , padFront and padBack must be 0.");
            return ge::GRAPH_FAILED;
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
        rightPad_[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    OP_LOGD(
        context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, originalRank).c_str(),
        ToString(rightPad_, originalRank).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DimensionCollapse()
{
    OP_LOGD(context_, "Start PadACTiling DimensionCollapse.");
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(
        context_, "Before collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", originalRank,
        ToString(tilingData_->inShape, originalRank).c_str(), Ops::Base::ToString(paddings_.padFront).c_str(),
        Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
        int64_t padFront = paddings_.padFront.GetDim(fastDim);
        int64_t padBack = paddings_.padBack.GetDim(fastDim);
        if (padFront < 0 || padBack < 0) {
            isPadAllPositive_ = false;
        }
        if (padFront > 0 || padBack > 0) {
            isPadAllNegative_ = false;
        }
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
        rightPad_[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    OP_LOGD(
        context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, originalRank).c_str(),
        ToString(rightPad_, originalRank).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void PadACTiling::GetPaddingsToShape(const gert::Tensor* paddingsTensor)
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo GetPaddings GetPaddingsToShape.");
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

ge::graphStatus PadACTiling::GetPaddings()
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo GetPaddings.");
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

ge::graphStatus PadACTiling::GetShapesAndDtypes()
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo GetShapesAndDtypes.");
    // get input shape & input shape's dim num
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    dimNum_ = inShapeVal.GetDimNum();
    if (dimNum_ > MAX_DIM_NUM) {
        OP_LOGE(context_, "input shape dim should <= 8, please check.");
        return ge::GRAPH_FAILED;
    }
    for (uint16_t i = 0; i < dimNum_; ++i) {
        if (inShapeVal.GetDim(i) < 0) {
            OP_LOGE(context_, "Input shape should >= 0, please check");
            return ge::GRAPH_FAILED;
        }
        if (inShapeVal.GetDim(i) == 0) {
            isEmptyTensor_ = true; // when empty tensor, set isEmptyTensor_ = true so that kernel will be memset
        }
        tilingData_->inShape[i] = inShapeVal.GetDim(i);
    }
    // 判断类型
    auto inputTensor = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputTensor);
    paramsDtype_ = inputTensor->GetDataType();
    dtypeBytes_ = GetSizeByDataType(paramsDtype_);
    if (isPadV3_ && padMode_ == ModeNum::CONSTANT) {
        auto inConstantValues = context_->GetInputDesc(2);
        if (inConstantValues && paramsDtype_ != inConstantValues->GetDataType()) {
            OP_LOGE(context_, "DataType of constant_values must equal inputData's DataType.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo isPadV3_:%d.", isPadV3_);
    if (isPadV3_) {
        auto const attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        auto* mode = attrs->GetAttrPointer<char>(0);
        // padv3 mode可选，默认constant; mirrorpad mode必选
        if (mode) {
            if (!strcmp(mode, "edge")) {
                padMode_ = ModeNum::EDGE;
            } else if (!strcmp(mode, "reflect") || (isMirrorPad_ && !strcmp(mode, "REFLECT"))) {
                padMode_ = ModeNum::REFLECT;
            } else if (!strcmp(mode, "symmetric") || (isMirrorPad_ && !strcmp(mode, "SYMMETRIC"))) {
                padMode_ = ModeNum::SYMMETRIC;
            }
            OP_CHECK_IF(
                !isMirrorPad_ && strcmp(mode, "constant") != 0 && strcmp(mode, "edge") != 0 &&
                    strcmp(mode, "reflect") != 0 && strcmp(mode, "symmetric") != 0,
                OP_LOGE(context_, "Current PadV3 AscendC impl only support constant/edge/reflect/symmetric mode."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                isMirrorPad_ && strcmp(mode, "REFLECT") != 0 && strcmp(mode, "SYMMETRIC") != 0,
                OP_LOGE(context_, "MirrorPad only support REFLECT and SYMMETRIC mode."), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                isMirrorPad_, OP_LOGE(context_, "MirrorPad mode attr cannot be empty"), return ge::GRAPH_FAILED);
        }

        if (!isMirrorPad_) {
            auto* paddingContiguous = attrs->GetAttrPointer<bool>(1);
            if (paddingContiguous) {
                paddingContiguous_ = *paddingContiguous;
            }
        }
    }

    OP_CHECK_IF(
        GetShapesAndDtypes() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo GetShapesAndDtypes error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetPaddings() == ge::GRAPH_FAILED, OP_LOGE(context_, "Get padding info failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void PadACTiling::FillsAndPrintTilingData()
{
    tilingData_->dimNum = dimNum_;
    tilingData_->ubAxis = ubAxis_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->ubPerCount = ubPerCount_;
    tilingData_->ubTotalCount = ubTotalCount_;
    tilingData_->outTileSize = outTileSize_;
    tilingData_->additionTileSize = additionTileSize_;
    OP_LOGI(
        context_,
        "tilingData is dimNum = %u, ubAxis = %u, ubFactor = %u, ubPerCount = %u, ubTotalCount = %u, outTileSize = %u, additionTileSize = %u, \
                inShape: %s, outShape: %s, inStride: %s, outStride: %s, leftPad: %s, rightPad: %s, tilingKey: %lu, coreNum is %u",
        tilingData_->dimNum, tilingData_->ubAxis, tilingData_->ubFactor, tilingData_->ubPerCount,
        tilingData_->ubTotalCount, tilingData_->outTileSize, tilingData_->additionTileSize,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->outShape, dimNum_).c_str(),
        ToString(tilingData_->inStride, dimNum_).c_str(), ToString(tilingData_->outStride, dimNum_).c_str(),
        ToString(tilingData_->leftPad, dimNum_).c_str(), ToString(rightPad_, dimNum_).c_str(), tilingKey_, coreNum_);
}

ge::graphStatus PadACTiling::Init()
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
    tilingData_ = context_->GetTilingData<PadACTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DoTiling()
{
    OP_LOGD(context_, "Start PadACTiling DoTiling.");
    OP_CHECK_IF(Init() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling Init error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetShapeAttrsInfo() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."),
        return ge::GRAPH_FAILED);
    if (padMode_ == ModeNum::EDGE) {
        OP_CHECK_IF(
            EdgeDimensionCollapse() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
            OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."), return ge::GRAPH_FAILED);
        if (isEmptyTensor_) {
            EmptyTensorCollapseEdge();
            DoTilingWithSIMTEdge();
        } else if (isPadAllPositive_) {
            DoTilingWithEdge();
        } else if (isPadAllNegative_) {
            tilingKey_ = CONSTANT_SLICE_BRANCH;
            DoTilingWithSliceOp();
            context_->SetTilingKey(tilingKey_);
            OP_LOGD(context_, "Exit PadACTiling DoTiling.");
            return ge::GRAPH_SUCCESS;
        } else {
            DoTilingWithSIMTEdge();
        }
    } else if (padMode_ == ModeNum::REFLECT || padMode_ == ModeNum::SYMMETRIC) {
        OP_CHECK_IF(
            ReflectDimensionCollapse() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
            OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."), return ge::GRAPH_FAILED);
        if (isEmptyTensor_) {
            EmptyTensorCollapseReflect();
            DoTilingWithSIMTReflect();
        } else if (isPadAllPositive_) {
            DoTilingWithReflect();
        } else if (isPadAllNegative_) {
            tilingKey_ = CONSTANT_SLICE_BRANCH;
            DoTilingWithSliceOp();
            context_->SetTilingKey(tilingKey_);
            OP_LOGD(context_, "Exit PadACTiling DoTiling.");
            return ge::GRAPH_SUCCESS;
        } else {
            DoTilingWithSIMTReflect();
        }
    } else {
        OP_CHECK_IF(
            DimensionCollapse() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
            OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."), return ge::GRAPH_FAILED);
        // 空tensor时，合为一根轴
        if (isEmptyTensor_) {
            EmptyTensorCollapse();
        }
        if (isPadAllPositive_) {
            DoTilingWithConstant();
        } else if (isPadAllNegative_) {
            tilingKey_ = CONSTANT_SLICE_BRANCH;
            DoTilingWithSliceOp();
            context_->SetTilingKey(tilingKey_);
            OP_LOGD(context_, "Exit PadACTiling DoTiling.");
            return ge::GRAPH_SUCCESS;
        } else {
            DoTilingWithSIMT();
        }
    }
    FillsAndPrintTilingData();
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORK_SPACE_SIZE;

    OP_LOGD(context_, "Exit PadACTiling DoTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PadV3Tiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "PadV3Tiling running begin");
    const PadV3CompileInfo* compile_info = reinterpret_cast<const PadV3CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Tiling4Pad dsl compile_info is Null, running AscendC tiling.");
    PadACTiling tilingObject(context);
    tilingObject.isPadV3_ = true;
    return tilingObject.DoTiling();
}

ge::graphStatus TilingPreparePadv3ForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init PadV3 AscendC Tiling.");
    auto ci = context->GetCompiledInfo<PadV3CompileInfo>();
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

static ge::graphStatus TilingPrepare4PadV3(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<PadV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4PadV3 Mode success!");
    TilingPreparePadv3ForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the PadV3 op.
IMPL_OP_OPTILING(PadV3).Tiling(PadV3Tiling).TilingParse<PadV3CompileInfo>(TilingPrepare4PadV3);
} // namespace optiling
