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
 * \file strided_slice_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDESLICE_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDESLICE_TILING_H
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <vector>

#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_api/op_util.h"
#include "../strided_slice_util.h"
#include "../../op_kernel/arch35/strided_slice_struct.h"

namespace optiling {
constexpr int64_t MAX_TWO_DIM_UB_SPLIT_AXIS_NUM = 2;
constexpr int64_t MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM = 4;
constexpr int64_t MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG = 4;
constexpr int64_t MAX_SLICE_GATHER_UB_SPLIT_AXIS_NUM = 5;
constexpr int64_t DATA_SPARSITY_THRESHOLD = 10;
constexpr int64_t DATA_COPY_SPARSITY_THRESHOLD = 5;
constexpr int64_t MAX_UINT16_NUM = 65535;
constexpr int64_t MAX_UINT32_NUM = 4294967295;
constexpr int64_t UB_RESERVE_SIZE = 8192;
constexpr int64_t INIT_INDEX_SIZE = 512; // 因为只考虑搬入连续的倒数2个维度
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t VL_SIZE = 256;
constexpr int64_t MIN_MOVE_ALIGN_LEN = 128;
constexpr int64_t NUMBER_THREE = 3;
constexpr int64_t NUMBER_FOUR = 4;
constexpr int64_t NUMBER_SIX = 6;
constexpr int64_t STRIDED_SLICE_KEY_MOVE_ALIGN = 100;
constexpr int64_t STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM = 101;
constexpr int64_t STRIDED_SLICE_KEY_NDDMA = 102;
constexpr int64_t STRIDED_SLICE_KEY_NDDMA_LAST_DIM = 103;
constexpr int64_t STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM = 150;

constexpr int64_t STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER = 300;
constexpr int64_t STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB = 301;
constexpr int64_t STRIDED_SLICE_KEY_NDDMA_GATHER = 302;
constexpr int64_t STRIDED_SLICE_KEY_NDDMA_UB2UB = 303;

constexpr int64_t STRIDED_SLICE_KEY_SIMT = 200;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_PACK_2B = 201;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_PACK_4B = 202;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_PACK_8B = 203;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_PACK_16B = 204;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_BIG_SHAPE = 210;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_2B = 211;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_4B = 212;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_8B = 213;
constexpr int64_t STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_16B = 214;
constexpr int64_t RESERVE_LAST_DIM_SIZE = 64;
constexpr int64_t RESERVE_HALF_LAST_DIM_SIZE = 32;
constexpr int64_t LAST_DIM_MIN_DATA_SIZE = 1024;
constexpr int64_t SIMT_MIN_OUTPUT_SIZE = 1024;
constexpr int64_t TWO_DIM_MIN_OUTPUT_SIZE = 4096; // 4 * 1024

struct StridedSliceCompileInfo {
    int32_t numBlocks{-1};
    int32_t ubSize{0};
    int32_t coreNum{0};
    uint32_t cacheLineSize{0};
    bool isAscendc{false};
    std::string to_string() const
    {
        std::string str = "numBlocks: " + std::to_string(numBlocks);
        str += " ubSize: " + std::to_string(ubSize);
        return str;
    }
};

struct SliceParametersRuntime2 {
    gert::Shape inputShape;
    gert::Shape outputShape;
    ops::QuickVector beginList;
    ops::QuickVector endList;
    ops::QuickVector strideList;
    int64_t tilingMode = 0;
    int64_t coreNum = 0;
    bool isBeginConst = true;
    bool isEndConst = true;

    std::string to_string() const
    {
        std::string result = "inputShape:" + Ops::Base::ToString(inputShape);
        result += " outputShape:" + Ops::Base::ToString(outputShape);
        result += " begin:" + Ops::Base::ToString(beginList);
        result += " end:" + Ops::Base::ToString(endList);
        result += " stride:" + Ops::Base::ToString(strideList);
        result += " tilingMode:" + std::to_string(tilingMode);
        result += " coreNum:" + std::to_string(coreNum);
        result += " isBeginConst:" + std::string(isBeginConst ? "true" : "false");
        result += " isEndConst:" + std::string(isEndConst ? "true" : "false");
        return result;
    }
};

ge::graphStatus StrideSliceTilingForAscendC(gert::TilingContext* tilingContext, int64_t coreNum, int64_t ubSize,
                                            int64_t cachelineSize, SliceParametersRuntime2& sliceParam,
                                            const ge::DataType& dtype);

struct MoveAlignV2Info {
    uint16_t blockCount = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    uint16_t loop1Size = 1;
    uint16_t loop2Size = 1;
    uint32_t loop1SrcStride = 0;
    uint16_t loop1DstStride = 0;
    uint32_t loop2SrcStride = 0;
    uint16_t loop2DstStride = 0;
};

class StrideSliceTiling {
public:
    explicit StrideSliceTiling(gert::TilingContext* tilingContext) : tilingContext_(tilingContext) {};
    ge::graphStatus Init(int64_t coreNum, int64_t ubSize, int64_t cacheLineSize, SliceParametersRuntime2& sliceParam,
                         const ge::DataType& dtype);
    ge::graphStatus RunStrideSliceTiling();

protected:
    virtual void CalMaxSplitDim();
    virtual void SetTilingMode();
    virtual void SetMovAlignV2TilingMode();
    virtual void SetNddmaTilingMode();
    virtual void FillTilingData();
    void FillStridedSliceBaseTilingData(StridedSliceBaseTilingData& tilingData);
    void FillStridedSliceTilingData100();
    void FillStridedSliceTilingData101();
    void FillStridedSliceTilingDataNDDMA();
    void FillStridedSliceTilingData150();
    void FillStridedSliceTilingDataSIMT();
    void FillStridedSliceTilingData300();
    void FillStridedSliceTilingData301();
    void FillStridedSliceTilingDataOther();
    virtual void PrintTilingData();
    void PrintStridedSliceBaseTilingData(StridedSliceBaseTilingData& tilingData);
    void PrintStridedSliceTilingData100();
    void PrintStridedSliceTilingData101();
    void PrintStridedSliceTilingDataNDDMA();
    void PrintStridedSliceTilingData150();
    void PrintStridedSliceTilingData300();
    void PrintStridedSliceTilingData301();
    void PrintStridedSliceTilingDataSIMT();
    void PrintStridedSliceTilingDataOther();
    void CalStridedSliceRowsStepsParams();
    virtual void SetBlockDimAndTilingKey();
    void SetRowsStepsParamsFor150(StridedSliceMALast2DimTilingData& tilingData);
    void SetRowsStepsParams(StridedSliceTilingData& tilingData);
    std::string ArrayToStr(const int64_t* arr, size_t aSize) const;
    void SetMoveAlignParams(StridedSliceMoveAlignParams& params, const MoveAlignV2Info& actInfo);
    void SetShortMoveAlignParams(StridedSliceShortMoveAlignParams& params, const MoveAlignV2Info& actInfo);
    void SetTwoDimTilingInfo();
    ge::graphStatus CheckPlatformParam();

private:
    void CalMaxSplitDimNeg();
    void CalcUbSplitInfoNeg();
    void SetTilingModeNeg();
    int64_t GetValidNumInCacheLine();
    int64_t GetDimStartPosition(int32_t idx);
    int64_t GetDimEndPosition(int32_t idx);
    void CalMaxSplitDimLastStrideNegative();
    void CalMaxSplitDimLastStrideOne();
    void CalMaxSplitDimLastStridePositive();
    void SetMovAlignV2TilingModeNeg();
    void SetNddmaTilingModeNeg();
    void SetAllInUbSplitInfo();
    void CalcUbSplitInfo();
    void CalcBlockSplitInfo();
    void MovAlignV2UbSplitLastOneDim();
    void MovAlignV2UbSplitLastTwoDim();
    void MovAlignV2UbSplitLastThreeDim();
    void MovAlignV2UbSplitLastFourDim();
    void SetSIMTTilingMode();
    void CalSimtPackFactor();
    void CalcThreadNum();
    void ResetMovAlignV2Para(int64_t newUbfactor);
    void CalInputOutputSize();

protected:
    StridedSliceTilingData tilingData_;
    StridedSliceMATilingData maTilingData_;
    StridedSliceMALastDimTilingData maLastDimTilingData_;
    StridedSliceMALast2DimTilingData maLast2DimTilingData_;
    StridedSliceMAGatherTilingData maGatherTilingData_;
    StridedSliceMAUB2UBTilingData maUB2UBTilingData_;
    StridedSliceNDDMATilingData nddmaTilingData_;
    StridedSliceSIMTTilingData simtTilingData_;

    gert::TilingContext* tilingContext_ = nullptr;
    int32_t dimNum_ = 0;
    int64_t ubElementNum_ = 0;
    int32_t maxSplitDim_ = MAX_SIMT_UB_SPLIT_AXIS_NUM;

    // static ge::graphStatus ConstructSliceParam中构造SliceParametersRuntime2
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;       // 正stride使用
    int64_t ubSizeInput_ = 0;  // 负stride使用
    int64_t ubSizeOutput_ = 0; // 负stride使用
    int64_t cacheLineSize_ = 0;
    SliceParametersRuntime2 sliceParam_;
    int64_t inputShape_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
    int64_t outputShape_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
    int64_t begin_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
    int64_t end_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
    int64_t strides_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};
    int64_t rowsOffsetSteps_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0}; // output每个维度包含的行数
    int64_t inputSteps_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};      // input每个维度跨度(个数)
    int64_t outputSteps_[MAX_AXIS_NUM_FOR_STRIDESLICE] = {0};     // output每个维度跨度(个数)
    // input: (7, 6, 5, 4)
    // inputSteps_: (7*6*5*4, 6*5*4, 5*4, 4)
    // output: (6, 5, 4, 3)
    // outputSteps_: (6*5*4*3, 5*4*3, 4*3, 3)
    // rowsOffsetSteps_: (6*5*4, 5*4, 4, 1)

    int64_t nddmaTotalNum_ = 1;
    int64_t nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {0};
    int64_t nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {0};
    int64_t nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {0};

    int64_t outputShapeProd_[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t inputShapeProd_[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};

    ge::DataType dtype_ = ge::DT_MAX;
    int64_t xDtypeSize_ = 0;
    int64_t ubIndex_ = -1L;
    int64_t ubFactor_ = -1L;
    int64_t ubTailFactor_ = 0;
    int64_t ubTailTailFactor_ = 0;
    bool isAllInUb_ = false;
    int64_t blkIndex_ = -1L;
    int64_t blkFactor_ = 0;
    int64_t blkTailFactor_ = 0;
    int64_t realCoreNum_ = 1;
    int64_t inLoopSteps_ = 1;
    int64_t outLoopSteps_ = 1;
    bool isShapeExceedUint32_ = false;
    bool isShapeBelowThreshold_ = false;
    bool isStrideNeg_ = false;
    bool isEmptyTensor_ = false;
    bool useGather_ = false;
    bool isNddma_ = false;
    bool isSliceGather_ = false;

    // insr parameter
    MoveAlignV2Info mainMoveAlignV2Info_;
    uint32_t outBlockLen_ = 0;

    uint32_t mainCoreNum_ = 0;

    int64_t tilingKey_ = -1L;
    int64_t totalOutputSize_ = 1;
    int64_t simtPackFactor_ = 1;
    int64_t mainBlockFactor_ = 0;
    uint16_t threadNum_ = 2048;

    int64_t lastOneOutputDim_ = 0;
    int64_t lastTwoOutputDim_ = 0;
    int64_t lastThreeOutputDim_ = 0;
    int64_t lastOneInputDim_ = 0;
    int64_t lastTwoInputDim_ = 0;
    int64_t lastThreeInputDim_ = 0;
    int64_t lastOneStride_ = 0;
    int64_t lastTwoStride_ = 0;
    int64_t lastThreeStride_ = 0;
    int64_t lastFourStride_ = 0;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDESLICE_TILING_H
