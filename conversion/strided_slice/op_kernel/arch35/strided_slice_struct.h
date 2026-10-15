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
 * \file strided_slice_struct.h
 * \brief tiling base data
 */

#ifndef STRIDED_SLICE_STRUCT_H
#define STRIDED_SLICE_STRUCT_H

#include <cstdint>

constexpr int64_t MAX_AXIS_NUM_FOR_STRIDESLICE = 8;
constexpr int64_t MAX_NDDMA_UB_SPLIT_AXIS_NUM = 5;
constexpr int64_t MAX_SIMT_UB_SPLIT_AXIS_NUM = 8;
constexpr int64_t NUMBER_TWO = 2;

struct StridedSliceMoveAlignParams {
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

struct StridedSliceShortMoveAlignParams {
    uint16_t blockCount = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
};

struct StridedSliceBaseTilingData {
    int64_t blkFactor = 0;
    int64_t blkTailFactor = 0;
    int64_t ubInLoopSteps = 0;
    int64_t ubOutLoopSteps = 0;
    int32_t ubSize = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
    int32_t ubTailTailFactor = 0;
    int16_t realCoreNum = 0;
    int16_t inputDims = 0;
    int16_t blkIndex = 0;
    int16_t ubIndex = 0;
    int64_t outputShape[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t begin[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t strides[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t inputSteps[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t rowsOffsetSteps[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
};

struct StridedSliceMATilingData {
    StridedSliceBaseTilingData stridedSliceBaseTilingData;
    StridedSliceMoveAlignParams moveAlignParams;
};

struct StridedSliceMALastDimTilingData {
    StridedSliceBaseTilingData stridedSliceBaseTilingData;
};

struct StridedSliceNDDMATilingData {
    StridedSliceBaseTilingData stridedSliceBaseTilingData;
    int64_t nddmaTotalNum = 1;
    int32_t ubSizeInput = 0;
    int64_t nddmaLoopSize[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {};
    int64_t nddmaLoopSrcStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {};
    int64_t nddmaLoopDstStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {};
};

struct StridedSliceMALast2DimTilingData {
    int64_t blkFactor = 0;
    int64_t blkTailFactor = 0;
    int64_t ubInLoopSteps = 0;
    int64_t ubOutLoopSteps = 0;
    int32_t ubSize = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
    int32_t ubTailTailFactor = 0;
    int16_t realCoreNum = 0;
    StridedSliceShortMoveAlignParams moveAlignParams;
    int64_t outputShape[NUMBER_TWO] = {};
    int64_t begin[NUMBER_TWO] = {};
    int64_t strides[NUMBER_TWO] = {};
    int64_t inputSteps[NUMBER_TWO] = {};
};

struct StridedSliceSIMTTilingData {
    uint16_t isEmptyTensor = 0;
    int16_t inputDims = 0;
    uint16_t threadNum = 2048;
    int64_t begin[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t strides[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t outputShapeProd[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {};
    int64_t inputShapeProd[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {};
};

struct StridedSliceMAGatherTilingData {
    StridedSliceBaseTilingData stridedSliceBaseTilingData;
    StridedSliceMoveAlignParams moveAlignParams;
    int32_t ubSizeInput = 0;
};

struct StridedSliceMAUB2UBTilingData {
    StridedSliceBaseTilingData stridedSliceBaseTilingData;
    StridedSliceMoveAlignParams moveAlignParams;
    int32_t ubSizeInput = 0;
};

struct StridedSliceTilingData {
    int8_t isBeginConst = 0;
    int64_t ubSize = 0;
    int64_t ubSizeInput = 0; // stride为负时使用
    int64_t coreNum = 0;
    int64_t ubIndex = 0;
    int64_t ubFactor = 0;
    int64_t ubTailFactor = 0;
    int64_t ubTailTailFactor = 0;
    int64_t realCoreNum = 0;
    int64_t inputDims = 0;
    int64_t blkIndex = 0;
    int64_t blkFactor = 0;
    int64_t blkTailFactor = 0;
    int64_t xDtypeSize = 0;
    int64_t tilingKey = 0;
    int64_t nddmaTotalNum = 0;
    int64_t ubInLoopSteps = 0;
    int64_t ubOutLoopSteps = 0;
    uint32_t isShapeExceedUint32 = 0;
    uint32_t isEmptyTensor = 0;
    StridedSliceMoveAlignParams moveAlignParams;
    int64_t outputShape[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t begin[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t strides[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t rowsOffsetSteps[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t inputSteps[MAX_AXIS_NUM_FOR_STRIDESLICE] = {};
    int64_t nddmaLoopSize[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {};
    int64_t nddmaLoopSrcStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {};
    int64_t nddmaLoopDstStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM] = {};
    int64_t outputShapeProd[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {};
    int64_t inputShapeProd[MAX_SIMT_UB_SPLIT_AXIS_NUM] = {};
};

#endif // STRIDED_SLICE_STRUCT_H
