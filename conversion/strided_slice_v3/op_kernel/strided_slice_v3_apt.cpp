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
 * \file strided_slice_v3.cpp
 * \brief
 */
#include "../strided_slice/arch35/strided_slice.h"

using namespace StridedSlice;

#define STRIDED_SLICE_KEY_MOVE_ALIGN 100
#define STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM 101
#define STRIDED_SLICE_KEY_NDDMA 102
#define STRIDED_SLICE_KEY_NDDMA_LAST_DIM 103
#define STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM 150
#define STRIDED_SLICE_KEY_SIMT 200
#if defined(__DAV_920R1__)
#define STRIDED_SLICE_KEY_SIMT_PACK_2B 201
#define STRIDED_SLICE_KEY_SIMT_PACK_4B 202
#define STRIDED_SLICE_KEY_SIMT_PACK_8B 203
#define STRIDED_SLICE_KEY_SIMT_PACK_16B 204
#endif
#define STRIDED_SLICE_KEY_SIMT_BIG_SHAPE 210
#if defined(__DAV_920R1__)
#define STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_2B 211
#define STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_4B 212
#define STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_8B 213
#define STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_16B 214
#endif
#define STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER 300
#define STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB 301
#define STRIDED_SLICE_KEY_NDDMA_GATHER 302
#define STRIDED_SLICE_KEY_NDDMA_UB2UB 303

extern "C" __global__ __aicore__ void strided_slice_v3(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR axes,
                                                       GM_ADDR strides, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(StridedSliceTilingData);
    if (TILING_KEY_IS(STRIDED_SLICE_KEY_MOVE_ALIGN)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceMATilingData, tilingData, tiling);
        StridedSliceMoveAlignProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceMALastDimTilingData, tilingData, tiling);
        StridedSliceMoveAlignLastDimProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_NDDMA)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceNDDMATilingData, tilingData, tiling);
        StridedSliceNDDMAProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_NDDMA_LAST_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceNDDMATilingData, tilingData, tiling);
        StridedSliceNDDMALastDimProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceMALast2DimTilingData, tilingData, tiling);
        StridedSliceMoveAlignTwoDimProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTProcess(x, begin, end, strides, y, tiling, &tilingData);
#if defined(__DAV_920R1__)
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_PACK_2B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTPack2BProcess(x, begin, end, strides, y, tiling, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_PACK_4B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTPack4BProcess(x, begin, end, strides, y, tiling, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_PACK_8B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTPack8BProcess(x, begin, end, strides, y, tiling, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_PACK_16B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTPack16BProcess(x, begin, end, strides, y, tiling, &tilingData);
#endif
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_BIG_SHAPE)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTBigShapeProcess(x, begin, end, strides, y, tiling, &tilingData);
#if defined(__DAV_920R1__)
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_2B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTBigShapePack2BProcess(x, begin, end, strides, y, tiling, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_4B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTBigShapePack4BProcess(x, begin, end, strides, y, tiling, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_8B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTBigShapePack8BProcess(x, begin, end, strides, y, tiling, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_SIMT_BIG_SHAPE_PACK_16B)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceSIMTTilingData, tilingData, tiling);
        StridedSliceSIMTBigShapePack16BProcess(x, begin, end, strides, y, tiling, &tilingData);
#endif
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceMAGatherTilingData, tilingData, tiling);
        StridedSliceMoveAlignGatherProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceMAUB2UBTilingData, tilingData, tiling);
        StridedSliceMoveAlignUb2UbProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_NDDMA_GATHER)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceNDDMATilingData, tilingData, tiling);
        StridedSliceNDDMAGatherProcess(x, begin, end, strides, y, &tilingData);
    } else if (TILING_KEY_IS(STRIDED_SLICE_KEY_NDDMA_UB2UB)) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceNDDMATilingData, tilingData, tiling);
        StridedSliceNDDMAUb2UbProcess(x, begin, end, strides, y, &tilingData);
    }
}
