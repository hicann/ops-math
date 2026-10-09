/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOOR_MOD_TILING_DATA_H
#define FLOOR_MOD_TILING_DATA_H

#include <cstdint>

namespace FloorModNs {

constexpr uint32_t FLOOR_MOD_MAX_BROADCAST_DIM = 8U;
constexpr uint32_t FLOOR_MOD_MAX_DIGITS = 8U;

enum FloorModKernelMode : uint32_t {
    FLOOR_MOD_MODE_DENSE = 0U,
    FLOOR_MOD_MODE_ROW_REUSE = 1U,
    FLOOR_MOD_MODE_EXPAND_REUSE = 2U,
    FLOOR_MOD_MODE_CROSSED = 3U,
};

enum FloorModReuseLayout : uint32_t {
    FLOOR_MOD_REUSE_PADDED_REPEAT = 0U,
    FLOOR_MOD_REUSE_PACKED_BROADCAST = 1U,
    FLOOR_MOD_REUSE_PADDED_COMPACT = 2U,
    FLOOR_MOD_REUSE_COMPACT_POS_BROADCAST = 3U,
    FLOOR_MOD_REUSE_PADDED_MATERIALIZED = 4U,
    // One operand is scalar only in the final dimension and both operands
    // have the same dense prefix. Cores own contiguous logical rows, so each
    // tile uses one compact seed read, one dense read, and one dense write.
    FLOOR_MOD_REUSE_DENSE_TAIL_BATCH = 5U,
    // A single trailing vector is reused by contiguous logical rows. Each
    // batch uses compact GM transfers. BroadCast first writes aligned FP32
    // rows, then GatherMask compacts them for contiguous arithmetic.
    FLOOR_MOD_REUSE_COMPACT_ROW_BATCH = 6U,
    // Same batched GM topology as COMPACT_ROW_BATCH, but vector repeat stride
    // zero reuses the resident vector. This avoids materialization when row
    // width gives sufficiently high repeat-lane utilization.
    FLOOR_MOD_REUSE_COMPACT_ROW_REPEAT = 7U,
};

enum FloorModArithmeticMode : uint32_t {
    FLOOR_MOD_ARITH_MUL_SUB = 0U,
    FLOOR_MOD_ARITH_FMA_NEG_DENOMINATOR = 1U,
};

struct FloorModTilingData {
    uint32_t mode;
    uint32_t coreNum;
    uint32_t dtypeKey;
    uint32_t dtypeBytes;
    uint32_t rank;
    uint32_t arithmeticMode; // FloorModArithmeticMode
    uint64_t totalElements;
    uint64_t x1Elements;
    uint64_t x2Elements;
    uint32_t seedIsX1;
    uint32_t swapped;

    // Dense path: each core owns one contiguous range and iterates denseTile.
    uint32_t denseTile;

    // Reuse paths: output = [pos][e1][D][e2].
    uint32_t e1;
    uint64_t D;
    uint32_t e2;
    uint32_t dTile;
    uint32_t dTileCount;
    uint32_t e2Tile;
    uint32_t e2TileCount;
    uint32_t reuseSchedule; // 0: tile-owned, 1: row-owned
    uint32_t reuseLayout;   // FloorModReuseLayout
    uint32_t rowPartitions;
    uint32_t batchRows;
    uint32_t posTile;
    uint64_t totalTasks;
    uint32_t maxPatternElems;
    uint32_t maxTRowElems;
    uint32_t maxFp32RowElems;
    uint32_t floorTmpBytes;
    uint32_t broadcastTmpBytes;
    uint32_t ubUsedBytes;
    uint32_t scalarDoubleBuffer;

    uint32_t posTotal;
    uint32_t posDigits;
    uint32_t posExtent[FLOOR_MOD_MAX_DIGITS];
    uint64_t seedStride[FLOOR_MOD_MAX_DIGITS];
    uint64_t denseStride[FLOOR_MOD_MAX_DIGITS];

    // Crossed path: output = [outer][A][M][B][crossD].
    uint32_t crossOuter;
    uint32_t crossA;
    uint32_t crossM;
    uint32_t crossB;
    uint32_t crossD;
    uint32_t crossDAligned;
    uint32_t crossOuterTile;
    uint32_t crossATile;
    uint32_t crossBTile;
    uint32_t crossUnitBTAligned;
    uint32_t crossUnitBAligned;
    uint32_t crossOuterTileCount;
    uint32_t crossATileCount;
    uint32_t crossBTileCount;
    uint64_t crossTotalTasks;

    // Shape metadata retained for diagnostics and cost-model observations.
    uint64_t outShape[FLOOR_MOD_MAX_BROADCAST_DIM];
    uint64_t x1Stride[FLOOR_MOD_MAX_BROADCAST_DIM];
    uint64_t x2Stride[FLOOR_MOD_MAX_BROADCAST_DIM];
};

} // namespace FloorModNs

#endif // FLOOR_MOD_TILING_DATA_H
