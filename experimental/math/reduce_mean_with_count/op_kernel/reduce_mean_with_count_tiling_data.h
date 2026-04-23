/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file reduce_mean_with_count_tiling_data.h
 * @brief ReduceMeanWithCount TilingData structure definition
 */

#ifndef _REDUCE_MEAN_WITH_COUNT_TILING_DATA_H_
#define _REDUCE_MEAN_WITH_COUNT_TILING_DATA_H_

struct ReduceMeanWithCountTilingData {
    // ---- Merged axis shape parameters ----
    uint64_t a1Length;         // A1 dimension size (outer preserved axis product)
    uint64_t rLength;          // R dimension size (reduction axis product)
    uint64_t a0Length;         // A0 dimension size (inner preserved axis product), 1 for AR mode

    // ---- Multi-core split parameters ----
    int32_t usedCoreNum;       // Actual number of cores used
    uint64_t tilesPerCore;     // Number of tiles per core (first usedCoreNum-1 cores)
    uint64_t tailCoreTiles;    // Number of tiles for tail core

    // ---- UB split parameters ----
    uint64_t tileA0Len;        // ARA mode: length along A0 per tile
    uint64_t chunkR;           // AR split-load mode: R columns per chunk

    // ---- Compute parameters ----
    float invCount;            // 1.0f / countResult, for Muls to compute mean
    int64_t countResult;       // Number of elements participating in reduction
    uint64_t tmpBufSize;       // ReduceSum sharedTmpBuffer size (bytes)

    // ---- Output info ----
    uint64_t outputLength;     // Total elements in mean_result output
};

#endif // _REDUCE_MEAN_WITH_COUNT_TILING_DATA_H_
