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
 * \file equal_tiling_data.h
 * \brief tiling data struct
 */

#ifndef EXPERIMENTAL_MATH_EQUAL_OP_KERNEL_EQUAL_TILING_DATA_H_
#define EXPERIMENTAL_MATH_EQUAL_OP_KERNEL_EQUAL_TILING_DATA_H_

constexpr uint32_t EQUAL_MAX_BROADCAST_DIM = 8;

enum EqualBroadcastMode : uint32_t {
    EQUAL_BROADCAST_CONTIGUOUS = 0,
    EQUAL_BROADCAST_X1_SCALAR = 1,
    EQUAL_BROADCAST_X2_SCALAR = 2,
    EQUAL_BROADCAST_GENERAL = 3,
    EQUAL_BROADCAST_SANDWICH = 4,
    EQUAL_BROADCAST_TAIL_REUSE = 5,
};

struct EqualTilingData {
    int64_t totalLength;
    int64_t blockLength;
    int64_t tailBlockLength;
    int64_t tileLength;
    uint32_t blockNum;
    uint32_t broadcastMode;
    uint32_t rank;
    uint32_t fastBroadcastInput;
    int64_t x1Length;
    int64_t x2Length;
    int64_t outShape[EQUAL_MAX_BROADCAST_DIM];
    int64_t x1Stride[EQUAL_MAX_BROADCAST_DIM];
    int64_t x2Stride[EQUAL_MAX_BROADCAST_DIM];
    int64_t fastOuter;
    int64_t fastMiddle;
    int64_t fastTail;
    int64_t fastPlaneLength;
    int64_t fastSourceLength;
    uint64_t fastTmpBytes;
};
#endif // EXPERIMENTAL_MATH_EQUAL_OP_KERNEL_EQUAL_TILING_DATA_H_
