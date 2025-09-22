/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_data_def.h
 * \brief
 */
#ifndef TILING_DATA_DEF_H
#define TILING_DATA_DEF_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__
#pragma pack(1)

struct EdgeTiling {
    uint32_t edgeCount;
    uint32_t tileCount;
    uint32_t additionalCount;
};

struct PadV3GradReplicationTilingData {
    uint32_t addTensorBlockNum;
    uint32_t addTensorByteSize;
    uint32_t addTensorSize;
    uint32_t moveTensorBlockNum;
    uint32_t moveTensorByteSize;
    uint32_t moveTensorSize;

    uint32_t inputShape[4]; // 虚拟dim
    uint32_t inputSize;
    uint32_t cubeInputSize;
    uint32_t layerInputSize;
    uint32_t cubeNumEachCore;
    uint32_t realUsedCoreNum;
    uint32_t cubeNumLastCore;

    uint32_t outputShape[4]; // 虚拟dim
    uint32_t outputSize;
    uint32_t cubeOutputSize;
    uint32_t layerOutputSize;
    uint32_t paddings[6];

    uint32_t topSize;
    uint32_t totalTopInputSizeEachCube;
    int64_t leftSize;
    uint32_t totalLeftInputSizeEachCube;
    uint32_t innerRowLength;
    uint32_t innerRowCount;
    uint32_t topToBottomSize;
    uint32_t topResultSize;
    uint32_t leftToRightSize;
    uint32_t leftResultSize;
    uint32_t workspaceSize;

    EdgeTiling topTiling;
    EdgeTiling leftTiling;
    EdgeTiling cornerTiling;
    EdgeTiling innerTiling;
    EdgeTiling paddingLayerTiling;
    EdgeTiling topTilingLastCore;
    EdgeTiling leftTilingLastCore;
    EdgeTiling cornerTilingLastCore;
    EdgeTiling paddingLayerTilingLastCore;
};

#pragma pack()

inline void InitPadV3GradReplicationTilingData(uint8_t* tiling, PadV3GradReplicationTilingData* data)
{
    memcpy(data, tiling, sizeof(PadV3GradReplicationTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    PadV3GradReplicationTilingData tilingData; \
    InitPadV3GradReplicationTilingData(tilingArg, &tilingData)
#endif