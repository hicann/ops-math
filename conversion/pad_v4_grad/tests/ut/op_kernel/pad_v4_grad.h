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
 * \file pad_v4_grad.cc
 * \brief
 */

#include <map>
#include <string>

#include "tiling_data_def.h"
namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr size_t MODE_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PAD_INPUT_INDEX = 1;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t FLOAT16_BYTES = 2;
constexpr size_t CHECK_DIM_NUM = 4;
constexpr uint32_t FLOAT_MINI_SHAPE_TILING_KEY = 1000;
constexpr uint32_t FLOAT_SMALL_H_LARGE_W_TILING_KEY = 1100;
constexpr uint32_t FLOAT_LARGE_H_SMALL_W_TILING_KEY = 1010;
constexpr uint32_t FLOAT_NO_W_PAD_TILING_KEY = 1110;
constexpr uint32_t FLOAT_NO_H_PAD_TILING_KEY = 1101;
constexpr uint32_t FLOAT_H_W_PAD_TILING_KEY = 1111;
constexpr uint32_t FLOAT16_MINI_SHAPE_TILING_KEY = 2000;
constexpr uint32_t FLOAT16_SMALL_H_LARGE_W_TILING_KEY = 2100;
constexpr uint32_t FLOAT16_LARGE_H_SMALL_W_TILING_KEY = 2010;
constexpr uint32_t FLOAT16_NO_W_PAD_TILING_KEY = 2110;
constexpr uint32_t FLOAT16_NO_H_PAD_TILING_KEY = 2101;
constexpr uint32_t FLOAT16_H_W_PAD_TILING_KEY = 2111;
constexpr uint32_t BFLOAT16_MINI_SHAPE_TILING_KEY = 3000;
constexpr uint32_t BFLOAT16_SMALL_H_LARGE_W_TILING_KEY = 3100;
constexpr uint32_t BFLOAT16_LARGE_H_SMALL_W_TILING_KEY = 3010;
constexpr uint32_t BFLOAT16_NO_W_PAD_TILING_KEY = 3110;
constexpr uint32_t BFLOAT16_NO_H_PAD_TILING_KEY = 3101;
constexpr uint32_t BFLOAT16_H_W_PAD_TILING_KEY = 3111;
constexpr uint32_t FLOAT_EDGE_TILING_KEY = 12;
constexpr uint32_t FLOAT16_EDGE_TILING_KEY = 22;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t PADDING_NUM_INDEX4 = 4;
constexpr uint32_t PADDING_NUM_INDEX5 = 5;
constexpr uint32_t PADDING_NUM_INDEX6 = 6;
constexpr uint32_t PADDING_NUM_INDEX7 = 7;
constexpr uint32_t RESERVED_UB = 11 * 1024;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t TRANSPOSE_LINES = 16;
constexpr uint32_t CAL_COUNT = 64;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t WORK_SPACE_PART = 64;
constexpr int32_t MINI_SHAPE_MAX_HEIGHT = 128;
constexpr int32_t FLOAT16_MINI_SHAPE_MAX_WIDTH = 224;
constexpr int32_t FLOAT_MINI_SHAPE_MAX_WIDTH = 112;
constexpr int32_t BFLOAT16_MINI_SHAPE_MAX_WIDTH = 112;
constexpr int32_t SMALL_W_LIMIT = 128;
constexpr int32_t SMALL_H_LIMIT = 64;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t CONST_VALUE_3 = 3;
constexpr uint32_t CONST_VALUE_4 = 4;
constexpr uint32_t CONST_VALUE_5 = 5;
constexpr uint32_t CONST_VALUE_6 = 6;
constexpr uint32_t CONST_VALUE_8 = 8;
constexpr uint32_t CONST_VALUE_12 = 12;
constexpr uint32_t REFLECTION_MODE = 0;
constexpr uint32_t EDGE_MODE = 1;
constexpr uint32_t FLOAT_DTYPE = 1;
constexpr uint32_t FLOAT16_DTYPE = 2;
constexpr uint32_t BF16_DTYPE = 3;
constexpr int32_t W_PAD_UPPER_LIMIT = 4128;
constexpr int32_t W_PAD_LOWER_LIMIT = 16;

template <typename TilingData, int32_t dataTypeLen>
class PadV3GradV2Tiling {
public:
    explicit PadV3GradV2Tiling(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize)
    {
        this->batch = param.batch;
        this->channel = param.channel;
        this->height = param.height;
        this->width = param.width;
        this->alignHeight = param.alignHeight;
        this->alignWidth = param.alignWidth;
        this->outHeight = param.outHeight;
        this->outWidth = param.outWidth;
        this->alignOutHeight = param.alignOutHeight;
        this->alignOutWidth = param.alignOutWidth;
        this->hPad1 = param.hPad1;
        this->hPad2 = param.hPad2;
        this->wPad1 = param.wPad1;
        this->wPad2 = param.wPad2;
        this->mode = param.mode;
        this->dtype = param.dtype;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        this->coreNum = inputCoreNum;
        return;
    }

    void GetTiling(TilingData* tilingData);

private:
    void GetTilingKey();
    void GetUsedCore();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 FloorDiv(T1 a, T2 b)
    {
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b)
    {
        return (a) / b * b;
    }

private:
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    int32_t hPad1 = 0;
    int32_t hPad2 = 0;
    int32_t wPad1 = 0;
    int32_t wPad2 = 0;
    uint32_t mode = 0;
    uint32_t ubSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t coreNum = 0;
    uint32_t ncPerCore = 1;
    uint32_t tailNC = 0;
    uint32_t ubFactorElement = 0;
    uint32_t tilingKey = 0;
    uint32_t dtype = 1;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t divideUbNum = 1;
    uint64_t workspacePerCore = 0;
    uint32_t wPadCopyCount = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetTilingKey()
{
    if (dtype == FLOAT_DTYPE && mode == REFLECTION_MODE) {
        if (height <= MINI_SHAPE_MAX_HEIGHT && width <= FLOAT_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT_MINI_SHAPE_TILING_KEY; // float, mini shape
            divideUbNum = CONST_VALUE_3;
        } else if (wPad1 == 0 && wPad2 == 0 && (hPad1 != 0 || hPad2 != 0) && height > SMALL_H_LIMIT) {
            tilingKey = FLOAT_NO_W_PAD_TILING_KEY; // float, w dim no pad
            divideUbNum = CONST_VALUE_8;
        } else if (hPad1 == 0 && hPad2 == 0 && (wPad1 != 0 || wPad2 != 0) && width > SMALL_W_LIMIT) {
            tilingKey = FLOAT_NO_H_PAD_TILING_KEY; // float, reflect, h dim no pad
            divideUbNum = CONST_VALUE_4;
        } else if (height <= SMALL_H_LIMIT && width > FLOAT_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT_SMALL_H_LARGE_W_TILING_KEY; // float, mini h dim
            divideUbNum = CONST_VALUE_3;
        } else if (height > MINI_SHAPE_MAX_HEIGHT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT_LARGE_H_SMALL_W_TILING_KEY; // float, mini w dim
            divideUbNum = CONST_VALUE_3;
        } else {
            tilingKey = FLOAT_H_W_PAD_TILING_KEY; // float32, big shape, impl h_dim and w_dim pad
            divideUbNum = CONST_VALUE_5;
        }
    } else if (dtype == FLOAT_DTYPE && mode == EDGE_MODE) {
        tilingKey = FLOAT_EDGE_TILING_KEY; // mode2: float, edge
    } else if (dtype == FLOAT16_DTYPE && mode == REFLECTION_MODE) {
        if (height <= MINI_SHAPE_MAX_HEIGHT && width <= FLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT16_MINI_SHAPE_TILING_KEY; // float16, mini shape
            divideUbNum = CONST_VALUE_3;
        } else if (wPad1 == 0 && wPad2 == 0 && (hPad1 != 0 || hPad2 != 0) && height > SMALL_H_LIMIT) {
            tilingKey = FLOAT16_NO_W_PAD_TILING_KEY; // w dim no pad
            divideUbNum = CONST_VALUE_8;
        } else if (hPad1 == 0 && hPad2 == 0 && (wPad1 != 0 || wPad2 != 0) && width > SMALL_W_LIMIT) {
            tilingKey = FLOAT16_NO_H_PAD_TILING_KEY; // h dim no pad
            divideUbNum = CONST_VALUE_4;
        } else if (height <= SMALL_H_LIMIT && width > FLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT16_SMALL_H_LARGE_W_TILING_KEY; // float, mini h dim
            divideUbNum = CONST_VALUE_3;
        } else if (height > MINI_SHAPE_MAX_HEIGHT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT16_LARGE_H_SMALL_W_TILING_KEY; // float, mini w dim
            divideUbNum = CONST_VALUE_3;
        } else {
            tilingKey = FLOAT16_H_W_PAD_TILING_KEY; // float16, big shape, impl h_dim and w_dim pad
            divideUbNum = CONST_VALUE_5;
        }
    } else if (dtype == FLOAT16_DTYPE && mode == EDGE_MODE) {
        tilingKey = FLOAT16_EDGE_TILING_KEY; // mode2: float16, edge
    } else if (dtype == BF16_DTYPE && mode == REFLECTION_MODE) {
        if (height <= MINI_SHAPE_MAX_HEIGHT && width <= BFLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = BFLOAT16_MINI_SHAPE_TILING_KEY; // bfloat16, mini shape
            divideUbNum = CONST_VALUE_6;
        } else if (wPad1 == 0 && wPad2 == 0 && (hPad1 != 0 || hPad2 != 0) && height > SMALL_H_LIMIT) {
            tilingKey = BFLOAT16_NO_W_PAD_TILING_KEY; // w dim no pad
            divideUbNum = CONST_VALUE_12;
        } else if (hPad1 == 0 && hPad2 == 0 && (wPad1 != 0 || wPad2 != 0) && width > SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_NO_H_PAD_TILING_KEY; // h dim no pad
            divideUbNum = CONST_VALUE_6;
        } else if (height <= SMALL_H_LIMIT && width > BFLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = BFLOAT16_SMALL_H_LARGE_W_TILING_KEY; // bfloat16, mini h dim
            divideUbNum = CONST_VALUE_6;
        } else if (height > MINI_SHAPE_MAX_HEIGHT && width <= SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_LARGE_H_SMALL_W_TILING_KEY; // bfloat16, mini w dim
            divideUbNum = CONST_VALUE_6;
        } else {
            tilingKey = BFLOAT16_H_W_PAD_TILING_KEY; // bfloat16, big shape, impl h_dim and w_dim pad
            divideUbNum = CONST_VALUE_8;
        }
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetUsedCore()
{
    uint64_t nMulC = batch * channel;
    if (tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        nMulC = nMulC * height;
    }
    if (nMulC <= coreNum) {
        ncPerCore = 1;
        usedCoreNum = nMulC;
        tailNC = 0;
        return;
    }
    ncPerCore = nMulC / coreNum;
    tailNC = nMulC % coreNum;
    usedCoreNum = coreNum;
    return;
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::SplitUb()
{
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (tilingKey == FLOAT_H_W_PAD_TILING_KEY || tilingKey == FLOAT16_H_W_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_H_W_PAD_TILING_KEY) {
        ubFactorElement =
            FloorAlign(FloorAlign(canUseUbSize / divideUbNum / TRANSPOSE_LINES, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (
        tilingKey == FLOAT_MINI_SHAPE_TILING_KEY || tilingKey == FLOAT16_MINI_SHAPE_TILING_KEY ||
        tilingKey == BFLOAT16_MINI_SHAPE_TILING_KEY) {
        ubFactorElement = FloorAlign(
            FloorAlign(canUseUbSize / divideUbNum / MINI_SHAPE_MAX_HEIGHT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (
        tilingKey == FLOAT_SMALL_H_LARGE_W_TILING_KEY || tilingKey == FLOAT16_SMALL_H_LARGE_W_TILING_KEY ||
        tilingKey == BFLOAT16_SMALL_H_LARGE_W_TILING_KEY) {
        ubFactorElement =
            FloorAlign(FloorAlign(canUseUbSize / divideUbNum / SMALL_H_LIMIT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (
        tilingKey == FLOAT_LARGE_H_SMALL_W_TILING_KEY || tilingKey == FLOAT16_LARGE_H_SMALL_W_TILING_KEY ||
        tilingKey == BFLOAT16_LARGE_H_SMALL_W_TILING_KEY) {
        ubFactorElement =
            FloorAlign(FloorAlign(canUseUbSize / divideUbNum / SMALL_W_LIMIT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else {
        ubFactorElement = FloorAlign(canUseUbSize / divideUbNum, ALIGN_256_BYTES) / dataTypeLen;
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::FillTilingData(TilingData* tilingData)
{
    tilingData->batch = batch;
    tilingData->channel = channel;
    tilingData->height = height;
    tilingData->width = width;
    tilingData->alignHeight = alignHeight;
    tilingData->alignWidth = alignWidth;
    tilingData->outHeight = outHeight;
    tilingData->outWidth = outWidth;
    tilingData->alignOutHeight = alignOutHeight;
    tilingData->alignOutWidth = alignOutWidth;
    tilingData->hPad1 = hPad1;
    tilingData->hPad2 = hPad2;
    tilingData->wPad1 = wPad1;
    tilingData->wPad2 = wPad2;
    tilingData->blockNum = usedCoreNum;
    tilingData->ubFactorElement = ubFactorElement;
    tilingData->ncPerCore = ncPerCore;
    tilingData->tailNC = tailNC;
    tilingData->tilingKey = tilingKey;
    if (tilingKey == FLOAT_NO_W_PAD_TILING_KEY || tilingKey == FLOAT16_NO_W_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_W_PAD_TILING_KEY || tilingKey == FLOAT_MINI_SHAPE_TILING_KEY ||
        tilingKey == FLOAT16_MINI_SHAPE_TILING_KEY || tilingKey == BFLOAT16_MINI_SHAPE_TILING_KEY) {
        workspacePerCore = 0;
    } else if (
        tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        wPadCopyCount = CeilAlign(CONST_VALUE_2 * std::max(wPad1, wPad2) + 1, elementsPerBlock);
        int32_t partWs = wPad1 < W_PAD_LOWER_LIMIT && wPad2 < W_PAD_LOWER_LIMIT ? CAL_COUNT : wPadCopyCount;
        workspacePerCore = CONST_VALUE_2 * partWs * dataTypeSize;
    } else {
        workspacePerCore = alignWidth * WORK_SPACE_PART * dataTypeSize;
    }
    tilingData->workspacePerCore = workspacePerCore;
    tilingData->wPadCopyCount = wPadCopyCount;
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetTiling(TilingData* tilingData)
{
    GetTilingKey();
    GetUsedCore();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetPadV3GradV2Tiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize)
{
    class PadV3GradV2Tiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}
} // namespace optiling