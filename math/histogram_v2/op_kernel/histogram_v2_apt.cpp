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
 * \file histogram_v2_apt.cpp
 * \brief
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "v35/histogram_v2_simt_full_load.h"
#include "v35/histogram_v2_simt_not_full_load.h"
#include "v35/histogram_v2_simt_not_full_load_simt.h"

using namespace HistogramV2SIMT;

#define TILING_KEY_UB_FULL_FP32 101
#define TILING_KEY_UB_FULL_INT32 102
#define TILING_KEY_UB_FULL_INT8 103
#define TILING_KEY_UB_FULL_UINT8 104
#define TILING_KEY_UB_FULL_INT16 105
#define TILING_KEY_UB_FULL_INT64 106
#define TILING_KEY_UB_FULL_FP16 107

#define TILING_KEY_UB_NOT_FULL_FP32 201
#define TILING_KEY_UB_NOT_FULL_INT32 202
#define TILING_KEY_UB_NOT_FULL_INT8 203
#define TILING_KEY_UB_NOT_FULL_UINT8 204
#define TILING_KEY_UB_NOT_FULL_INT16 205
#define TILING_KEY_UB_NOT_FULL_INT64 206
#define TILING_KEY_UB_NOT_FULL_FP16 207

#define TILING_KEY_UB_NOT_FULL_SIMT_FP32 301
#define TILING_KEY_UB_NOT_FULL_SIMT_INT32 302
#define TILING_KEY_UB_NOT_FULL_SIMT_INT8 303
#define TILING_KEY_UB_NOT_FULL_SIMT_UINT8 304
#define TILING_KEY_UB_NOT_FULL_SIMT_INT16 305
#define TILING_KEY_UB_NOT_FULL_SIMT_INT64 306
#define TILING_KEY_UB_NOT_FULL_SIMT_FP16 307

extern "C" __global__ __aicore__ void histogram_v2(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    AscendC::TPipe tpipe;
    if (TILING_KEY_IS(TILING_KEY_UB_FULL_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<float, float> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_FULL_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<int32_t, int64_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_FULL_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<int8_t, int32_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_FULL_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<uint8_t, int32_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_FULL_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<int16_t, int32_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_FULL_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<int64_t, int64_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_FULL_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtFullLoad<half, float> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<float, float> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<int32_t, int64_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<int8_t, int32_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<uint8_t, int32_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<int16_t, int32_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<int64_t, int64_t> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoad<half, float> op;
        op.Init(x, min, max, y, tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<float, float> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<int32_t, int64_t> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<int8_t, int32_t> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<uint8_t, int32_t> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<int16_t, int32_t> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<int64_t, int64_t> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_UB_NOT_FULL_SIMT_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
        const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        HistogramV2SIMT::HistogramV2SimtNotFullLoadGmAtomicAdd<half, float> op;
        op.Init(x, min, max, y, tilingData);
        op.Process();
    }
}
