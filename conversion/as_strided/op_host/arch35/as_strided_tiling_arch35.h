/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_AS_STRIDED_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_AS_STRIDED_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {

const int64_t TILING_ARRAY_LEN = 10;
const int64_t TILING_NDDMA_LEN = 5;

BEGIN_TILING_DATA_DEF(AsStridedTilingData)

TILING_DATA_FIELD_DEF(int64_t, storageOffset);        //  input offset
TILING_DATA_FIELD_DEF(uint32_t, blockNum);            //  Tne number of actual using block
TILING_DATA_FIELD_DEF(uint32_t, loopsTailCore);       //  Loop of UB carry on tail block
TILING_DATA_FIELD_DEF(uint32_t, tilingAxisIdx);       //  The index of the cut axis in the output
TILING_DATA_FIELD_DEF(uint32_t, outerAxisFactor);     //  The value of outer axis
TILING_DATA_FIELD_DEF(uint32_t, innerAxisFactor);     //  The value of inner axis
TILING_DATA_FIELD_DEF(uint32_t, outerAxisNum);        //  The number of outer axis
TILING_DATA_FIELD_DEF(uint32_t, innerAxisNum);        //  The number of inner axis
TILING_DATA_FIELD_DEF(uint32_t, loopsPerCore);        //  The loop number of every core
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);            //  The size of the UB used
TILING_DATA_FIELD_DEF(uint32_t, ubFactorTail);        //  The UB size of the tail
TILING_DATA_FIELD_DEF(uint32_t, ubSize);              //  The size of the total UB
TILING_DATA_FIELD_DEF(uint32_t, innerAxisFactorTail); //  The value of inner tail axis
TILING_DATA_FIELD_DEF(uint32_t, axisOutTotalFactor);
TILING_DATA_FIELD_DEF(uint32_t, en32BAligned);
//  The specific value of each axis within the UB
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN, innerAxis);
//  The stride corresponds to each axis outside the UB
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN, outStrideArr);
//  The number of loops on the outer axis of the UB
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN, outLoopArr);
//  The number of loops in each dimension of NDDMA
TILING_DATA_FIELD_DEF_ARR(uint32_t, TILING_NDDMA_LEN, nddmaLoop);
//  The number of loops in each dimension of NDDMA on tail
TILING_DATA_FIELD_DEF_ARR(uint32_t, TILING_NDDMA_LEN, nddmaTailLoop);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_NDDMA_LEN, nddmaSrcStride); //  NDDMA src stride
TILING_DATA_FIELD_DEF_ARR(uint32_t, TILING_NDDMA_LEN, nddmaDstStride); //  NDDMA dst stride
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN, gmOutStride);
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN, gmShape);
TILING_DATA_FIELD_DEF_ARR(int32_t, TILING_ARRAY_LEN, gmInStride);

END_TILING_DATA_DEF;

ge::graphStatus TilingForAsStridedOfAsc(gert::TilingContext* context, uint32_t maxCoreNum, uint32_t ubSizePlatform);
REGISTER_TILING_DATA_CLASS(AsStrided, AsStridedTilingData)

struct AsStridedTilingParam {
    int64_t storageOffset = 0;
    uint32_t ubFactor = 0;
    uint32_t ubFactorTail = 0;
    uint32_t preSize = 1;
    uint32_t numCore = 0;
    uint32_t innerAxisFactor = 1;
    uint32_t outerAxisFactor = 0;
    uint32_t tilingFlag = 0;
    uint32_t tilingAxisIdx = 0;
    uint32_t curAxisFactor = 0;
    uint32_t outerAxisNum = 0;
    uint32_t innerAxisNum = 0;
    uint32_t innerAxisFactorTail = 0;
    uint32_t blockNum = 0;
    uint32_t axisOutTotalFactor = 0;
    uint32_t ubSize = 0;
    uint64_t ubSizePlatForm = 0;
    uint32_t sizeofDtype = 0;
    uint32_t loopsPerCore = 0;
    uint32_t ubUseFactor = 0;
    int64_t tilingKey = 0;
    uint32_t en32BAligned = 0;
    bool movealignFlag = false;
    bool dualCutFlag = false;
    int32_t innerAxis[TILING_ARRAY_LEN] = {0};
    uint32_t nddmaDstStride[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaLoop[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaTailLoop[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    int32_t outLoopArr[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t nddmaSrcStride[TILING_NDDMA_LEN] = {0};
    int32_t outStrideArr[TILING_ARRAY_LEN] = {0};
    int32_t gmOutStride[TILING_ARRAY_LEN] = {0};
    int32_t gmShape[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int32_t gmInStride[TILING_ARRAY_LEN] = {0};
};

} // namespace optiling

#endif