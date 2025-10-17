/**	
 * This program is free software, you can redistribute it and/or modify it.	
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.	
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").	
 * Please refer to the License for details. You may not use this file except in compliance with the License.	
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING	
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.	
 * See LICENSE in the root of the software repository for the full text of the License.	
 */
#ifndef _DIAG_FLAT_TILING_H_
#define _DIAG_FLAT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct DiagV2TilingDataTest {
  int64_t xWidth = 0;
  int64_t xHeight = 0;
  int64_t gmOffset = 0;
  int64_t numOut = 0;
  int64_t realCoreNum = 0;
  int64_t numPerCore = 0;
  int64_t tailNum = 0;
  int64_t tilingKey = 0;
  int64_t matrixRowLength = 0;
  int64_t inputNum = 0;
  int64_t usedCoreNum = 0;
  int64_t totalCoreNum = 0;
  int64_t normalCoreHandleNum = 0;
  int64_t lastCoreHandleNum = 0;
  int64_t diagonal = 0;
  int64_t align2 = 0;
  int64_t align3 = 0;
  int64_t align4 = 0;
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
  DiagV2TilingData tilingData;                                               \
  INIT_TILING_DATA(DiagV2TilingData, tilingDataPointer, tilingPointer);      \
  (tilingData).xWidth = tilingDataPointer->xWidth;                           \
  (tilingData).xHeight = tilingDataPointer->xHeight;                         \
  (tilingData).gmOffset = tilingDataPointer->gmOffset;                       \
  (tilingData).numOut = tilingDataPointer->numOut;                           \
  (tilingData).realCoreNum = tilingDataPointer->realCoreNum;                 \
  (tilingData).numPerCore = tilingDataPointer->numPerCore;                   \
  (tilingData).tailNum = tilingDataPointer->tailNum;                         \
  (tilingData).matrixRowLength = tilingDataPointer->matrixRowLength;         \
  (tilingData).inputNum = tilingDataPointer->inputNum;                       \
  (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                 \
  (tilingData).totalCoreNum = tilingDataPointer->totalCoreNum;               \
  (tilingData).normalCoreHandleNum = tilingDataPointer->normalCoreHandleNum; \
  (tilingData).lastCoreHandleNum = tilingDataPointer->lastCoreHandleNum;     \
  (tilingData).diagonal = tilingDataPointer->diagonal;

#endif