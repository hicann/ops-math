/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef _GE_EXP_TILING_H_
 #define _GE_EXP_TILING_H_
 
 #include <cstdint>
 #include <cstring>
 #include "kernel_tiling/kernel_tiling.h"
 
 #define __CCE_UT_TEST__
 
 #pragma pack(1)
 
 struct EleBaseTilingData {
     uint64_t scheMode;
     int64_t dim0;
     int64_t blockFormer;
     int64_t blockNum;
     int64_t ubFormer;
     int64_t ubLoopOfFormerBlock;
     int64_t ubLoopOfTailBlock;
     int64_t ubTailOfFormerBlock;
     int64_t ubTailOfTailBlock;
     int64_t elemNum;
 };
 
 struct IsInfRegbaseTilingData {
     EleBaseTilingData baseTiling;
 };
 
 #pragma pack()
 
 inline void InitTilingData(uint8_t* tiling, IsInfRegbaseTilingData* const_data)
 {
     memcpy(const_data, tiling, sizeof(IsInfRegbaseTilingData));
 }
 
 #define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
     tiling_struct tiling_data;                                              \
     InitTilingData(tiling_arg, &tiling_data)
 
 #define GET_TILING_DATA(tiling_data, tiling_arg) \
     IsInfRegbaseTilingData tiling_data;            \
     InitTilingData(tiling_arg, &tiling_data)
 
 #endif