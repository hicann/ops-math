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

/*!
 * \file is_finite_tiling_common.h
 * \brief
 */

#ifndef IS_FINITE_TILING_H
#define IS_FINITE_TILING_H

#include "../op_kernel/is_finite_struct.h"
#include "torch_extension/tiling_utils.h"
#include "platform/platform_ascendc.h"

namespace IsFiniteNs {

class IsFiniteTiling {
  public:
    constexpr static int64_t MINIMUM_ELEMENT_PER_CORE = 32;
    constexpr static int64_t DATA_BLOCK = 32;
    constexpr static int64_t RESERVERD_UB_SIZE = 1024;
    constexpr static int64_t UB_DIVIDER_FOR_TMP_CASTING = 10;

    template<typename T>
    static void IsFiniteCommonTiling(T x, IsFiniteTilingData& tilingData, uint32_t coreNum, uint64_t ubSize) {
      int64_t elementCount = 1;

      for(uint16_t i = 0; i < TilingUtils::GetDimNum(x); i++) {
        elementCount *= TilingUtils::GetDim(x, i);
      }

      uint32_t blockDim = (elementCount + MINIMUM_ELEMENT_PER_CORE -1) / MINIMUM_ELEMENT_PER_CORE;
      if (blockDim > coreNum) {
        blockDim = coreNum;
      }

      uint32_t dataBlockSize = DATA_BLOCK * sizeof(T);
      uint32_t usableUbSize = uint32_t(ubSize - RESERVERD_UB_SIZE - sizeof(IsFiniteTilingData)) / UB_DIVIDER_FOR_TMP_CASTING;
      usableUbSize = usableUbSize / dataBlockSize * dataBlockSize;

      uint64_t perCoreDataCount = elementCount / blockDim;
      perCoreDataCount = perCoreDataCount / DATA_BLOCK * DATA_BLOCK;
      uint64_t tempTailDataCount = elementCount -perCoreDataCount * blockDim;
      uint64_t tailDataCoreNum = tempTailDataCount / DATA_BLOCK;
      uint64_t lastCoreDataCount = perCoreDataCount + tempTailDataCount % DATA_BLOCK;

      tilingData.usableUbSize = usableUbSize;
      tilingData.needCoreNum = blockDim;
      tilingData.totalDataCount = elementCount;
      tilingData.perCoreDataCount = perCoreDataCount;
      tilingData.tailDataCoreNum = tailDataCoreNum;
      tilingData.lastCoreDataCount = lastCoreDataCount;
    }
};

}
#endif // IS_FINITE_TILING_H