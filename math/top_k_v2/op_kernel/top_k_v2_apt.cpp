/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file top_k_v2_apt.cpp
 * \brief top k v2 impl
 */

#include "arch35/radix_sort_top_k.h"
#include "arch35/radix_topk_constant.h"
#include "arch35/top_k_merge_sort.h"
#include "arch35/radix_sort_top_k_single_block.h"
#include "arch35/radix_sort_top_k_single_core.h"
#include "arch35/radix_sort_top_k_inter_core_template_optimization.h"

using namespace AscendC;

const uint32_t SINGLE_CORE_MODE = 1;
const uint32_t MULT_CORE_OPTIM_MODE = 4;

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, true, true, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        } else {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, true, false, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        }
    } else {
        if (isSort) {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, false, true, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        } else {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, false, false, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleCoreOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, true, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        } else {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, true, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        }
    } else {
        if (isSort) {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, false, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        } else {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, false, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        }
    }
}

template <typename T, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKMultiCoreOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopKMultiCoreOptimization<T, true, true, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        } else {
            RadixSortTopKMultiCoreOptimization<T, true, false, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        }
    } else {
        if (isSort) {
            RadixSortTopKMultiCoreOptimization<T, false, true, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        } else {
            RadixSortTopKMultiCoreOptimization<T, false, false, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlockOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, true, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        } else {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, true, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        }
    } else {
        if (isSort) {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, false, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        } else {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, false, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX_TO>
__aicore__ inline void generateOpObject(
  GM_ADDR x,
  GM_ADDR k,
  GM_ADDR values,
  GM_ADDR indices,
  GM_ADDR globalWorkGm,
  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    bool isSingleBlock = (tilingData.lastDimNeedCore == 1) ? true : false;
    bool isSingleCore = (tilingData.modeType == SINGLE_CORE_MODE) ? true : false;
    bool isInInt32Range = (tilingData.isInInt32Range > 0) ? true : false;
    bool isMultiCoreOptimMode = (tilingData.modeType == MULT_CORE_OPTIM_MODE) ? true : false;

    // 核内模板
    if (isSingleBlock) {
        if (isInInt32Range) {
            RadixSortTopKSingleBlockOpObject<T, UNSINGED_TYPE, NUM_PASS, int32_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        } else {
            RadixSortTopKSingleBlockOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    // 单核多次处理模板（930新增模板用于性能优化）
    if (isSingleCore) {
        if (isInInt32Range) {
            RadixSortTopKSingleCoreOpObject<T, UNSINGED_TYPE, NUM_PASS, int32_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        } else {
            RadixSortTopKSingleCoreOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    if (isMultiCoreOptimMode) {
        if (isInInt32Range) {
            RadixSortTopKMultiCoreOpObject<T, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    // 多核处理排序轴模板（老模板）
    if (isInInt32Range) {
        RadixSortTopKOpObject<T, UNSINGED_TYPE, NUM_PASS, int32_t, T_INDEX_TO>(
            x, k, values, indices, globalWorkGm, tiling);
    } else {
        RadixSortTopKOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
            x, k, values, indices, globalWorkGm, tiling);
    }
}

template <typename T, typename CONVERT_TYPE, typename INDEX_DTYPE>
__aicore__ inline void generateMergeTopKObject(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    TPipe pipe;
    if (isLargest) {
        MergeSort<T, CONVERT_TYPE, TopKV2TilingDataSimd, true, INDEX_DTYPE> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    } else {
        MergeSort<T, CONVERT_TYPE, TopKV2TilingDataSimd, false, INDEX_DTYPE> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    }
}

extern "C" __global__ __aicore__ void top_k_v2(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
  if (workspace == nullptr) {
      return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR globalWorkGm = GetUserWorkspace(workspace);
  if (globalWorkGm == nullptr) {
      return;
  }
  if (TILING_KEY_IS(1004)) {
      generateOpObject<int64_t, uint64_t, B64_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(1003)) {
      generateOpObject<int32_t, uint32_t, B32_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(1002)) {
      generateOpObject<int16_t, uint16_t, B16_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(2004)) {
      generateOpObject<uint64_t, uint64_t, B64_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(2003)) {
      generateOpObject<uint32_t, uint32_t, B32_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(2002)) {
      generateOpObject<uint16_t, uint16_t, B16_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(3003)) {
      generateOpObject<float, uint32_t, B32_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(3002)) {
      generateOpObject<half, uint16_t, B16_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(4002)) {
      generateOpObject<bfloat16_t, uint16_t, B16_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(2001)) {
      generateOpObject<uint8_t, uint8_t, B8_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(1001)) {
      generateOpObject<int8_t, uint8_t, B8_BITE_SIZE, DTYPE_INDICES>(
          x,
          k,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(13003)) {
      generateMergeTopKObject<float, float, DTYPE_INDICES>(
          x,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(13002)) {
      generateMergeTopKObject<half, half, DTYPE_INDICES>(
          x,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  } else if (TILING_KEY_IS(14002)) {
      generateMergeTopKObject<bfloat16_t, float, DTYPE_INDICES>(
          x,
          values,
          indices,
          globalWorkGm,
          tiling
      );
  }
}
