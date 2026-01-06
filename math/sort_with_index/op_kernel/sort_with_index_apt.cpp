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
 * \file sort_with_index.cpp
 * \brief sort_with_index cc impl
 */

#include "arch35/radix_sort_with_index_simd_entry.h"
#include "arch35/merge_sort_with_index.h"

using namespace AscendC;

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS>
__aicore__ inline void generateOpObject(GM_ADDR x, GM_ADDR index, GM_ADDR sortedX, GM_ADDR sortedIndex, GM_ADDR globalWorkGm,
                                        GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    bool isDescend = tilingData.isDescend > 0 ? true : false;
    if (isDescend) {
        RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, true, int32_t, int32_t> radixSort;
        radixSort.Init(x, index, sortedX, sortedIndex, globalWorkGm, &tilingData);
        radixSort.ProcessSort();
    } else {
        RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, false, int32_t, int32_t> radixSort;
        radixSort.Init(x, index, sortedX, sortedIndex, globalWorkGm, &tilingData);
        radixSort.ProcessSort();
    }
}

template <typename T, typename CONVERT_TYPE>
__aicore__ inline void generateMergeSortObject(GM_ADDR x, GM_ADDR index, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm,
                                               GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isDescend = (tilingData.isDescend > 0) ? true : false;
    TPipe pipe;
    if (isDescend) {
        MergeSortWithIndex<T, CONVERT_TYPE, SortWithIndexTilingDataSimt, true, int32_t> mergeSort;
        mergeSort.Init(x, index, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    } else {
        MergeSortWithIndex<T, CONVERT_TYPE, SortWithIndexTilingDataSimt, false, int32_t> mergeSort;
        mergeSort.Init(x, index, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    }
}

extern "C" __global__ __aicore__ void sort_with_index(GM_ADDR x, GM_ADDR index, GM_ADDR sortedX, GM_ADDR sortedIndex,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR globalWorkGm = GetUserWorkspace(workspace);
    if (globalWorkGm == nullptr) {
        return;
    }
    if (TILING_KEY_IS(1001)) {
        generateOpObject<int8_t, int8_t, uint8_t, B8_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(1002)) {
        generateOpObject<int16_t, int16_t, uint16_t, B16_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                    tiling);
    } else if (TILING_KEY_IS(1003)) {
        generateOpObject<int32_t, int32_t, uint32_t, B32_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                    tiling);
    } else if (TILING_KEY_IS(1004)) {
        generateOpObject<int64_t, int64_t, uint64_t, B64_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                    tiling);
    } else if (TILING_KEY_IS(2001)) {
        generateOpObject<uint8_t, uint8_t, uint8_t, B8_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(2004)) {
        generateOpObject<uint64_t, uint64_t, uint64_t, B64_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                      tiling);
    } else if (TILING_KEY_IS(2003)) {
        generateOpObject<uint32_t, uint32_t, uint32_t, B32_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                      tiling);
    } else if (TILING_KEY_IS(2002)) {
        generateOpObject<uint16_t, uint16_t, uint16_t, B16_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                      tiling);
    } else if (TILING_KEY_IS(3003)) {
        generateOpObject<float, float, uint32_t, B32_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(3002)) {
        generateOpObject<half, half, uint16_t, B16_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(4002)) {
        generateOpObject<bfloat16_t, float, uint16_t, B16_BITE_SIZE>(x, index, sortedX, sortedIndex, globalWorkGm,
                                                                     tiling);
    } else if (TILING_KEY_IS(13003)) {
        generateMergeSortObject<float, float>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(13002)) {
        generateMergeSortObject<half, half>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(14002)) {
        generateMergeSortObject<bfloat16_t, float>(x, index, sortedX, sortedIndex, globalWorkGm, tiling);
    }
}