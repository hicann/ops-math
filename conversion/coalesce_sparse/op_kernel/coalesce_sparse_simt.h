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
 * \file coalesce_sparse_simt.h
 * \brief
 */
#ifndef COALESCE_SPARSE_SIMT_H
#define COALESCE_SPARSE_SIMT_H

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"

using namespace AscendC;
constexpr int64_t THREAD_NUM = 512;

template <typename uIdxType, typename idxType, typename dataType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ValueAtomicAdd(__gm__ uIdxType* uniqueIndices, __gm__ idxType* indices, __gm__ dataType* values, __gm__ idxType* newIndices, __gm__ dataType* newValue, uint32_t taskLen, uint64_t m, uint64_t valueSize)
{
    for(uint32_t index = threadIdx.x; index < taskLen; index += blockDim.x) {
        uint32_t outputIdx = *(uniqueIndices + index);
        __gm__ idxType* indices_base = (__gm__ idxType*)indices + index * m;
        __gm__ dataType* values_base = (__gm__ dataType*)values + index * valueSize;
        __gm__ idxType* newIndices_base = (__gm__ idxType*)newIndices + outputIdx * m;
        __gm__ dataType* newValue_base = (__gm__ dataType*)newValue + outputIdx * valueSize;
        for (uint32_t m_iter = 0; m_iter < m; m_iter++) {
            *(newIndices_base) = *(indices_base);
            indices_base++;
            newIndices_base++;
        }
        for (uint32_t v_iter = 0; v_iter < valueSize; v_iter++) {
            asc_atomic_add(newValue_base, *(values_base));
            values_base++;
            newValue_base++;
        }
    }
}

template <typename uIdxType, typename idxType, typename dataType>
class KernelCoalesceSparseSimt {
public:
    __aicore__ inline KernelCoalesceSparseSimt() = default;
    __aicore__ inline void Init(
        GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices, GM_ADDR newValue,
        const CoalesceSparseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTilingValue(const CoalesceSparseTilingData* __restrict tilingData);

private:
    __gm__ uIdxType* uniqueIndicesAddr = nullptr;
    __gm__ idxType* indicesAddr = nullptr;
    __gm__ dataType* valuesAddr = nullptr;
    __gm__ idxType* newIndicesAddr = nullptr;
    __gm__ dataType* newValueAddr = nullptr;

    uint64_t usedCoreNum{0};
    uint64_t m{0};
    uint64_t valueSize{0};
    uint64_t taskNum{0};
    uint64_t taskTail{0};

    uint64_t taskLen{0};
    const CoalesceSparseTilingData* __restrict tilingDevice{nullptr};
};

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparseSimt<uIdxType, idxType, dataType>::Init(
    GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices, GM_ADDR newValue,
    const CoalesceSparseTilingData* __restrict tilingData)
{
    InitTilingValue(tilingData);
    uint64_t coreId = GetBlockIdx();
    uint64_t beginOffset = coreId * taskNum;
    uint64_t indicesBeginOffset = beginOffset * m;
    uint64_t valueBeginOffset = beginOffset * valueSize;

    if (coreId < usedCoreNum - 1) {
        taskLen = taskNum;
    } else if (coreId == usedCoreNum - 1) {
        taskLen = taskTail;
    }

    uniqueIndicesAddr = (__gm__ uIdxType*)uniqueIndices + beginOffset;
    indicesAddr = (__gm__ idxType*)indices + indicesBeginOffset;
    valuesAddr = (__gm__ dataType*)values + valueBeginOffset;
    newIndicesAddr = (__gm__ idxType*)newIndices;
    newValueAddr = (__gm__ dataType*)newValue;
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparseSimt<uIdxType, idxType, dataType>::InitTilingValue(
    const CoalesceSparseTilingData* __restrict tilingData)
{
    // Get tilingData
    this->tilingDevice = tilingData;
    usedCoreNum = tilingDevice->usedCoreNum;
    m = tilingDevice->m;
    valueSize = tilingDevice->valueSize;
    taskNum = tilingDevice->taskNum;
    taskTail = tilingDevice->taskTail;
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparseSimt<uIdxType, idxType, dataType>::Process()
{
    if (taskLen == 0) {
        return;
    }

    asc_vf_call<ValueAtomicAdd<uIdxType, idxType, dataType>>(
        dim3{THREAD_NUM, 1, 1},
        uniqueIndicesAddr,
        indicesAddr,
        valuesAddr,
        newIndicesAddr,
        newValueAddr,
        taskLen, m, valueSize);
}

#endif // COALESCE_SPARSE_SIMT