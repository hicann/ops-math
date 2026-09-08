/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DENSE_BINCOUNT_REGBASE_H
#define DENSE_BINCOUNT_REGBASE_H

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "dense_bincount_tiling_data.h"

namespace NsDenseBincount {
using namespace AscendC;

constexpr uint32_t UB_BLOCK_BYTES = 32U;

template <uint32_t THREAD_NUM>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void InitializeOutput(uint32_t coreIdx, uint32_t coreNum,
                                                                             __gm__ float* output, int64_t outputSize)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * static_cast<int64_t>(blockDim.x) +
                          static_cast<int64_t>(threadIdx.x);
    const int64_t stride = static_cast<int64_t>(coreNum) * static_cast<int64_t>(blockDim.x);
    for (int64_t index = first; index < outputSize; index += stride) {
        output[index] = 0.0F;
    }
}

template <bool PRIVATE, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterOne(__gm__ float* output, __ubuf__ float* histogram, int64_t outputOffset,
                                       int64_t inputOffset, __gm__ float* weights)
{
    if constexpr (PRIVATE) {
        if constexpr (BINARY_OUTPUT) {
            (void)asc_atomic_exch(histogram + outputOffset, 1.0F);
        } else if constexpr (HAS_WEIGHTS) {
            (void)asc_atomic_add(histogram + outputOffset, weights[inputOffset]);
        } else {
            (void)asc_atomic_add(histogram + outputOffset, 1.0F);
        }
    } else {
        if constexpr (BINARY_OUTPUT) {
            (void)asc_atomic_exch(output + outputOffset, 1.0F);
        } else if constexpr (HAS_WEIGHTS) {
            (void)asc_atomic_add(output + outputOffset, weights[inputOffset]);
        } else {
            (void)asc_atomic_add(output + outputOffset, 1.0F);
        }
    }
}

template <typename T, uint32_t THREAD_NUM, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS, bool PRIVATE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterValues(
    uint32_t coreIdx, uint32_t coreNum, __gm__ T* input, __gm__ float* weights, __gm__ float* output,
    __ubuf__ float* histogram, int64_t numValues, int64_t inputRows, int64_t inputCols, int64_t size)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * static_cast<int64_t>(blockDim.x) +
                          static_cast<int64_t>(threadIdx.x);
    const int64_t stride = static_cast<int64_t>(coreNum) * static_cast<int64_t>(blockDim.x);
    for (int64_t index = first; index < numValues; index += stride) {
        int64_t bin = static_cast<int64_t>(input[index]);
        int64_t row = 0;
        if constexpr (!IS_1D) {
            row = index / inputCols;
            if (bin < 0) {
                const int64_t quotient = bin / size;
                const int64_t remainder = bin % size;
                const int64_t adjustment = remainder == 0 ? 0 : 1;
                if (quotient < adjustment - row) {
                    continue;
                }
                row += quotient - adjustment;
                bin = remainder == 0 ? 0 : remainder + size;
            }
        }
        if (bin < 0 || bin >= size || row < 0 || row >= inputRows) {
            continue;
        }
        const int64_t outputOffset = IS_1D ? bin : row * size + bin;
        ScatterOne<PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>(output, histogram, outputOffset, index, weights);
    }
}

template <typename T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterBinaryRow(uint32_t coreIdx, uint32_t coresPerRow,
                                                                             __gm__ T* input, __gm__ float* output,
                                                                             __ubuf__ float* histogram,
                                                                             int64_t inputRows, int64_t inputCols,
                                                                             int64_t size)
{
    const uint32_t row = coreIdx / coresPerRow;
    const uint32_t coreInRow = coreIdx - row * coresPerRow;
    const int64_t rowBegin = static_cast<int64_t>(row) * inputCols;
    const int64_t firstCol = static_cast<int64_t>(coreInRow) * static_cast<int64_t>(blockDim.x) +
                             static_cast<int64_t>(threadIdx.x);
    const int64_t stride = static_cast<int64_t>(coresPerRow) * static_cast<int64_t>(blockDim.x);
    for (int64_t col = firstCol; col < inputCols; col += stride) {
        const int64_t inputOffset = rowBegin + col;
        int64_t bin = static_cast<int64_t>(input[inputOffset]);
        if (bin >= 0 && bin < size) {
            (void)asc_atomic_exch(histogram + bin, 1.0F);
            continue;
        }
        if (bin >= 0) {
            continue;
        }
        int64_t targetRow = static_cast<int64_t>(row);
        const int64_t quotient = bin / size;
        const int64_t remainder = bin % size;
        const int64_t adjustment = remainder == 0 ? 0 : 1;
        if (quotient < adjustment - targetRow) {
            continue;
        }
        targetRow += quotient - adjustment;
        bin = remainder == 0 ? 0 : remainder + size;
        if (targetRow >= 0 && targetRow < inputRows) {
            (void)asc_atomic_exch(output + targetRow * size + bin, 1.0F);
        }
    }
}

template <typename T, typename S, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
class DenseBincountRegbase {
public:
    __aicore__ inline DenseBincountRegbase(const DenseBincountTilingData& tiling, TPipe* pipe)
        : tiling_(tiling), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR size, GM_ADDR weights, GM_ADDR output)
    {
        input_ = reinterpret_cast<__gm__ T*>(input);
        size_ = reinterpret_cast<__gm__ S*>(size);
        weights_ = reinterpret_cast<__gm__ float*>(weights);
        output_ = reinterpret_cast<__gm__ float*>(output);
    }

    template <uint32_t THREAD_NUM>
    __aicore__ inline void ProcessPrivateHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t size)
    {
        const uint32_t histogramBytes = tiling_.privateHistElems * static_cast<uint32_t>(sizeof(float));
        const uint32_t bufferBytes = ((histogramBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES) * UB_BLOCK_BYTES;
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe_->InitBuffer(histogramQueue, 1, bufferBytes);
        LocalTensor<float> histogram = histogramQueue.AllocTensor<float>();
        Duplicate(histogram, 0.0F, tiling_.privateHistElems);
        asc_vf_call<ScatterValues<T, THREAD_NUM, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS, true>>(
            dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, output_,
            reinterpret_cast<__ubuf__ float*>(histogram.GetPhyAddr()), tiling_.numValues, tiling_.inputRows,
            tiling_.inputCols, size);
        histogramQueue.EnQue(histogram);
        histogram = histogramQueue.DeQue<float>();
        if constexpr (BINARY_OUTPUT) {
            SetAtomicMax<float>();
        } else {
            SetAtomicAdd<float>();
        }
        GlobalTensor<float> outputGlobal;
        outputGlobal.SetGlobalBuffer(output_);
        const DataCopyExtParams copyParams{static_cast<uint16_t>(1), histogramBytes, 0U, 0U, 0U};
        DataCopyPad(outputGlobal, histogram, copyParams);
        SetAtomicNone();
        histogramQueue.FreeTensor(histogram);
    }

    template <uint32_t THREAD_NUM>
    __aicore__ inline void ProcessBinaryRowHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t size)
    {
        const uint32_t coresPerRow = coreNum / static_cast<uint32_t>(tiling_.inputRows);
        const uint32_t activeCoreNum = coresPerRow * static_cast<uint32_t>(tiling_.inputRows);
        if (coreIdx >= activeCoreNum) {
            return;
        }
        const uint32_t histogramBytes = static_cast<uint32_t>(size) * static_cast<uint32_t>(sizeof(float));
        const uint32_t bufferBytes = ((histogramBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES) * UB_BLOCK_BYTES;
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe_->InitBuffer(histogramQueue, 1, bufferBytes);
        LocalTensor<float> histogram = histogramQueue.AllocTensor<float>();
        Duplicate(histogram, 0.0F, static_cast<uint32_t>(size));
        asc_vf_call<ScatterBinaryRow<T, THREAD_NUM>>(dim3(THREAD_NUM), coreIdx, coresPerRow, input_, output_,
                                                     reinterpret_cast<__ubuf__ float*>(histogram.GetPhyAddr()),
                                                     tiling_.inputRows, tiling_.inputCols, size);
        histogramQueue.EnQue(histogram);
        histogram = histogramQueue.DeQue<float>();
        SetAtomicMax<float>();
        GlobalTensor<float> outputGlobal;
        const uint32_t row = coreIdx / coresPerRow;
        outputGlobal.SetGlobalBuffer(output_ + static_cast<int64_t>(row) * size);
        const DataCopyExtParams copyParams{static_cast<uint16_t>(1), histogramBytes, 0U, 0U, 0U};
        DataCopyPad(outputGlobal, histogram, copyParams);
        SetAtomicNone();
        histogramQueue.FreeTensor(histogram);
    }

    __aicore__ inline void Process()
    {
        const int64_t size = static_cast<int64_t>(size_[0]);
        if (size <= 0) {
            return;
        }
        const int64_t outputSize = IS_1D ? size : tiling_.inputRows * size;
        constexpr uint32_t THREAD_NUM = sizeof(T) == sizeof(int64_t) ? 512U : 1024U;
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t launchCoreNum = static_cast<uint32_t>(GetBlockNum());
        const uint32_t coreNum = tiling_.usedCoreNum < launchCoreNum ? tiling_.usedCoreNum : launchCoreNum;
        const bool participates = coreIdx < coreNum;
        if (participates) {
            asc_vf_call<InitializeOutput<THREAD_NUM>>(dim3(THREAD_NUM), coreIdx, coreNum, output_, outputSize);
        }
        SyncAll();
        if (!participates) {
            return;
        }
        const bool useBinaryRowHistogram = !IS_1D && BINARY_OUTPUT && tiling_.inputRows > 1 &&
                                           tiling_.privateHistElems == static_cast<uint32_t>(size);
        if (useBinaryRowHistogram) {
            ProcessBinaryRowHistogram<THREAD_NUM>(coreIdx, coreNum, size);
        } else if (tiling_.privateHistElems > 0U) {
            ProcessPrivateHistogram<THREAD_NUM>(coreIdx, coreNum, size);
        } else {
            asc_vf_call<ScatterValues<T, THREAD_NUM, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS, false>>(
                dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, output_, (__ubuf__ float*)nullptr,
                tiling_.numValues, tiling_.inputRows, tiling_.inputCols, size);
        }
    }

private:
    const DenseBincountTilingData& tiling_;
    TPipe* pipe_ = nullptr;
    __gm__ T* input_ = nullptr;
    __gm__ S* size_ = nullptr;
    __gm__ float* weights_ = nullptr;
    __gm__ float* output_ = nullptr;
};
} // namespace NsDenseBincount

#endif
