/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ragged_bin_count_simt.h
 * \brief SIMT implementation for RaggedBinCount on DAV_3510.
 */
#ifndef RAGGED_BIN_COUNT_SIMT_H
#define RAGGED_BIN_COUNT_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "ragged_bin_count_tiling_data.h"

namespace NsRaggedBinCount {

using namespace AscendC;

constexpr uint32_t THREAD_NUM_U32 = 1024U;
constexpr uint32_t THREAD_NUM_U64 = 512U;
constexpr uint32_t MAPPING_MODE_ROW = 0U;
constexpr uint32_t MAPPING_MODE_VALUE = 1U;
constexpr uint32_t UB_BLOCK_BYTES = 32U;

template <bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterWrite(__gm__ float* output, int64_t outputOffset, uint64_t valueIndex,
                                         __gm__ float* weights)
{
    if constexpr (BINARY_OUTPUT) {
        // The bin only has to be set once and every writer stores the same
        // value, so reading first is safe and skips most of the atomics: a
        // stale "not set" costs one redundant exchange, and "already set" can
        // only be observed after some thread wrote 1.0F.  Output stays bitwise
        // in {0.0F, 1.0F}.
        //
        // Four variants were measured on the 74 binary cases of the performance
        // set (median NPU time, and how many fall below folded G/N 0.1):
        //
        //   asc_atomic_exch            correct   424 us   51/74 below   <- was here
        //   read + asc_atomic_exch     correct   1.27x faster, 47/74    <- is here
        //   plain store                WRONG     75x faster,   0/32
        //   asc_atomic_or(0x3F800000)  correct   0.6x, i.e. SLOWER, 56/74
        //
        // The plain store is not merely risky, it is measurably wrong: cores
        // write back whole cache lines, so a 16-row x 1-bin output (64 B, 16
        // cores) kept only one core's writes -- 6.25% correct.  The corruption
        // fraction tracks output size exactly, which is the false-sharing
        // signature, and it is why the atomic cannot simply be dropped.
        //
        // So every GM atomic serialises under contention and swapping which
        // atomic is used does not help; the read above is the only cheap win
        // available without restructuring.  ScatterWriteLocal below is the
        // restructuring: when the output fits in UB the host switches the
        // scatter over to it and this path only runs for outputs too large to
        // privatise, where the hits are spread thin enough not to serialise.
        if (output[outputOffset] != 1.0F) {
            (void)asc_atomic_exch(output + outputOffset, 1.0F);
        }
    } else if constexpr (HAS_WEIGHTS) {
        (void)asc_atomic_add(output + outputOffset, weights[valueIndex]);
    } else {
        (void)asc_atomic_add(output + outputOffset, 1.0F);
    }
}

template <bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterWriteLocal(__ubuf__ float* histogram, int64_t outputOffset, uint64_t valueIndex,
                                              __gm__ float* weights)
{
    // Same three cases as ScatterWrite, but against the core's private UB copy of the output. The
    // atomics stay -- the 1024 threads of one core still collide on a bin -- yet they now resolve in
    // core-local SRAM instead of crossing to GM, and no cache line is shared with another core.
    if constexpr (BINARY_OUTPUT) {
        if (histogram[outputOffset] != 1.0F) {
            (void)asc_atomic_exch(histogram + outputOffset, 1.0F);
        }
    } else if constexpr (HAS_WEIGHTS) {
        (void)asc_atomic_add(histogram + outputOffset, weights[valueIndex]);
    } else {
        (void)asc_atomic_add(histogram + outputOffset, 1.0F);
    }
}

template <bool PRIVATE, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterDispatch(__gm__ float* output, __ubuf__ float* histogram, int64_t outputOffset,
                                            uint64_t valueIndex, __gm__ float* weights)
{
    if constexpr (PRIVATE) {
        ScatterWriteLocal<BINARY_OUTPUT, HAS_WEIGHTS>(histogram, outputOffset, valueIndex, weights);
    } else {
        ScatterWrite<BINARY_OUTPUT, HAS_WEIGHTS>(output, outputOffset, valueIndex, weights);
    }
}

__simt_callee__ inline int64_t FindRaggedRow(__gm__ int64_t* splits, int64_t numRows, int64_t valueIndex)
{
    // Find the first row whose end split is greater than valueIndex. Using
    // upper-bound semantics correctly skips empty rows represented by repeated splits.
    int64_t low = 0;
    int64_t high = numRows;
    while (low < high) {
        const int64_t middle = low + ((high - low) >> 1);
        if (splits[middle + 1] <= valueIndex) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low >= numRows) {
        return -1;
    }
    const int64_t rowBegin = splits[low];
    const int64_t rowEnd = splits[low + 1];
    return (rowBegin <= valueIndex && valueIndex < rowEnd) ? low : -1;
}

template <typename INDEX_TYPE, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void InitializeOutputAndFlag(int64_t outputElements,
                                                                                         __gm__ float* output,
                                                                                         __gm__ uint32_t* invalidFlag)
{
    const INDEX_TYPE first = static_cast<INDEX_TYPE>(blockIdx.x) * static_cast<INDEX_TYPE>(blockDim.x) +
                             static_cast<INDEX_TYPE>(threadIdx.x);
    const INDEX_TYPE stride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(gridDim.x);
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        *invalidFlag = 0U;
    }
    for (INDEX_TYPE index = first; index < static_cast<INDEX_TYPE>(outputElements); index += stride) {
        output[index] = 0.0F;
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ValidateInputs(int64_t numSplits, int64_t numValues,
                                                                                __gm__ int64_t* splits,
                                                                                __gm__ VALUE_TYPE* values,
                                                                                __gm__ uint32_t* invalidFlag)
{
    const INDEX_TYPE first = static_cast<INDEX_TYPE>(blockIdx.x) * static_cast<INDEX_TYPE>(blockDim.x) +
                             static_cast<INDEX_TYPE>(threadIdx.x);
    const INDEX_TYPE stride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(gridDim.x);

    for (INDEX_TYPE splitIndex = first; splitIndex < static_cast<INDEX_TYPE>(numSplits); splitIndex += stride) {
        const int64_t split = splits[splitIndex];
        bool invalid = split < 0 || split > numValues;
        invalid = invalid || (splitIndex == 0U && split != 0);
        invalid = invalid || (splitIndex + 1U == static_cast<INDEX_TYPE>(numSplits) && split != numValues);
        if (splitIndex + 1U < static_cast<INDEX_TYPE>(numSplits)) {
            invalid = invalid || split > splits[splitIndex + 1U];
        }
        if (invalid) {
            (void)asc_atomic_or(invalidFlag, 1U);
        }
    }

    for (INDEX_TYPE valueIndex = first; valueIndex < static_cast<INDEX_TYPE>(numValues); valueIndex += stride) {
        if (values[valueIndex] < static_cast<VALUE_TYPE>(0)) {
            (void)asc_atomic_or(invalidFlag, 1U);
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, bool PRIVATE, bool BINARY_OUTPUT,
          bool HAS_WEIGHTS>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ScatterByRow(
    int64_t numRows, int64_t numValues, int64_t numBins, __gm__ int64_t* splits, __gm__ VALUE_TYPE* values,
    __gm__ float* weights, __gm__ float* output, __ubuf__ float* histogram)
{
    for (INDEX_TYPE row = static_cast<INDEX_TYPE>(blockIdx.x); row < static_cast<INDEX_TYPE>(numRows);
         row += static_cast<INDEX_TYPE>(gridDim.x)) {
        const int64_t rowBegin = splits[row];
        const int64_t rowEnd = splits[row + 1];
        if (rowBegin < 0 || rowEnd < rowBegin || rowEnd > numValues) {
            continue;
        }
        const INDEX_TYPE first = static_cast<INDEX_TYPE>(rowBegin) + static_cast<INDEX_TYPE>(threadIdx.x);
        for (INDEX_TYPE valueIndex = first; valueIndex < static_cast<INDEX_TYPE>(rowEnd);
             valueIndex += static_cast<INDEX_TYPE>(blockDim.x)) {
            const int64_t bin = static_cast<int64_t>(values[valueIndex]);
            if (bin >= 0 && bin < numBins) {
                const int64_t outputOffset = static_cast<int64_t>(row) * numBins + bin;
                ScatterDispatch<PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>(output, histogram, outputOffset,
                                                                     static_cast<uint64_t>(valueIndex), weights);
            }
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, bool PRIVATE, bool BINARY_OUTPUT,
          bool HAS_WEIGHTS>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ScatterByValue(
    int64_t numRows, int64_t numValues, int64_t numBins, __gm__ int64_t* splits, __gm__ VALUE_TYPE* values,
    __gm__ float* weights, __gm__ float* output, __ubuf__ float* histogram)
{
    const INDEX_TYPE first = static_cast<INDEX_TYPE>(blockIdx.x) * static_cast<INDEX_TYPE>(blockDim.x) +
                             static_cast<INDEX_TYPE>(threadIdx.x);
    const INDEX_TYPE stride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(gridDim.x);
    for (INDEX_TYPE valueIndex = first; valueIndex < static_cast<INDEX_TYPE>(numValues); valueIndex += stride) {
        const int64_t bin = static_cast<int64_t>(values[valueIndex]);
        if (bin < 0 || bin >= numBins) {
            continue;
        }
        const int64_t row = FindRaggedRow(splits, numRows, static_cast<int64_t>(valueIndex));
        if (row >= 0) {
            const int64_t outputOffset = row * numBins + bin;
            ScatterDispatch<PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>(output, histogram, outputOffset,
                                                                 static_cast<uint64_t>(valueIndex), weights);
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, uint32_t MAPPING_MODE, bool PRIVATE,
          bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__aicore__ inline void LaunchScatter(const RaggedBinCountTilingData* tilingData, __gm__ int64_t* splitsGm,
                                     __gm__ VALUE_TYPE* valuesGm, __gm__ float* weightsGm, __gm__ float* outputGm,
                                     __ubuf__ float* histogram)
{
    if constexpr (MAPPING_MODE == MAPPING_MODE_ROW) {
        asc_vf_call<ScatterByRow<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>>(
            dim3(THREAD_NUM), tilingData->numRows, tilingData->numValues, tilingData->numBins, splitsGm, valuesGm,
            weightsGm, outputGm, histogram);
    } else {
        asc_vf_call<ScatterByValue<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>>(
            dim3(THREAD_NUM), tilingData->numRows, tilingData->numValues, tilingData->numBins, splitsGm, valuesGm,
            weightsGm, outputGm, histogram);
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, uint32_t MAPPING_MODE, bool BINARY_OUTPUT,
          bool HAS_WEIGHTS>
__aicore__ inline void ProcessWithIndexType(GM_ADDR splits, GM_ADDR values, GM_ADDR weights, GM_ADDR output,
                                            GM_ADDR userWorkspace, const RaggedBinCountTilingData* tilingData,
                                            TPipe* pipe)
{
    __gm__ int64_t* splitsGm = reinterpret_cast<__gm__ int64_t*>(splits);
    __gm__ VALUE_TYPE* valuesGm = reinterpret_cast<__gm__ VALUE_TYPE*>(values);
    __gm__ float* weightsGm = reinterpret_cast<__gm__ float*>(weights);
    __gm__ float* outputGm = reinterpret_cast<__gm__ float*>(output);
    __gm__ uint32_t* invalidFlag = reinterpret_cast<__gm__ uint32_t*>(userWorkspace);

    asc_vf_call<InitializeOutputAndFlag<INDEX_TYPE, THREAD_NUM>>(dim3(THREAD_NUM), tilingData->outputElements, outputGm,
                                                                 invalidFlag);
    SyncAll();

    asc_vf_call<ValidateInputs<VALUE_TYPE, INDEX_TYPE, THREAD_NUM>>(
        dim3(THREAD_NUM), tilingData->numSplits, tilingData->numValues, splitsGm, valuesGm, invalidFlag);
    SyncAll();
    if (*invalidFlag != 0U) {
        return;
    }

    // The host privatises whenever the whole output fits in the dynamic UB budget and the extra
    // write-back is cheaper than the global atomics it removes. Both mapping modes qualify: ROW owns
    // each row outright, VALUE has every core touching every row, and a full-output private copy
    // serves both without the kernel having to know which.
    const int32_t privateHistElems = static_cast<int32_t>(tilingData->privateHistElems);
    if (privateHistElems > 0) {
        const uint32_t histogramBytes = static_cast<uint32_t>(privateHistElems) * static_cast<uint32_t>(sizeof(float));
        // Reserve whole 32-byte blocks so Duplicate's vectorised tail cannot run past the allocation;
        // the write-back below still copies exactly histogramBytes, and the host checked the same
        // rounded-up figure against the UB budget.
        const uint32_t histogramBufferBytes = ((histogramBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES) *
                                              UB_BLOCK_BYTES;
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe->InitBuffer(histogramQueue, 1, histogramBufferBytes);
        LocalTensor<float> histogram = histogramQueue.template AllocTensor<float>();
        Duplicate(histogram, 0.0F, privateHistElems);

        LaunchScatter<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, MAPPING_MODE, true, BINARY_OUTPUT, HAS_WEIGHTS>(
            tilingData, splitsGm, valuesGm, weightsGm, outputGm, (__ubuf__ float*)histogram.GetPhyAddr());

        // EnQue/DeQue is what orders the SIMT writes into UB against the DMA that reads them back out.
        histogramQueue.EnQue(histogram);
        histogram = histogramQueue.template DeQue<float>();

        // Every core folds its private copy into the already-zeroed output. Counting and weighting are
        // sums, so atomic add reproduces the total exactly. The binary path only ever stores 0.0F/1.0F,
        // and atomic max is the OR of those -- adding would yield 2.0F whenever two cores both saw a
        // bin, which VALUE mapping makes routine.
        if constexpr (BINARY_OUTPUT) {
            SetAtomicMax<float>();
        } else {
            SetAtomicAdd<float>();
        }
        GlobalTensor<float> outputGlobal;
        outputGlobal.SetGlobalBuffer(outputGm);
        const DataCopyExtParams copyParams{static_cast<uint16_t>(1), histogramBytes, 0U, 0U, 0U};
        DataCopyPad(outputGlobal, histogram, copyParams);
        SetAtomicNone();

        histogramQueue.FreeTensor(histogram);
    } else {
        LaunchScatter<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, MAPPING_MODE, false, BINARY_OUTPUT, HAS_WEIGHTS>(
            tilingData, splitsGm, valuesGm, weightsGm, outputGm, (__ubuf__ float*)nullptr);
    }
}

template <typename VALUE_TYPE, uint32_t MAPPING_MODE, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__aicore__ inline void Process(GM_ADDR splits, GM_ADDR values, GM_ADDR weights, GM_ADDR output, GM_ADDR userWorkspace,
                               const RaggedBinCountTilingData* tilingData, TPipe* pipe)
{
    // Every uint32 index path either steps by `blockDim.x * gridDim.x` (InitializeOutputAndFlag,
    // ValidateInputs, ScatterByValue) or offsets by `threadIdx.x` off a split boundary (ScatterByRow),
    // so the bound must leave a full stride of headroom below 2^32 rather than running up to the type
    // limit. Without it, a count within one stride of 2^32 makes the accumulator wrap: the loop
    // condition becomes true again at a low index, so ScatterByRow re-attributes the leading values to
    // whichever row sits at the end. INDEX_HEADROOM is that stride's upper bound; anything above the
    // reduced limit falls to the uint64 path, which is correct at any size.
    constexpr uint64_t U32_MAX_VALUE = 0xFFFFFFFFULL;
    constexpr uint64_t MAX_SUPPORTED_CORES = 128ULL;
    constexpr uint64_t INDEX_HEADROOM = static_cast<uint64_t>(THREAD_NUM_U32) * MAX_SUPPORTED_CORES;
    constexpr uint64_t U32_INDEX_LIMIT = U32_MAX_VALUE - INDEX_HEADROOM;
    const bool useUint32Index = static_cast<uint64_t>(tilingData->numSplits) <= U32_INDEX_LIMIT &&
                                static_cast<uint64_t>(tilingData->numValues) <= U32_INDEX_LIMIT &&
                                static_cast<uint64_t>(tilingData->outputElements) <= U32_INDEX_LIMIT;

    if (useUint32Index) {
        ProcessWithIndexType<VALUE_TYPE, uint32_t, THREAD_NUM_U32, MAPPING_MODE, BINARY_OUTPUT, HAS_WEIGHTS>(
            splits, values, weights, output, userWorkspace, tilingData, pipe);
    } else {
        ProcessWithIndexType<VALUE_TYPE, uint64_t, THREAD_NUM_U64, MAPPING_MODE, BINARY_OUTPUT, HAS_WEIGHTS>(
            splits, values, weights, output, userWorkspace, tilingData, pipe);
    }
}

} // namespace NsRaggedBinCount

#endif // RAGGED_BIN_COUNT_SIMT_H
