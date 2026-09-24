/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARALLEL_CONCAT_KERNEL_SIMT_H
#define PARALLEL_CONCAT_KERNEL_SIMT_H

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "parallel_concat_tiling_struct.h"

constexpr uint32_t PC_SIMT_THREAD_NUM = 1024;
constexpr uint64_t ROW_PER_THREAD_MIN_ROWS = 32;

__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b) { return a < b ? a : b; }

__simt_vf__ __aicore__ LAUNCH_BOUND(PC_SIMT_THREAD_NUM) inline void ParallelConcatSimtCopyRowsVf(
    uint64_t rowBegin, uint64_t rowCount, uint64_t rowBytes, const __gm__ uint8_t* values, __gm__ uint8_t* y)
{
    const __gm__ uint64_t* desc = reinterpret_cast<const __gm__ uint64_t*>(values);
    const uint64_t descLen = desc[0];
    const __gm__ uint64_t* addrTable = reinterpret_cast<const __gm__ uint64_t*>(values + descLen);
    for (uint64_t r = threadIdx.x; r < rowCount; r += blockDim.x) {
        const uint64_t row = rowBegin + r; // global row index (dynamic input #row)
        __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(addrTable[row]);
        __gm__ uint8_t* dst = y + row * rowBytes;
        for (uint64_t b = 0; b < rowBytes; ++b) {
            dst[b] = src[b]; // bit-exact (flat(y)[row·L+t'] = flat(v_row)[t])
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(PC_SIMT_THREAD_NUM) inline void ParallelConcatSimtCopyBytesVf(
    uint64_t rowBegin, uint64_t rowCount, uint64_t rowBytes, const __gm__ uint8_t* values, __gm__ uint8_t* y)
{
    const __gm__ uint64_t* desc = reinterpret_cast<const __gm__ uint64_t*>(values);
    const uint64_t descLen = desc[0];
    const __gm__ uint64_t* addrTable = reinterpret_cast<const __gm__ uint64_t*>(values + descLen);
    const uint64_t coreBytes = rowCount * rowBytes;
    __gm__ uint8_t* dstBase = y + rowBegin * rowBytes;
    for (uint64_t i = threadIdx.x; i < coreBytes; i += blockDim.x) {
        const uint64_t rowLocal = i / rowBytes; // core-local row index
        __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(addrTable[rowBegin + rowLocal]);
        dstBase[i] = src[i - rowLocal * rowBytes]; // plain GM deref: bitwise byte contract
    }
}

class ParallelConcatKernelSimt {
public:
    __aicore__ inline void Init(GM_ADDR values, GM_ADDR outputData, const ParallelConcatTilingData* tilingData)
    {
        // Core guard: non-active cores return; the empty tensor already
        // returned at the kernel entry.
        if (AscendC::GetBlockIdx() >= tilingData->numActiveCores) {
            active_ = false;
            return;
        }
        td_ = tilingData;
        yGm_ = reinterpret_cast<__gm__ uint8_t*>(outputData);    // output byte view (VF raw pointer)
        inputBases_ = reinterpret_cast<__gm__ uint8_t*>(values); // dynamic input descriptor base
        // RowLayout one-shot boundary decomposition (chunksPerRow ≡ 1 in this
        // branch, so chunk index == row index).
        const uint64_t c = AscendC::GetBlockIdx();
        const uint64_t totalChunks = td_->n; // = n × 1 (branch degeneration)
        const uint64_t baseC = td_->perCoreChunks;
        const uint64_t remC = totalChunks - baseC * td_->numActiveCores;
        rowBegin_ = c * baseC + MinU64(c, remC);
        rowCount_ = baseC + ((c < remC) ? 1ULL : 0ULL);
    }

    /**
     * Process: single asc_vf_call for this core's whole share, joined by the
     * framework before kernel exit. The dim3 grid is ALWAYS the compile-time
     * constant PC_SIMT_THREAD_NUM (same as LAUNCH_BOUND); the VF entry is
     * selected here on the kernel side:
     *   - rowCount ∈ [32, 1024]: one row per thread iteration;
     *   - otherwise: thread-stride byte loop.
     */
    __aicore__ inline void Process()
    {
        if (!active_ || rowCount_ == 0ULL) {
            return; // non-active core / degenerate share (empty tensor returned at entry)
        }
        const uint64_t rowBytes = td_->rowBytes;
        if (rowCount_ >= ROW_PER_THREAD_MIN_ROWS && rowCount_ <= PC_SIMT_THREAD_NUM) {
            asc_vf_call<ParallelConcatSimtCopyRowsVf>(dim3{PC_SIMT_THREAD_NUM, 1, 1}, rowBegin_, rowCount_, rowBytes,
                                                      inputBases_, yGm_); // async; joined before kernel exit
        } else {
            asc_vf_call<ParallelConcatSimtCopyBytesVf>(dim3{PC_SIMT_THREAD_NUM, 1, 1}, rowBegin_, rowCount_, rowBytes,
                                                       inputBases_, yGm_); // async; joined before kernel exit
        }
    }

private:
    const ParallelConcatTilingData* td_ = nullptr; // const pointer bind
    bool active_ = true;                           // core-guard flag
    uint64_t rowBegin_ = 0;                        // RowLayout: first row of this core
    uint64_t rowCount_ = 0;                        // RowLayout: row count of this core
    __gm__ uint8_t* yGm_ = nullptr;                // output uint8 byte view (VF dst base)
    __gm__ uint8_t* inputBases_ = nullptr;         // dynamic input address table base
};

#endif // PARALLEL_CONCAT_KERNEL_SIMT_H
