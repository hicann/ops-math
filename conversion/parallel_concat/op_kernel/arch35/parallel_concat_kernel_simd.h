/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARALLEL_CONCAT_KERNEL_SIMD_H
#define PARALLEL_CONCAT_KERNEL_SIMD_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h" // dynamic input address table (ListTensorDesc)
#include "op_kernel/math_util.h"              // Ops::Base::CeilDiv
#include "parallel_concat_tiling_struct.h"    // shared TilingData POD + routing threshold

class ParallelConcatKernelSimd {
public:
    __aicore__ inline void Init(GM_ADDR values, GM_ADDR outputData, const ParallelConcatTilingData* td,
                                AscendC::TPipe* pipe)
    {
        td_ = td;
        if (AscendC::GetBlockIdx() >= td_->numActiveCores) {
            return;
        }
        active_ = true;
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outputData));
        valuesList_ = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void*>(values));
        pipe->InitBuffer(dataQue_, UB_SLOT_NUM, static_cast<uint32_t>(td_->bufferSize));
        rowBytes_ = td_->rowBytes;
        bufferSize_ = static_cast<uint64_t>(td_->bufferSize);
        chunksPerRow_ = Ops::Base::CeilDiv(rowBytes_, bufferSize_);
        GetCoreChunkRange(td_, AscendC::GetBlockIdx(), chunkBegin_, chunkEnd_);
        rFirst_ = chunkBegin_ / chunksPerRow_;
        kFirst_ = chunkBegin_ % chunksPerRow_;
        rLast_ = (chunkEnd_ - 1) / chunksPerRow_;
        kLast_ = (chunkEnd_ - 1) % chunksPerRow_;
        rowCount_ = rLast_ - rFirst_ + 1;
        if (rowCount_ <= ROW_BASE_PREFETCH_MAX) {
            for (uint64_t i = 0; i < rowCount_; ++i) {
                rowBaseCache_[i] = GetRowBase(rFirst_ + i);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return; // inactive cores: zero iterations
        }
        for (uint64_t r = rFirst_; r <= rLast_; ++r) {
            const uint64_t kBegin = (r == rFirst_) ? kFirst_ : 0;
            const uint64_t kEnd = (r == rLast_) ? (kLast_ + 1) : chunksPerRow_;
            // Row GM view: resolved once per row (prefetched register row
            // base for rowCount <= 8, on-demand address-table read above).
            srcGm_.SetGlobalBuffer((rowCount_ <= ROW_BASE_PREFETCH_MAX) ? rowBaseCache_[r - rFirst_] : GetRowBase(r));
            for (uint64_t k = kBegin; k < kEnd; ++k) {
                AscendC::LocalTensor<uint8_t> ub = dataQue_.AllocTensor<uint8_t>();
                CopyIn(k, ub);
                dataQue_.EnQue(ub);
                AscendC::LocalTensor<uint8_t> ready = dataQue_.DeQue<uint8_t>();
                CopyOut(r, k, ready);
                dataQue_.FreeTensor(ready);
            }
        }
    }

private:
    /**
     * ChunkBytes: valid bytes of in-row chunk k — full bufferSize except the
     * row tail chunk, which narrows to the 1B-granular remainder.
     */
    __aicore__ inline uint32_t ChunkBytes(uint64_t k) const
    {
        return static_cast<uint32_t>(AscendC::Std::min(bufferSize_, rowBytes_ - k * bufferSize_));
    }

    /**
     * CopyIn: MTE2 copy-in GM -> UB slot, DataCopyPad 4-param.
     * src = this row's input GM base (per-row SetGlobalBuffer in Process) +
     * in-row chunk offset (k × bufferSize); isPad=false — no padding
     * injection (bitwise contract).
     */
    __aicore__ inline void CopyIn(uint64_t k, const AscendC::LocalTensor<uint8_t>& ub)
    {
        const uint32_t chunkBytes = ChunkBytes(k);
        AscendC::DataCopyExtParams inParams = {1, chunkBytes, 0, 0, 0};      // blockCount=1 contiguous; strides=0 (gap)
        AscendC::DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0}; // isPad=false: no padding
        AscendC::DataCopyPad(ub, srcGm_[k * bufferSize_], inParams, padParams); // MTE2: GM -> UB slot
    }

    /**
     * CopyOut: MTE3 copy-out UB -> GM, DataCopyPad 3-param. dst = output GM
     * flat row area r × rowBytes + k × bufferSize (in-row identity with the
     * source side — the copy IS the computation); blockLen carries the
     * 1B-granular tail (UBToGM 1B-alignment semantics, any row width).
     * Non-inplace: writes the independent output_data buffer directly.
     */
    __aicore__ inline void CopyOut(uint64_t r, uint64_t k, const AscendC::LocalTensor<uint8_t>& ub)
    {
        const uint32_t chunkBytes = ChunkBytes(k); // same tail narrowing as CopyIn (read=write width)
        AscendC::DataCopyExtParams outParams = {1, chunkBytes, 0, 0, 0};            // blockLen 1B-granular
        AscendC::DataCopyPad(yGm_[r * rowBytes_ + k * bufferSize_], ub, outParams); // MTE3: UB -> GM
    }

    // ---- shared skeleton helpers (class-scoped: the TPL build compiles both
    // engine headers in one translation unit; common math comes from opbase
    // Ops::Base::CeilDiv and AscendC Std::min) ----

    /**
     * GetCoreChunkRange: this core's chunk interval in the N-first flattened
     * chunk stream — core c owns
     *   [c×baseC + min(c, remC), c×baseC + min(c, remC) + baseC + (c < remC ? 1 : 0))
     * (first remC cores take baseC+1 chunks each; intervals contiguous and
     * disjoint -> per-core output GM segments disjoint).
     */
    static __aicore__ inline void GetCoreChunkRange(const ParallelConcatTilingData* td, uint64_t coreId,
                                                    uint64_t& chunkBegin, uint64_t& chunkEnd)
    {
        const uint64_t chunksPerRow = Ops::Base::CeilDiv(td->rowBytes, static_cast<uint64_t>(td->bufferSize));
        const uint64_t totalChunks = td->n * chunksPerRow; // global chunk id g = r×chunksPerRow + k
        const uint64_t baseC = td->perCoreChunks;          // host-filled: totalChunks / numActiveCores
        const uint64_t remC = totalChunks - baseC * static_cast<uint64_t>(td->numActiveCores);
        chunkBegin = coreId * baseC + AscendC::Std::min(coreId, remC); // first remC cores take one extra chunk
        chunkEnd = chunkBegin + baseC + ((coreId < remC) ? 1ULL : 0ULL);
    }

    /**
     * GetRowBase: GM data base of dynamic input v_r (uint8 byte view) via the
     * ListTensorDesc address table (the standard TensorList descriptor
     * contract, built by the framework).
     */
    __aicore__ inline __gm__ uint8_t* GetRowBase(uint64_t r)
    {
        return valuesList_.GetDataPtr<uint8_t>(static_cast<uint32_t>(r));
    }

    static constexpr int32_t UB_SLOT_NUM = 2;
    static constexpr uint64_t ROW_BASE_PREFETCH_MAX = 8;

    const ParallelConcatTilingData* td_ = nullptr; // const pointer bind
    bool active_ = false;                          // core-guard flag (Init raises it on active cores)
    // TPipe is defined OUTSIDE the class (kernel entry function local) and
    // bound here only through the Init parameter — compiler constant folding.
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, UB_SLOT_NUM>
        dataQue_;                          // shared in/out double buffer
    AscendC::GlobalTensor<uint8_t> yGm_;   // output GM byte view (MTE3 dst)
    AscendC::GlobalTensor<uint8_t> srcGm_; // input row view (per-row SetGlobalBuffer reuse)
    AscendC::ListTensorDesc valuesList_;   // dynamic input address table (TensorList materialisation)
    // RowLayout scalars (fixed once in Init, never recomputed downstream).
    uint64_t rowBytes_ = 0;     // row width = td_->rowBytes
    uint64_t bufferSize_ = 0;   // single UB block size = td_->bufferSize (slot = bufferSize bytes)
    uint64_t chunksPerRow_ = 0; // in-row chunk count = CeilDiv(rowBytes, bufferSize)
    uint64_t chunkBegin_ = 0;   // this core's first global chunk id
    uint64_t chunkEnd_ = 0;     // this core's chunk end (exclusive)
    uint64_t rFirst_ = 0;       // method B: first row of this core = chunkBegin / chunksPerRow
    uint64_t kFirst_ = 0;       // method B: first row's in-row start chunk = chunkBegin % chunksPerRow
    uint64_t rLast_ = 0;        // method B: last row (inclusive) = (chunkEnd − 1) / chunksPerRow
    uint64_t kLast_ = 0;        // method B: last row's in-row last chunk (inclusive) = (chunkEnd − 1) % chunksPerRow
    uint64_t rowCount_ = 0;     // rows owned by this core = rLast_ − rFirst_ + 1
    __gm__ uint8_t* rowBaseCache_[ROW_BASE_PREFETCH_MAX] = {}; // Init-prefetched row bases (rowCount_ <= 8)
};

#endif // PARALLEL_CONCAT_KERNEL_SIMD_H
