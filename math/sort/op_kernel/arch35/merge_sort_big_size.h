/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file merge_sort_big_size.h
 * \brief merge_sort kernel entry
 */
#ifndef MERGE_SORT_BIG_SIZE_H
#define MERGE_SORT_BIG_SIZE_H
#include <cmath>
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "common/util_type_simd.h"
#include "common/merge_sort_constants.h"
#include "common/merge_more_core_base.h"

using namespace AscendC;
#include "simt_api/asc_simt.h"
namespace SortMergePath {
constexpr uint32_t WARP_SIZE = 32U;
constexpr uint32_t PARTITION_THREADS = WARP_SIZE * 2U;
constexpr uint32_t PROPOSAL_WORDS = 2U;
constexpr uint32_t FLOAT_SIGN = 0x80000000U;
constexpr uint32_t FLOAT_MAGNITUDE = 0x7fffffffU;
constexpr uint32_t FLOAT_INFINITY = 0x7f800000U;
constexpr uint32_t CANONICAL_NAN = 0x7fc00000U;
constexpr uint32_t OUTPUT_THREADS = 256U;
template <bool Descend>
__simt_callee__ __aicore__ inline uint32_t Key(__gm__ uint32_t* input, int64_t pos)
{
    // Proposals contain the sign-flipped value and its original uint32 index.
    // NaNs are canonicalized before sorting; signed zeros compare equal.
    uint32_t bits = input[pos * PROPOSAL_WORDS];
    if constexpr (!Descend) {
        bits ^= FLOAT_SIGN;
    }
    uint32_t magnitude = bits & FLOAT_MAGNITUDE;
    if (magnitude == 0U) {
        return FLOAT_SIGN;
    }
    return (bits & FLOAT_SIGN) ? ~bits : (bits ^ FLOAT_SIGN);
}
template <bool Descend>
__simt_callee__ __aicore__ inline bool Before(__gm__ uint32_t* input, int64_t a, int64_t b)
{
    uint32_t ka = Key<Descend>(input, a), kb = Key<Descend>(input, b);
    return Descend ? ka > kb : ka < kb;
}
template <bool Descend>
__simt_vf__ LAUNCH_BOUND(PARTITION_THREADS) __aicore__
    void Partition(__gm__ uint32_t* input, __ubuf__ int64_t* ranks, int64_t a, int64_t b, int64_t first, int64_t end)
{
    uint32_t lane = threadIdx.x;
    if (lane != 0U && lane != WARP_SIZE) {
        return;
    }
    uint32_t query = lane / WARP_SIZE;
    int64_t diagonal = query == 0 ? first : end;
    int64_t low = diagonal > b ? diagonal - b : 0;
    int64_t high = diagonal < a ? diagonal : a;
    while (low <= high) {
        int64_t i = (low + high) / 2;
        int64_t j = diagonal - i;
        if (i > 0 && j < b && Before<Descend>(input, a + j, i - 1)) {
            high = i - 1;
        } else if (j > 0 && i < a && !Before<Descend>(input, a + j - 1, i)) {
            low = i + 1;
        } else {
            ranks[query] = i;
            return;
        }
    }
    ranks[query] = low;
}
template <bool Descend, typename Index>
__simt_vf__ LAUNCH_BOUND(OUTPUT_THREADS) __aicore__
    void WriteOutput(__ubuf__ uint32_t* proposals, __gm__ uint32_t* input, __gm__ uint32_t* values,
                     __gm__ Index* indices, uint32_t count)
{
    for (uint32_t i = threadIdx.x; i < count; i += OUTPUT_THREADS) {
        uint32_t bits = proposals[PROPOSAL_WORDS * i];
        uint32_t index = proposals[PROPOSAL_WORDS * i + 1U];
        if constexpr (!Descend) {
            bits ^= FLOAT_SIGN;
        }
        if ((bits & FLOAT_MAGNITUDE) > FLOAT_INFINITY) {
            bits = input[index];
        }
        values[i] = bits;
        indices[i] = static_cast<Index>(index);
    }
}
__aicore__ inline void CanonicalizeNan(LocalTensor<float> input, uint32_t count)
{
    constexpr uint32_t vl = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    uint16_t repeats = Ops::Base::CeilDiv(count, vl);
    __ubuf__ uint32_t* address = (__ubuf__ uint32_t*)input.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> value, magnitude, magnitudeMask, canonical, result;
        Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(magnitudeMask, FLOAT_MAGNITUDE, full);
        Reg::Duplicate(canonical, CANONICAL_NAN, full);
        for (uint16_t i = 0; i < repeats; ++i) {
            Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(count), nan;
            Reg::LoadAlign(value, address + i * vl);
            Reg::And(magnitude, value, magnitudeMask, valid);
            Reg::Compares<uint32_t, CMPMODE::GT>(nan, magnitude, FLOAT_INFINITY, valid);
            Reg::Select(result, canonical, value, nan);
            Reg::StoreAlign(address + i * vl, result, valid);
        }
    }
}

} // namespace SortMergePath

// Import shared constants from MergeSortConstants namespace
using MergeSortConstants::DEALING_EXTRACT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_LIST_MAX_NUM;
using MergeSortConstants::MERGE_MORE_BUFFER_NUM;
using MergeSortConstants::UB_BLOCK_BYTES;
using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE = false,
          bool USE_MERGE_PATH = false>
struct MergeSortBigSize
    : public MergeMoreCoreCommon::MergeMoreCoreBase<
          MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>, T, CONVERT_TYPE,
          IS_DESCEND, INDEX_TYPE> {
    using Base = MergeMoreCoreCommon::MergeMoreCoreBase<
        MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>, T, CONVERT_TYPE,
        IS_DESCEND, INDEX_TYPE>;
    friend Base;

    __aicore__ inline MergeSortBigSize() {}
    __aicore__ inline void Init(GM_ADDR inputValue, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
                                const SortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void PrepareRowWorkspace();
    __aicore__ inline void InitSortBuffers();
    __aicore__ inline void InitMergeBuffers();
    __aicore__ inline void ExtractAndCopyOut();
    __aicore__ inline void CopyMergePathOutput();
    __aicore__ inline void OnInputLoaded(LocalTensor<T>, uint32_t) {}
    __aicore__ inline void PrepareInputForSort(LocalTensor<T> input, uint32_t count)
    {
        if constexpr (USE_MERGE_PATH) {
            // Native proposals must have the same NaN ordering as the co-rank search.
            // Output gathers restore the original NaN payloads through their stable indices.
            SortMergePath::CanonicalizeNan(input, count);
        }
    }

    __aicore__ inline void FinalSortAndCopy(bool rowActive)
    {
        if constexpr (!USE_MERGE_PATH) {
            Base::FinalSortAndCopy(rowActive);
            return;
        } else {
            if (this->listNum_ != MergeSortConstants::TWO_WAY_MERGE_LIST_NUM ||
                this->frontCoreNum_ < MergeSortConstants::TWO_WAY_MERGE_LIST_NUM) {
                Base::FinalSortAndCopy(rowActive);
                return;
            }
            if (rowActive) {
                // Each core owns disjoint output ranks. Co-ranking chooses equal keys from
                // the earlier input run first, preserving stable indices across partitions.
                int64_t first = static_cast<int64_t>(this->outputLastDimValue_) * this->rowCoreIdx_ /
                                this->frontCoreNum_;
                int64_t end = static_cast<int64_t>(this->outputLastDimValue_) * (this->rowCoreIdx_ + 1U) /
                              this->frontCoreNum_;
                LocalTensor<int64_t> ranks = this->castIndexQueue_.template AllocTensor<int64_t>();
                asc_vf_call<SortMergePath::Partition<IS_DESCEND>>(
                    dim3(SortMergePath::PARTITION_THREADS), (__gm__ uint32_t*)this->workspaceInput_.GetPhyAddr(),
                    (__ubuf__ int64_t*)ranks.GetPhyAddr(), this->currentElements_, this->currentTailElements_, first,
                    end);
                event_t ready = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(ready);
                WaitFlag<HardEvent::V_S>(ready);
                int64_t startA = ranks.GetValue(0), endA = ranks.GetValue(1);
                this->castIndexQueue_.FreeTensor(ranks);
                this->ClearCache();
                this->outOffset_ = first;
                this->offsets_[0] = GetSortOffset<CONVERT_TYPE>(startA);
                this->offsets_[1] = GetSortOffset<CONVERT_TYPE>(this->currentElements_ + first - startA);
                this->listRemainElements_[0] = endA - startA;
                this->listRemainElements_[1] = end - first - (endA - startA);
                this->allRemainElements_ = end - first;
                while (this->allRemainElements_ > 0) {
                    this->CopyInMultiCore();
                    this->UpdateMrgParam();
                    this->DealingMergeSort();
                    this->UpdateSortInfo();
                    ExtractAndCopyOut();
                }
                this->ClearCache();
            }
            SyncAll();
        }
    }

    __gm__ CONVERT_TYPE* workspaceBase_ = nullptr;
    uint64_t rowWorkspaceElements_ = 0;
};

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::Init(
    GM_ADDR inputValue, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace, const SortRegBaseTilingData* tilingData,
    TPipe* pipe)
{
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    this->outputLastDimValue_ = tilingData->lastDimTileNum;
    this->numTileData_ = tilingData->numTileDataSize;
    this->frontCoreNum_ = tilingData->lastDimNeedCore;
    if (this->frontCoreNum_ == 0U) {
        return;
    }
    if constexpr (DIRECT_SCHEDULE) {
        this->syncMergeSortMode_ = false;
    } else {
        this->rowsPerRound_ = tilingData->unsortedDimParallel == 0U ?
                                  static_cast<uint32_t>(tilingData->unsortedDimNum) :
                                  tilingData->unsortedDimParallel;
        this->sortLoopTimes_ = tilingData->sortLoopTimes == 0U ? 1U : tilingData->sortLoopTimes;
        this->batchNum_ = tilingData->unsortedDimNum;
        this->logicalBlockSize_ = tilingData->keyParams1;
        this->syncMergeSortMode_ = this->logicalBlockSize_ > this->numTileData_ && this->numTileData_ > 0U;
        if (this->rowsPerRound_ == 0U || this->sortLoopTimes_ == 0U) {
            return;
        }
    }
    uint32_t sortBufferSize = 8;
    this->rowGroupIdx_ = this->blockIdx_ / this->frontCoreNum_;
    this->rowCoreIdx_ = this->blockIdx_ % this->frontCoreNum_;
    // Per-row workspace stores Sort API sort-struct data. This capacity uses sortBufferSize bytes per
    // original element and UB-block byte alignment; it must cover later GetSortLen-based accesses.
    uint64_t rowWorkspaceBytes = ROUND_UP_AGLIN_UINT64(static_cast<uint64_t>(this->outputLastDimValue_) *
                                                       sortBufferSize);
    this->rowWorkspaceElements_ = rowWorkspaceBytes / sizeof(CONVERT_TYPE);
    this->onceMaxElements_ = tilingData->keyParams0 / DEALING_SORT_NUM_ONCE * DEALING_SORT_NUM_ONCE;

    this->inputValueGm_.SetGlobalBuffer((__gm__ T*)(inputValue));
    this->outValueGm_.SetGlobalBuffer((__gm__ T*)(value));
    this->outIndexGm_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    this->workspaceBase_ = (__gm__ CONVERT_TYPE*)workSpace;
    if constexpr (DIRECT_SCHEDULE) {
        this->rowIdx_ = this->rowGroupIdx_;
        PrepareRowWorkspace();
        InitSortBuffers();
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void
MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::Process()
{
    if constexpr (DIRECT_SCHEDULE) {
        this->ProcessDirect();
    } else {
        Base::Process();
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void
MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::PrepareRowWorkspace()
{
    this->rowDataOffset_ = static_cast<int64_t>(this->rowIdx_) * static_cast<int64_t>(this->outputLastDimValue_);
    this->rowWorkspaceOffset_ = static_cast<int64_t>(this->rowIdx_) *
                                static_cast<int64_t>(this->rowWorkspaceElements_) * 2;
    this->workspaceGm_[0].SetGlobalBuffer(this->workspaceBase_ + this->rowWorkspaceOffset_,
                                          this->rowWorkspaceElements_);
    this->workspaceGm_[1].SetGlobalBuffer(
        this->workspaceBase_ + this->rowWorkspaceOffset_ + this->rowWorkspaceElements_, this->rowWorkspaceElements_);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void
MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::InitSortBuffers()
{
    uint32_t sortBufferSize = 8;
    uint32_t sortTileNum;
    if constexpr (DIRECT_SCHEDULE) {
        sortTileNum = this->outputLastDimValue_ - (this->frontCoreNum_ - 1) * this->numTileData_;
    } else {
        sortTileNum = this->syncMergeSortMode_ ?
                          this->numTileData_ :
                          this->outputLastDimValue_ - (this->frontCoreNum_ - 1) * this->numTileData_;
    }
    uint32_t alignTile = ROUND_UP_AGLIN(sortTileNum);
    this->pipe_->InitBuffer(this->inputQueue_, MERGE_MORE_BUFFER_NUM, alignTile * sizeof(T));

    this->pipe_->InitBuffer(this->sortedValueUb_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortedValueIndexUb_, alignTile * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->sortTempBuf_, alignTile * sortBufferSize);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void
MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::InitMergeBuffers()
{
    uint32_t sortBufferSize = 8;
    this->pipe_->InitBuffer(this->sortedQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sortBufferSize);
    this->pipe_->InitBuffer(this->copyInQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sortBufferSize);
    if constexpr (USE_MERGE_PATH) {
        // Fused SIMT output only needs proposals. Reuse this queue for the two
        // int64 co-ranks, rounded up to one UB block; no cast/output buffers.
        this->pipe_->InitBuffer(this->castIndexQueue_, MERGE_MORE_BUFFER_NUM, UB_BLOCK_BYTES);
        return;
    }
    this->pipe_->InitBuffer(this->castValueQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(CONVERT_TYPE));
    this->pipe_->InitBuffer(this->castIndexQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(uint32_t));
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        this->pipe_->InitBuffer(this->outIndexQueue_, MERGE_MORE_BUFFER_NUM,
                                MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(INDEX_TYPE));
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void
MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::CopyMergePathOutput()
{
    LocalTensor<CONVERT_TYPE> proposals = this->sortedQueue_.template DeQue<CONVERT_TYPE>();
    asc_vf_call<SortMergePath::WriteOutput<IS_DESCEND, INDEX_TYPE>>(
        dim3(SortMergePath::OUTPUT_THREADS), (__ubuf__ uint32_t*)proposals.GetPhyAddr(),
        (__gm__ uint32_t*)this->inputValueGm_[this->rowDataOffset_].GetPhyAddr(),
        (__gm__ uint32_t*)this->outValueGm_[this->rowDataOffset_ + this->outOffset_].GetPhyAddr(),
        (__gm__ INDEX_TYPE*)this->outIndexGm_[this->rowDataOffset_ + this->outOffset_].GetPhyAddr(),
        static_cast<uint32_t>(this->curLoopSortedNum_));
    event_t done = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(done);
    WaitFlag<HardEvent::V_S>(done);
    this->sortedQueue_.FreeTensor(proposals);
    this->outOffset_ += this->curLoopSortedNum_;
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool DIRECT_SCHEDULE,
          bool USE_MERGE_PATH>
__aicore__ inline void
MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, DIRECT_SCHEDULE, USE_MERGE_PATH>::ExtractAndCopyOut()
{
    if constexpr (USE_MERGE_PATH) {
        CopyMergePathOutput();
        return;
    }
    LocalTensor<INDEX_TYPE> ubOutput2;
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        ubOutput2 = this->outIndexQueue_.template AllocTensor<INDEX_TYPE>();
    }
    LocalTensor<CONVERT_TYPE> sortTempBuffer = this->sortedQueue_.template DeQue<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> castValue = this->castValueQueue_.template AllocTensor<CONVERT_TYPE>();
    LocalTensor<uint32_t> castIndex = this->castIndexQueue_.template AllocTensor<uint32_t>();
    AscendC::Extract(castValue, castIndex, sortTempBuffer,
                     Ops::Base::CeilDiv(this->curLoopSortedNum_, static_cast<int64_t>(DEALING_EXTRACT_NUM_ONCE)));
    if constexpr (!IS_DESCEND) {
        this->FlipSignBit(castValue, ROUND_UP_AGLIN(this->curLoopSortedNum_));
    }
    DataCopyExtParams copyParamsValue;
    copyParamsValue.blockCount = 1;
    copyParamsValue.blockLen = this->curLoopSortedNum_ * sizeof(T);
    copyParamsValue.srcStride = 0;
    copyParamsValue.dstStride = 0;

    DataCopyExtParams copyParamsIndex;
    copyParamsIndex.blockCount = 1;
    copyParamsIndex.blockLen = this->curLoopSortedNum_ * sizeof(INDEX_TYPE);
    copyParamsIndex.srcStride = 0;
    copyParamsIndex.dstStride = 0;

    this->castValueQueue_.EnQue(castValue);
    castValue = this->castValueQueue_.template DeQue<T>();
    DataCopyPad(this->outValueGm_[this->rowDataOffset_ + this->outOffset_], castValue, copyParamsValue);
    this->castValueQueue_.FreeTensor(castValue);

    uint32_t sortedIndexAlign = ROUND_UP_AGLIN(this->curLoopSortedNum_ * sizeof(uint32_t)) / sizeof(uint32_t);
    LocalTensor<int32_t> castIndexTemp = castIndex.template ReinterpretCast<int32_t>();
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        AscendC::Cast(ubOutput2, castIndexTemp, AscendC::RoundMode::CAST_NONE, sortedIndexAlign);
        this->outIndexQueue_.EnQue(ubOutput2);
        ubOutput2 = this->outIndexQueue_.template DeQue<INDEX_TYPE>();
        DataCopyPad(this->outIndexGm_[this->rowDataOffset_ + this->outOffset_], ubOutput2, copyParamsIndex);
        this->outIndexQueue_.FreeTensor(ubOutput2);
    } else {
        this->castIndexQueue_.EnQue(castIndex);
        castIndex = this->castIndexQueue_.template DeQue<uint32_t>();
        castIndexTemp = castIndex.template ReinterpretCast<int32_t>();
        DataCopyPad(this->outIndexGm_[this->rowDataOffset_ + this->outOffset_], castIndexTemp, copyParamsIndex);
    }
    this->castIndexQueue_.FreeTensor(castIndex);
    this->sortedQueue_.FreeTensor(sortTempBuffer);
    this->outOffset_ += this->curLoopSortedNum_;
}
#endif // MERGE_SORT_BIG_SIZE_H
