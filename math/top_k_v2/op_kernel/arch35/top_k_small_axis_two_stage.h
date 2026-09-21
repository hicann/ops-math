/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file top_k_small_axis_two_stage.h
 * @brief TopK小轴两段排序模板（复用sort的SmallAxisTwoStageBase实现）
 */

#ifndef TOP_K_SMALL_AXIS_TWO_STAGE_H
#define TOP_K_SMALL_AXIS_TWO_STAGE_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "../../sort/arch35/common/small_axis_two_stage_base.h"
#include "top_k_small_size_bitonic_sort.h"

namespace topkV2 {
using namespace AscendC;

/**
 * @brief SIMT store：last轴，每段写前k个，输出GM布局 [totalSegs, k]
 */
template <typename T, typename OUT_IDX_T>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void SimtStoreTopKTwoStageBatch(uint32_t validSegs, uint32_t segmentLen, uint32_t kValue, uint64_t outputStart,
                                    __ubuf__ T* finalValues, __ubuf__ uint32_t* finalIdx,
                                    __gm__ volatile T* outputValue, __gm__ volatile OUT_IDX_T* outputIndex)
{
    uint32_t total = validSegs * kValue;
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < total; idx += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        uint32_t seg = idx / kValue;
        uint32_t rank = idx - seg * kValue;
        uint32_t srcOffset = seg * segmentLen + rank;
        outputValue[outputStart + idx] = finalValues[srcOffset];
        outputIndex[outputStart + idx] = static_cast<OUT_IDX_T>(finalIdx[srcOffset]);
    }
}

/**
 * @brief SIMT store：非last轴，输出GM布局 [outer, k, inner]
 */
template <typename T, typename OUT_IDX_T>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void SimtStoreTopKNonLastTwoStageBatch(uint32_t validSegs, uint32_t segmentLen, uint32_t kValue,
                                           uint64_t outerBaseOffset, uint64_t innerStart, uint64_t innerSize,
                                           __ubuf__ T* finalValues, __ubuf__ uint32_t* finalIdx,
                                           __gm__ volatile T* outputValue, __gm__ volatile OUT_IDX_T* outputIndex)
{
    uint32_t total = validSegs * kValue;
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < total; idx += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        uint32_t rank = idx / validSegs;
        uint32_t seg = idx - rank * validSegs;
        uint32_t srcOffset = seg * segmentLen + rank;
        uint64_t gmOffset = outerBaseOffset + static_cast<uint64_t>(rank) * innerSize + innerStart + seg;
        outputValue[gmOffset] = finalValues[srcOffset];
        outputIndex[gmOffset] = static_cast<OUT_IDX_T>(finalIdx[srcOffset]);
    }
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
class TopKSmallAxisTwoStage
    : public SmallAxisCommon::SmallAxisTwoStageBase<TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>, T, uint32_t,
                                                    IsDescend> {
    using Base = SmallAxisCommon::SmallAxisTwoStageBase<TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>, T, uint32_t,
                                                        IsDescend>;

public:
    __aicore__ inline TopKSmallAxisTwoStage() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const TopKV2TilingDataSimd* tilingData,
                                TPipe* pipe);
    __aicore__ inline void Process() { Base::Process(); }

    friend Base;

private:
    using Base::batchNum_;
    using Base::batchSize_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::finalIdx_;
    using Base::finalValues_;
    using Base::maxFlatElems_;
    using Base::pipe_;
    using Base::segmentLen_;

    // Main pipeline functions (in call order)
    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void RunBitonicFinalize(uint32_t validSegs);
    __aicore__ inline void StoreTopK(int64_t segStart, uint32_t validSegs);
    __aicore__ inline void StoreNonLastTopK(uint64_t outerId, uint64_t innerStart, uint32_t validSegs);

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OUT_IDX_T> outIdxGm_;

    uint32_t kValue_ = 0;
    bool useBitonicFinalize_ = false;
    int64_t totalSegs_ = 0;
    int64_t outerSize_ = 1;
    int64_t innerSize_ = 1;
    uint32_t innerLoopNum_ = 1;
    bool isNonLastAxis_ = false;
};

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                                            const TopKV2TilingDataSimd* tilingData,
                                                                            TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tilingData->keyParams0;
    batchNum_ = tilingData->keyParams1;
    segmentLen_ = tilingData->numTileDataSize;
    maxFlatElems_ = batchSize_ * segmentLen_;
    kValue_ = static_cast<uint32_t>(tilingData->topKRealValue);
    useBitonicFinalize_ = tilingData->keyParams5 != 0U;
    isNonLastAxis_ = tilingData->keyParams3 != 0U;
    innerLoopNum_ = tilingData->keyParams4;
    if (isNonLastAxis_) {
        // Mode-8 convention: unsortedDimNum is innerSize and oneCoreRowNum is outerSize.
        outerSize_ = static_cast<int64_t>(tilingData->oneCoreRowNum);
        innerSize_ = static_cast<int64_t>(tilingData->unsortedDimNum);
    } else {
        totalSegs_ = static_cast<int64_t>(tilingData->unsortedDimNum);
    }

    inputGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)values);
    outIdxGm_.SetGlobalBuffer((__gm__ OUT_IDX_T*)indices);

    if (batchSize_ == 0 || segmentLen_ == 0 || maxFlatElems_ == 0 || kValue_ == 0 || kValue_ > segmentLen_) {
        return;
    }

    constexpr uint32_t kAliasElemBytes = static_cast<uint32_t>(sizeof(uint32_t));
    Base::InitSortBuffers(pipe, maxFlatElems_, tilingData->tmpUbSize, tilingData->keyParams2 != 0U, kAliasElemBytes);
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline bool TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || batchSize_ == 0 || segmentLen_ == 0 || kValue_ == 0 || kValue_ > segmentLen_ ||
           (isNonLastAxis_ && (innerLoopNum_ == 0 || outerSize_ <= 0 || innerSize_ <= 0));
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline uint32_t TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::ComputeValidSegs(uint32_t batchId) const
{
    if (isNonLastAxis_) {
        // Non-last batches are grouped by outer slice, then by inner tile.
        // The last inner tile can be shorter than batchSize_.
        uint32_t tileInOuter = batchId % innerLoopNum_;
        int64_t tileStart = static_cast<int64_t>(tileInOuter) * static_cast<int64_t>(batchSize_);
        int64_t remainingInTile = innerSize_ - tileStart;
        if (remainingInTile <= 0) {
            return 0;
        }
        return remainingInTile >= static_cast<int64_t>(batchSize_) ? batchSize_ :
                                                                     static_cast<uint32_t>(remainingInTile);
    }
    int64_t batchStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t remainingSegs = totalSegs_ - batchStart;
    if (remainingSegs <= 0) {
        return 0;
    }
    if (remainingSegs >= static_cast<int64_t>(batchSize_)) {
        return batchSize_;
    }
    return static_cast<uint32_t>(remainingSegs);
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::ProcessBatch(uint32_t batchId,
                                                                                    uint32_t validSegs)
{
    uint32_t totalElems = validSegs * segmentLen_;
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    // Pipeline: load -> two-stage sort (Stage1 flat sort + Stage2 per-segment restore) -> topk store.
    uint64_t outerId = 0;
    uint64_t innerStart = 0;
    if (isNonLastAxis_) {
        // batchId is linearized as outerId * innerLoopNum_ + innerTileId.
        outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint32_t innerTileId = batchId % innerLoopNum_;
        innerStart = static_cast<uint64_t>(innerTileId) * static_cast<uint64_t>(batchSize_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        Base::LoadNonLastBatch(inputGm_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_), validSegs,
                               totalElems);
    } else {
        Base::LoadContiguousBatch(inputGm_, segStart * static_cast<int64_t>(segmentLen_), totalElems);
    }
    Base::RunTwoStageSort(totalElems);
    RunBitonicFinalize(validSegs);
    if (isNonLastAxis_) {
        StoreNonLastTopK(outerId, innerStart, validSegs);
    } else {
        StoreTopK(segStart, validSegs);
    }
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::RunBitonicFinalize(uint32_t validSegs)
{
    if (!useBitonicFinalize_ || validSegs == 0U) {
        return;
    }
    RunBitonicFinalizeSelectionRows<T, uint32_t, IsDescend>(finalValues_, finalIdx_, kValue_, validSegs, segmentLen_,
                                                            segmentLen_);
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::StoreTopK(int64_t segStart, uint32_t validSegs)
{
    if (validSegs == 0U) {
        return;
    }
    // Rank-inverse writes final buffers via SIMT VF; BuildOutputs writes them via vector APIs.
    // Wait for the producing VF/vector work before the SIMT topk store reads them.
    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // Output layout is [totalSegs, k]; each sorted segment contributes its first k elements.
    uint64_t outputStart = static_cast<uint64_t>(segStart) * static_cast<uint64_t>(kValue_);
    asc_vf_call<SimtStoreTopKTwoStageBatch<T, OUT_IDX_T>>(
        dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), validSegs, segmentLen_, kValue_, outputStart,
        (__ubuf__ T*)finalValues_.GetPhyAddr(), (__ubuf__ uint32_t*)finalIdx_.GetPhyAddr(),
        (__gm__ volatile T*)outValueGm_.GetPhyAddr(), (__gm__ volatile OUT_IDX_T*)outIdxGm_.GetPhyAddr());
    eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisTwoStage<T, OUT_IDX_T, IsDescend>::StoreNonLastTopK(uint64_t outerId,
                                                                                        uint64_t innerStart,
                                                                                        uint32_t validSegs)
{
    // Output layout is [outer, k, inner]; the topk axis replaces segmentLen with kValue_.
    uint64_t outputBase = outerId * static_cast<uint64_t>(kValue_) * static_cast<uint64_t>(innerSize_);
    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    asc_vf_call<SimtStoreTopKNonLastTwoStageBatch<T, OUT_IDX_T>>(
        dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), validSegs, segmentLen_, kValue_, outputBase, innerStart,
        static_cast<uint64_t>(innerSize_), (__ubuf__ T*)finalValues_.GetPhyAddr(),
        (__ubuf__ uint32_t*)finalIdx_.GetPhyAddr(), (__gm__ volatile T*)outValueGm_.GetPhyAddr(),
        (__gm__ volatile OUT_IDX_T*)outIdxGm_.GetPhyAddr());
    eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

} // namespace topkV2

#endif
