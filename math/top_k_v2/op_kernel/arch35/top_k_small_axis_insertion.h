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
 * @file top_k_small_axis_insertion.h
 * @brief TopK小轴插入排序模板
 * @details 适用场景：轴长极小（N ≤ insertionMaxN），排序语义为TopK候选序
 */

#ifndef TOP_K_SMALL_AXIS_INSERTION_H
#define TOP_K_SMALL_AXIS_INSERTION_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "common/top_k_small_axis_insertion_base.h"
#include "top_k_small_size_bitonic_sort.h"

namespace topkV2 {
using namespace AscendC;

/**
 * @brief SIMT store：last轴，每段写前k个，输出GM布局 [totalSegs, k]
 */
template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T>
__simt_vf__ LAUNCH_BOUND(TopKSmallAxis::INSERTION_THREAD_NUM) __aicore__
    void SimtStoreTopKInsertionBatch(uint32_t validSegs, uint32_t kValue, uint32_t valueRowElems,
                                     uint32_t indexRowElems, uint64_t outputStart, __ubuf__ CONVERT_TYPE* values,
                                     __ubuf__ uint32_t* indices, __gm__ volatile T* outputValue,
                                     __gm__ volatile OUT_IDX_T* outputIndex)
{
    uint32_t total = validSegs * kValue;
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < total; idx += TopKSmallAxis::INSERTION_THREAD_NUM) {
        uint32_t seg = idx / kValue;
        uint32_t rank = idx - seg * kValue;
        outputValue[outputStart + idx] = static_cast<T>(values[seg * valueRowElems + rank]);
        outputIndex[outputStart + idx] = static_cast<OUT_IDX_T>(indices[seg * indexRowElems + rank]);
    }
}

/**
 * @brief SIMT store：非last轴，输出GM布局 [outer, k, inner]
 */
template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T>
__simt_vf__ LAUNCH_BOUND(TopKSmallAxis::INSERTION_THREAD_NUM) __aicore__
    void SimtStoreTopKNonLastInsertionBatch(uint32_t validSegs, uint32_t kValue, uint32_t valueRowElems,
                                            uint32_t indexRowElems, uint64_t outerBaseOffset, uint64_t innerStart,
                                            uint64_t innerSize, __ubuf__ CONVERT_TYPE* values,
                                            __ubuf__ uint32_t* indices, __gm__ volatile T* outputValue,
                                            __gm__ volatile OUT_IDX_T* outputIndex)
{
    uint32_t total = validSegs * kValue;
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < total; idx += TopKSmallAxis::INSERTION_THREAD_NUM) {
        uint32_t rank = idx / validSegs;
        uint32_t seg = idx - rank * validSegs;
        uint64_t gmOffset = outerBaseOffset + static_cast<uint64_t>(rank) * innerSize + innerStart + seg;
        outputValue[gmOffset] = static_cast<T>(values[seg * valueRowElems + rank]);
        outputIndex[gmOffset] = static_cast<OUT_IDX_T>(indices[seg * indexRowElems + rank]);
    }
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
class TopKSmallAxisInsertion
    : public TopKSmallAxis::SmallAxisInsertionBase<TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>, T,
                                                   CONVERT_TYPE, uint32_t, IsDescend,
                                                   TopKSmallAxis::TopKCandidateOrder> {
    using Base = TopKSmallAxis::SmallAxisInsertionBase<TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>, T,
                                                       CONVERT_TYPE, uint32_t, IsDescend,
                                                       TopKSmallAxis::TopKCandidateOrder>;

public:
    __aicore__ inline TopKSmallAxisInsertion() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const TopKV2TilingDataSimd* tilingData,
                                TPipe* pipe);
    __aicore__ inline void Process() { Base::Process(); }

    friend Base;

private:
    using Base::batchNum_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::castBuf_;
    using Base::castRowStride_;
    using Base::indexRowStride_;
    using Base::indices_;
    using Base::pipe_;
    using Base::segmentLen_;
    using Base::segmentsPerBatch_;
    using Base::valueRowStride_;
    using Base::values_;

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

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::Init(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, const TopKV2TilingDataSimd* tilingData, TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    segmentLen_ = tilingData->numTileDataSize;
    segmentsPerBatch_ = tilingData->keyParams0;
    batchNum_ = tilingData->keyParams1;
    kValue_ = static_cast<uint32_t>(tilingData->topKRealValue);
    useBitonicFinalize_ = tilingData->keyParams5 != 0U;
    isNonLastAxis_ = tilingData->keyParams3 != 0U;
    innerLoopNum_ = tilingData->keyParams2;
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

    if (segmentLen_ == 0 || segmentsPerBatch_ == 0 || segmentsPerBatch_ > TopKSmallAxis::MAX_DATACOPY_BLOCK_COUNT ||
        kValue_ == 0 || kValue_ > segmentLen_) {
        return;
    }

    Base::InitInsertionBuffers(pipe);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline bool TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || segmentLen_ == 0 || segmentsPerBatch_ == 0 || kValue_ == 0 ||
           kValue_ > segmentLen_ || (isNonLastAxis_ && (innerLoopNum_ == 0 || outerSize_ <= 0 || innerSize_ <= 0));
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline uint32_t TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ComputeValidSegs(
    uint32_t batchId) const
{
    if (isNonLastAxis_) {
        // Non-last batching iterates inner tiles inside each outer slice; the tail tile
        // may contain fewer segments than the configured batch size.
        uint32_t innerTileId = batchId % innerLoopNum_;
        int64_t innerStart = static_cast<int64_t>(innerTileId) * static_cast<int64_t>(segmentsPerBatch_);
        int64_t innerRemain = innerSize_ - innerStart;
        if (innerRemain <= 0) {
            return 0;
        }
        return innerRemain >= static_cast<int64_t>(segmentsPerBatch_) ? segmentsPerBatch_ :
                                                                        static_cast<uint32_t>(innerRemain);
    }
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(segmentsPerBatch_)) {
        return segmentsPerBatch_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ProcessBatch(uint32_t batchId,
                                                                                                   uint32_t validSegs)
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    uint64_t outerId = 0;
    uint64_t innerStart = 0;
    if (isNonLastAxis_) {
        // batchId is linearized as [outerId, innerTileId] for non-last-axis work.
        outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint32_t innerTileId = batchId % innerLoopNum_;
        innerStart = static_cast<uint64_t>(innerTileId) * static_cast<uint64_t>(segmentsPerBatch_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        Base::LoadNonLastBatch(inputGm_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_), validSegs);
    } else {
        Base::LoadContiguousBatch(inputGm_, segStart * static_cast<int64_t>(segmentLen_), validSegs, false);
    }
    Base::SortBatch(validSegs);
    RunBitonicFinalize(validSegs);
    if (isNonLastAxis_) {
        StoreNonLastTopK(outerId, innerStart, validSegs);
    } else {
        StoreTopK(segStart, validSegs);
    }
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::RunBitonicFinalize(
    uint32_t validSegs)
{
    if (!useBitonicFinalize_ || validSegs == 0U) {
        return;
    }
    RunBitonicFinalizeSelectionRows<CONVERT_TYPE, uint32_t, IsDescend>(values_, indices_, kValue_, validSegs,
                                                                       valueRowStride_, indexRowStride_);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::StoreTopK(int64_t segStart,
                                                                                                uint32_t validSegs)
{
    if (validSegs == 0U) {
        return;
    }
    // Output layout is [totalSegs, k]; each sorted segment contributes its first k elements.
    uint64_t outputStart = static_cast<uint64_t>(segStart) * static_cast<uint64_t>(kValue_);
    asc_vf_call<SimtStoreTopKInsertionBatch<T, CONVERT_TYPE, OUT_IDX_T>>(
        dim3(TopKSmallAxis::INSERTION_THREAD_NUM), validSegs, kValue_, valueRowStride_, indexRowStride_, outputStart,
        (__ubuf__ CONVERT_TYPE*)values_.GetPhyAddr(), (__ubuf__ uint32_t*)indices_.GetPhyAddr(),
        (__gm__ volatile T*)outValueGm_.GetPhyAddr(), (__gm__ volatile OUT_IDX_T*)outIdxGm_.GetPhyAddr());
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void TopKSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::StoreNonLastTopK(
    uint64_t outerId, uint64_t innerStart, uint32_t validSegs)
{
    // Output layout is [outer, k, inner]; the topk axis replaces segmentLen with kValue_.
    uint64_t outputBase = outerId * static_cast<uint64_t>(kValue_) * static_cast<uint64_t>(innerSize_);
    asc_vf_call<SimtStoreTopKNonLastInsertionBatch<T, CONVERT_TYPE, OUT_IDX_T>>(
        dim3(TopKSmallAxis::INSERTION_THREAD_NUM), validSegs, kValue_, valueRowStride_, indexRowStride_, outputBase,
        innerStart, static_cast<uint64_t>(innerSize_), (__ubuf__ CONVERT_TYPE*)values_.GetPhyAddr(),
        (__ubuf__ uint32_t*)indices_.GetPhyAddr(), (__gm__ volatile T*)outValueGm_.GetPhyAddr(),
        (__gm__ volatile OUT_IDX_T*)outIdxGm_.GetPhyAddr());
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

} // namespace topkV2

#endif
