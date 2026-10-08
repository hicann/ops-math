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
 * \file top_k_merge_sort_intra_core.h
 * \brief Intra-core block merge topk for fp32 (inherits from SortMergeIntraCore)
 * \details Inherits from Sort::SortMergeIntraCore and overrides 4 key functions to limit output to topKValue.
 *          Uses this-> to access base class members (cleaner than using declarations).
 *          Reuses 95% of base class code through inheritance.
 */

#ifndef TOP_K_MERGE_SORT_INTRA_CORE_H
#define TOP_K_MERGE_SORT_INTRA_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../sort/arch35/sort_merge_intra_core.h"
#include "../../sort/arch35/sort_tiling_data.h"

namespace topkV2 {
using namespace AscendC;

template <typename ValueType, typename IndexType, bool IsDescend>
class TopKMergeSortIntraCore : public Sort::SortMergeIntraCore<ValueType, IndexType, IsDescend> {
public:
    using Base = Sort::SortMergeIntraCore<ValueType, IndexType, IsDescend>;

    __aicore__ inline TopKMergeSortIntraCore() : Base() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace,
                                const TopKV2TilingDataSimd* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    uint32_t topKValue_ = 0;

    __aicore__ inline uint32_t MergeSingleBatch();
    __aicore__ inline void DoIncrementalMerge(int64_t dstOffsetBase, typename Base::MergeListContext& ctx);
    __aicore__ inline void MergeOneGroup(uint32_t groupStartIdx, uint32_t groupSegCount, uint32_t fullSegElemNum,
                                         uint32_t fullSegSortLen, uint32_t tailSegElemNum, uint32_t pendingSegNum,
                                         uint32_t pingPongRegionSel, uint32_t& passDstOffsetAcc,
                                         uint32_t& groupMergedElemNum);
    __aicore__ inline void CopyRemainingList(typename Base::MergeListContext& ctx, int64_t dstOffsetBase,
                                             uint32_t& dstCumulativeOffset, uint32_t& dstElemCount);
    __aicore__ inline void ExtractAndCopyOut(int64_t batchIdx, uint32_t resultRegion);
};

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace, const TopKV2TilingDataSimd* tilingData, TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }

    topKValue_ = static_cast<uint32_t>(tilingData->topKRealValue);

    SortRegBaseTilingData sortTilingData;
    sortTilingData.unsortedDimNum = tilingData->unsortedDimNum;
    sortTilingData.lastAxisNum = tilingData->lastAxisNum;
    sortTilingData.numTileDataSize = tilingData->numTileDataSize;
    sortTilingData.lastDimTileNum = tilingData->lastDimTileNum;
    sortTilingData.lastDimNeedCore = tilingData->lastDimNeedCore;
    sortTilingData.keyParams0 = tilingData->keyParams0;
    sortTilingData.keyParams3 = tilingData->keyParams3;
    sortTilingData.keyParams4 = tilingData->keyParams4;
    sortTilingData.keyParams5 = tilingData->keyParams5;

    Base::Init(x, value, indices, workspace, &sortTilingData, pipe);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::Process()
{
    // Calculate batch range for this core
    int64_t startBatch = static_cast<int64_t>(this->blockIdx_) * this->batchPerCore_;
    int64_t endBatch = (startBatch + this->batchPerCore_ < this->batchNum_) ? (startBatch + this->batchPerCore_) :
                                                                              this->batchNum_;

    if (startBatch >= this->batchNum_) {
        return; // This core has no work
    }

    // Process each batch through all 3 phases before moving to the next batch.
    // This allows cache (workspace) to be reused across batches within the same core.

    // Precompute buffer sizes (loop-invariant)
    uint32_t mergeBufferSize = Sort::MERGE_LIST_MAX_NUM * this->sortBufferSize_;
    uint32_t extractInSize = AscendC::GetSortLen<ValueType>(this->extractChunkSize_) * sizeof(ValueType);

    for (int64_t batchIdx = startBatch; batchIdx < endBatch; batchIdx++) {
        // ========== Phase 1: Sort blocks in UB ==========
        this->pipe_->InitBuffer(this->inQueueX_, DOUBLE_BUFFER, this->blockSortSize_ * sizeof(ValueType));
        this->pipe_->InitBuffer(this->sortTmpBuf_, this->sortBufferSize_);
        this->pipe_->InitBuffer(this->sortedOutQueue_, DOUBLE_BUFFER, this->sortBufferSize_);
        this->pipe_->InitBuffer(this->indexTmpBuf_, this->blockSortSize_ * sizeof(uint32_t));

        this->SortSingleBatchInUb(this->inputXGm_, batchIdx * this->sortAxisNum_);

        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        this->pipe_->Reset();

        // ========== Phase 2: Merge sorted blocks (4-way) ==========
        this->pipe_->InitBuffer(this->mergeInQueue_, 1, mergeBufferSize);
        this->pipe_->InitBuffer(this->mergeOutQueue_, 1, mergeBufferSize);

        uint32_t resultRegion = MergeSingleBatch();

        eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        this->pipe_->Reset();

        // ========== Phase 3: Extract and copy output ==========
        this->pipe_->InitBuffer(this->extractInQueue_, DOUBLE_BUFFER, extractInSize);
        this->pipe_->InitBuffer(this->outValueQueue_, DOUBLE_BUFFER, this->extractChunkSize_ * sizeof(ValueType));
        this->pipe_->InitBuffer(this->outIdxQueue_, DOUBLE_BUFFER, this->extractChunkSize_ * sizeof(uint32_t));
        if constexpr (IsSameType<int64_t, IndexType>::value) {
            this->pipe_->InitBuffer(this->outIdxInt64Queue_, DOUBLE_BUFFER, this->extractChunkSize_ * sizeof(int64_t));
        }

        ExtractAndCopyOut(batchIdx, resultRegion);

        this->pipe_->Reset();
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline uint32_t TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::MergeSingleBatch()
{
    uint32_t pendingSegNum = this->blocksPerRow_;
    uint32_t fullSegElemNum = this->blockSortSize_;
    uint32_t tailSegElemNum = this->lastBlockSize_;
    uint32_t fullSegSortLen = this->blockSortLen_;

    uint32_t pingPongRegionSel = 0;
    uint32_t mergePassNum = 0;

    while (pendingSegNum > 1) {
        uint32_t passDstOffsetAcc = 0;
        uint32_t nextSegNum = 0;
        uint32_t nextTailSegElemNum = 0;
        uint32_t nextFullSegElemNum = 0;

        for (uint32_t groupStartIdx = 0; groupStartIdx < pendingSegNum; groupStartIdx += Sort::MERGE_LIST_MAX_NUM) {
            uint32_t groupSegCount = (groupStartIdx + Sort::MERGE_LIST_MAX_NUM <= pendingSegNum) ?
                                         Sort::MERGE_LIST_MAX_NUM :
                                         (pendingSegNum - groupStartIdx);

            uint32_t groupMergedElemNum = 0;
            MergeOneGroup(groupStartIdx, groupSegCount, fullSegElemNum, fullSegSortLen, tailSegElemNum, pendingSegNum,
                          pingPongRegionSel, passDstOffsetAcc, groupMergedElemNum);
            if (nextSegNum == 0) {
                nextFullSegElemNum = groupMergedElemNum;
            }
            nextTailSegElemNum = groupMergedElemNum;
            nextSegNum++;
        }

        if (nextSegNum == 0 || nextSegNum >= pendingSegNum) {
            break;
        }

        pendingSegNum = nextSegNum;
        if (pendingSegNum == 1) {
            fullSegElemNum = nextTailSegElemNum;
        } else {
            fullSegElemNum = nextFullSegElemNum;
        }
        fullSegSortLen = AscendC::GetSortLen<ValueType>(fullSegElemNum);
        tailSegElemNum = nextTailSegElemNum;
        pingPongRegionSel = 1 - pingPongRegionSel;
        mergePassNum++;
        event_t barrierEvtId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(barrierEvtId);
        WaitFlag<HardEvent::MTE3_MTE2>(barrierEvtId);
    }

    return (mergePassNum == 0) ? 0 : pingPongRegionSel;
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::MergeOneGroup(
    uint32_t groupStartIdx, uint32_t groupSegCount, uint32_t fullSegElemNum, uint32_t fullSegSortLen,
    uint32_t tailSegElemNum, uint32_t pendingSegNum, uint32_t pingPongRegionSel, uint32_t& passDstOffsetAcc,
    uint32_t& groupMergedElemNum)
{
    typename Base::MergeListContext segListCtx;
    segListCtx.listCount = groupSegCount;

    int64_t srcRegionBase = (pingPongRegionSel == 0) ? 0 : this->batchSortLen_;
    int64_t dstRegionBase = (pingPongRegionSel == 0) ? this->batchSortLen_ : 0;

    for (uint32_t segIdxInGroup = 0; segIdxInGroup < groupSegCount; segIdxInGroup++) {
        uint32_t pendingSegIdx = groupStartIdx + segIdxInGroup;
        segListCtx.srcOffsets[segIdxInGroup] = srcRegionBase + pendingSegIdx * fullSegSortLen;
        segListCtx.elemCounts[segIdxInGroup] = (pendingSegIdx < pendingSegNum - 1) ? fullSegElemNum : tailSegElemNum;
    }

    if (groupSegCount > 1) {
        DoIncrementalMerge(dstRegionBase + passDstOffsetAcc, segListCtx);

        uint32_t groupTotalElemNum = 0;
        for (uint32_t segIdxInGroup = 0; segIdxInGroup < groupSegCount; segIdxInGroup++) {
            groupTotalElemNum += segListCtx.elemCounts[segIdxInGroup];
        }
        groupMergedElemNum = (groupTotalElemNum > topKValue_) ? topKValue_ : groupTotalElemNum; // TopK limitation
        passDstOffsetAcc += AscendC::GetSortLen<ValueType>(groupMergedElemNum);
    } else if (groupSegCount == 1) {
        groupMergedElemNum = (segListCtx.elemCounts[0] > topKValue_) ? topKValue_ :
                                                                       segListCtx.elemCounts[0]; // TopK limitation
        if (segListCtx.elemCounts[0] > 0) {
            int64_t singleSegGmOffset = segListCtx.srcOffsets[0];
            uint32_t remainCopyElemNum = groupMergedElemNum;
            uint32_t srcChunkElemOffset = 0;
            while (remainCopyElemNum > 0) {
                uint32_t copyChunkElemNum = (remainCopyElemNum > this->blockSortSize_) ? this->blockSortSize_ :
                                                                                         remainCopyElemNum;
                if (copyChunkElemNum == 0)
                    break;
                this->CopyBlockChunk(
                    singleSegGmOffset + AscendC::GetSortLen<ValueType>(srcChunkElemOffset),
                    dstRegionBase + passDstOffsetAcc + AscendC::GetSortLen<ValueType>(srcChunkElemOffset),
                    copyChunkElemNum);
                remainCopyElemNum -= copyChunkElemNum;
                srcChunkElemOffset += copyChunkElemNum;
            }
            passDstOffsetAcc += AscendC::GetSortLen<ValueType>(groupMergedElemNum);
        }
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::DoIncrementalMerge(
    int64_t dstOffsetBase, typename Base::MergeListContext& ctx)
{
    if (ctx.listCount > Sort::MERGE_LIST_MAX_NUM)
        return;

    for (uint32_t i = 0; i < ctx.listCount; i++) {
        ctx.remains[i] = ctx.elemCounts[i];
    }

    uint32_t dstCumulativeOffset = 0, dstElemCount = 0, loopGuard = 0;
    while (dstElemCount < topKValue_) { // TopK: early termination
        uint32_t activeLists = 0;
        for (uint32_t i = 0; i < ctx.listCount; i++) {
            if (ctx.remains[i] > 0)
                activeLists++;
        }
        if (activeLists <= 1)
            break;

        LocalTensor<ValueType> ubMainInput = this->mergeInQueue_.template AllocTensor<ValueType>();
        uint16_t elementCountList[Sort::MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
        uint32_t remainListNum = this->LoadListsToUb(ubMainInput, elementCountList, ctx);
        this->mergeInQueue_.EnQue(ubMainInput);

        LocalTensor<ValueType> ubMainInputCalc = this->mergeInQueue_.template DeQue<ValueType>();
        LocalTensor<ValueType> dstLocal = this->mergeOutQueue_.template AllocTensor<ValueType>();

        uint32_t listSortedNums[Sort::MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
        uint32_t mergedCount = this->ExecuteMrgSort(dstLocal, ubMainInputCalc, elementCountList, listSortedNums,
                                                    remainListNum);
        if (mergedCount == 0) {
            this->mergeInQueue_.FreeTensor(ubMainInputCalc);
            this->mergeOutQueue_.FreeTensor(dstLocal);
            break;
        }

        this->mergeOutQueue_.EnQue(dstLocal);
        this->mergeInQueue_.FreeTensor(ubMainInputCalc);

        LocalTensor<ValueType> dstLocalOut = this->mergeOutQueue_.template DeQue<ValueType>();
        uint32_t copyCount = mergedCount;
        if (dstElemCount + copyCount > topKValue_) { // TopK: limit output
            copyCount = topKValue_ - dstElemCount;
        }
        DataCopyExtParams outCopyParams{
            1, static_cast<uint32_t>(AscendC::GetSortLen<ValueType>(copyCount) * sizeof(ValueType)), 0, 0, 0};
        DataCopyPad(this->cacheGm_[dstOffsetBase + dstCumulativeOffset], dstLocalOut, outCopyParams);
        this->mergeOutQueue_.FreeTensor(dstLocalOut);

        uint32_t j = 0;
        for (uint32_t i = 0; i < ctx.listCount; i++) {
            if (ctx.remains[i] > 0) {
                ctx.gmOffsets[i] += AscendC::GetSortLen<ValueType>(listSortedNums[j]);
                ctx.remains[i] -= listSortedNums[j];
                j++;
            }
        }
        dstCumulativeOffset += AscendC::GetSortLen<ValueType>(copyCount);
        dstElemCount += copyCount;
        if (++loopGuard > this->maxMergeIterations_)
            break;
    }

    CopyRemainingList(ctx, dstOffsetBase, dstCumulativeOffset, dstElemCount);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::CopyRemainingList(
    typename Base::MergeListContext& ctx, int64_t dstOffsetBase, uint32_t& dstCumulativeOffset, uint32_t& dstElemCount)
{
    for (uint32_t listIdx = 0; listIdx < ctx.listCount && dstElemCount < topKValue_; listIdx++) { // TopK: early stop
        while (ctx.remains[listIdx] > 0) {
            if (dstElemCount >= topKValue_) {
                break;
            }
            uint32_t loadCount = (ctx.remains[listIdx] > this->blockSortSize_) ? this->blockSortSize_ :
                                                                                 ctx.remains[listIdx];
            uint32_t remainingTopK = topKValue_ - dstElemCount;
            loadCount = (loadCount > remainingTopK) ? remainingTopK : loadCount; // TopK: limit chunk
            if (loadCount == 0)
                break;

            this->CopyBlockChunk(ctx.srcOffsets[listIdx] + ctx.gmOffsets[listIdx], dstOffsetBase + dstCumulativeOffset,
                                 loadCount);

            ctx.gmOffsets[listIdx] += AscendC::GetSortLen<ValueType>(loadCount);
            ctx.remains[listIdx] -= loadCount;
            dstCumulativeOffset += AscendC::GetSortLen<ValueType>(loadCount);
            dstElemCount += loadCount;
        }
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void TopKMergeSortIntraCore<ValueType, IndexType, IsDescend>::ExtractAndCopyOut(int64_t batchIdx,
                                                                                                  uint32_t resultRegion)
{
    int64_t outputOffset = batchIdx * static_cast<int64_t>(topKValue_); // TopK: output offset

    int64_t cacheBatchOffset = (resultRegion == 1) ? this->batchSortLen_ : 0;

    uint32_t totalElem = topKValue_; // TopK: limit output count
    uint32_t elemProcessed = 0;
    uint32_t cacheOffset = 0;

    while (elemProcessed < totalElem) {
        uint32_t elemCount = (elemProcessed + this->extractChunkSize_ <= totalElem) ? this->extractChunkSize_ :
                                                                                      (totalElem - elemProcessed);
        if (elemCount == 0)
            break;

        this->ExtractAndCopyChunk(cacheBatchOffset, cacheOffset, outputOffset, elemProcessed, elemCount);

        elemProcessed += elemCount;
        cacheOffset += AscendC::GetSortLen<ValueType>(elemCount);
    }
}

} // namespace topkV2

#endif // TOP_K_MERGE_SORT_INTRA_CORE_H
