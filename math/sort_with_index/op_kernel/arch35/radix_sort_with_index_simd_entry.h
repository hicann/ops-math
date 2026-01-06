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
 * \file radix_sort_with_index_simd_entry.h
 * \brief radix_sort_with_index_simd_entry kernel entry
 */

#ifndef RADIX_SORT_WITH_INDEX_SIMD_ENTRY_H
#define RADIX_SORT_WITH_INDEX_SIMD_ENTRY_H

#include <cmath>
#include "kernel_operator.h"
#include "radix_block_sort_b8.h"
#include "radix_block_sort_b16.h"
#include "radix_block_sort_b32.h"
#include "radix_block_sort_b64.h"
#include "util_type_simd.h"
#include "constant_var_simd.h"
#include "radix_sort_simd_entry.h"

using namespace AscendC;
template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
struct RadixSortWithIndexSimdEntry : public RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO> {
public:
    __aicore__ inline RadixSortWithIndexSimdEntry()
    {
    }
    __aicore__ inline void Init(GM_ADDR inputValue, GM_ADDR presetIndices, GM_ADDR sortedInputValue, GM_ADDR sortIndex,
                                GM_ADDR workspace, const SortWithIndexTilingDataSimt* tilingData);
    __aicore__ inline void ProcessSort();
    __aicore__ inline void ProcessRadixSort(GlobalTensor<T> inputX, GlobalTensor<int32_t> presetIndices);
    __aicore__ inline void ProcessRadixBlockSort(GlobalTensor<T> inputX, GlobalTensor<int32_t> presetIndices);

protected:
    __aicore__ inline void ScatterKeysGlobal(LocalTensor<T> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                             LocalTensor<int32_t> xInputIndexLocal,
                                             LocalTensor<uint8_t> inputX8BitValue,
                                             LocalTensor<uint16_t> blockExcusiveSum,
                                             LocalTensor<int32_t> blockDataInGlobalPos, uint32_t round,
                                             uint32_t tileDataStart, uint32_t cureTileSize);
    __aicore__ inline void SortAndCopy(GlobalTensor<T> inputX, GlobalTensor<int32_t> presetIndices,
                                       uint64_t tileOffset, uint64_t gmOffset, uint32_t currTileSize,
                                       LocalTensor<T> sortedValueLocal, LocalTensor<uint32_t> sortedValueIndexLocal);

public:
    GlobalTensor<int32_t> presetIndicesGm_;
    static constexpr SortConfig sortConfigMuti_AAAAAA{SortType::RADIX_SORT, false};
    static constexpr SortConfig sortConfigSingle_AAAAAA{SortType::RADIX_SORT, IS_DESCEND};
};

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::Init(
    GM_ADDR inputValue, GM_ADDR presetIndices, GM_ADDR sortedInputValue, GM_ADDR sortIndex, GM_ADDR workspace,
    const SortWithIndexTilingDataSimt* tilingData)
{
    RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::Init(
                                                        inputValue, sortedInputValue, sortIndex, workspace, tilingData);
    this->presetIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)(presetIndices));
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessSort()
{
    for (int32_t i = 0; i < this->sortLoopTimes_; i++) {
        this->sortLoopRound_ = i;
        uint64_t loopOffset = i * this->unsortedDimParallel_ * this->totalDataNum_ * this->oneCoreRowNum_;
        if (this->lastDimRealCore_ == LAST_DIM_SINGLE_CORE_MOD) {
            this->ProcessRadixBlockSort(this->inputValueGm_[loopOffset], this->presetIndicesGm_[loopOffset]);
        } else {
            this->ProcessRadixSort(this->inputValueGm_[loopOffset], this->presetIndicesGm_[loopOffset]);
        }
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void
RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessRadixBlockSort(
    GlobalTensor<T> inputX, GlobalTensor<int32_t> presetIndices)
{
    uint32_t currTileSize = this->numTileData_;
    uint32_t tileId = GetBlockIdx();
    uint32_t unsortedDimIndex =
        (GetBlockIdx() + this->sortLoopRound_ * this->unsortedDimParallel_) * this->oneCoreRowNum_;
    if (unsortedDimIndex >= this->unsortedDimNum_) {
        return;
    }
    // get buffer
    AscendC::LocalTensor<T> sortedValueLocal = this->sortedValueQueue_.template AllocTensor<T>();
    AscendC::LocalTensor<uint32_t> sortedValueIndexLocal =
        this->sortedValueIndexQueue_.template AllocTensor<uint32_t>();
    // offset
    uint64_t gmOffset = this->sortLoopRound_ * this->unsortedDimParallel_ * this->totalDataNum_ * this->oneCoreRowNum_;
    uint64_t tileOffset = tileId * this->numTileData_ * this->oneCoreRowNum_;
    this->SortAndCopy(inputX, presetIndices, tileOffset, gmOffset, currTileSize, sortedValueLocal,
                      sortedValueIndexLocal);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::SortAndCopy(
    GlobalTensor<T> inputX, GlobalTensor<int32_t> presetIndices, uint64_t tileOffset, uint64_t gmOffset,
    uint32_t currTileSize, LocalTensor<T> sortedValueLocal, LocalTensor<uint32_t> sortedValueIndexLocal)
{
    this->CopyDataIn(inputX, tileOffset, currTileSize);
    this->CopyIndexDataIn(presetIndices, tileOffset, currTileSize);

    AscendC::LocalTensor<T> xLocal = this->inQueueX_.template DeQue<T>();
    AscendC::LocalTensor<uint32_t> xIndexLocal = this->inQueueIndex_.template DeQue<uint32_t>();
    AscendC::LocalTensor<uint8_t> shareTmpBuffer = this->sortedShareMemTbuf_.template Get<uint8_t>();
    // need add static
    AscendC::Sort<T, uint32_t, false, sortConfigSingle_AAAAAA>(sortedValueLocal, sortedValueIndexLocal, xLocal,
                                                               xIndexLocal, shareTmpBuffer, currTileSize);
    PipeBarrier<PIPE_ALL>();
    this->sortedValueQueue_.template EnQue<T>(sortedValueLocal);
    this->sortedValueIndexQueue_.template EnQue<uint32_t>(sortedValueIndexLocal);
    this->inQueueX_.FreeTensor(xLocal);
    this->inQueueIndex_.FreeTensor(xIndexLocal);
    // copy result out
    // copy sorted value
    AscendC::LocalTensor<T> sortedValueOutLocal = this->sortedValueQueue_.template DeQue<T>();
    AscendC::DataCopyExtParams dataCopyParamValue{
        static_cast<uint16_t>(1), static_cast<uint32_t>(currTileSize * sizeof(T) * this->oneCoreRowNum_), 0, 0, 0};
    AscendC::DataCopyPad(this->sortedValueGm_[gmOffset + tileOffset], sortedValueOutLocal, dataCopyParamValue);
    this->sortedValueQueue_.FreeTensor(sortedValueOutLocal);
    // copy sorted value index
    AscendC::LocalTensor<int32_t> sortedValueIndexOutLocal = this->sortedValueIndexQueue_.template DeQue<int32_t>();
    AscendC::DataCopyExtParams dataCopyParamIndex{
        static_cast<uint16_t>(1), static_cast<uint32_t>(currTileSize * sizeof(uint32_t) * this->oneCoreRowNum_), 0, 0,
        0};
    AscendC::DataCopyPad(this->sortedValueIndexGm_[gmOffset + tileOffset], sortedValueIndexOutLocal,
                         dataCopyParamIndex);
    this->sortedValueIndexQueue_.FreeTensor(sortedValueIndexOutLocal);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessRadixSort(
    GlobalTensor<T> inputX, GlobalTensor<int32_t> presetIndices)
{
    // get tile num
    const int32_t ubHoldDataCount = this->numTileData_;
    // get global hist/excusive
    int32_t tileCount = (this->totalDataNum_ + ubHoldDataCount - 1) / ubHoldDataCount;
    // new tile id
    uint32_t startTileId = GetBlockIdx() % this->lastDimRealCore_;
    uint32_t unsortedAxisId = GetBlockIdx() / this->lastDimRealCore_;
    uint32_t inputXUnsortedAxisOffset = unsortedAxisId * this->totalDataNum_;
    this->GetGlobalExcusiveSum(inputX, this->totalDataNum_, tileCount, unsortedAxisId, startTileId, inputXUnsortedAxisOffset);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    
    // get local use ub buffer
    LocalTensor<uint8_t> inputX8BitValue = this->inputX8BitValueTbuf_.template Get<uint8_t>();
    LocalTensor<uint8_t> inputX8BitValueCopy = this->inputX8BitValueCopyTbuf_.template Get<uint8_t>();
    LocalTensor<uint16_t> blockExcusive = this->blockExcusiveTbuf_.template Get<uint16_t>();
    LocalTensor<uint32_t> ubFlagTensor = this->blockUbFlagTbuf_.template Get<uint32_t>();
    
    AscendC::LocalTensor<uint8_t> shareTmpBuffer = this->sortedShareMemTbuf_.template Get<uint8_t>();

    uint32_t unsortedDimIndex = unsortedAxisId + this->sortLoopRound_ * this->unsortedDimParallel_;
    uint32_t gmOffset = this->sortLoopRound_ * this->unsortedDimParallel_ * this->totalDataNum_;
    // gm double buffer
    if (NUM_PASS % DOUBLE_BUFFER_FACTOR == 0) {
        this->inputXDoubleBuffer_.SetDoubleBuffer(this->sortedValueGm_[gmOffset], this->sortedValueCopyGm_);
        this->indexDoubleBufferGm_.SetDoubleBuffer(this->sortedValueIndexGm_[gmOffset], this->inputIndexCopyGm_);
    } else {
        this->inputXDoubleBuffer_.SetDoubleBuffer(this->sortedValueCopyGm_, this->sortedValueGm_[gmOffset]);
        this->indexDoubleBufferGm_.SetDoubleBuffer(this->inputIndexCopyGm_, this->sortedValueIndexGm_[gmOffset]);
    }

    // handle sort
    for (int32_t round = 0; round < NUM_PASS; round++) {
        // clear gm value
        this->ClearHistGmBuffer(startTileId, unsortedAxisId, tileCount);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        if (unsortedDimIndex < this->unsortedDimNum_) {
            for (uint32_t tileId = startTileId; tileId < tileCount; tileId += this->lastDimRealCore_) {
                // offset
                uint64_t tileOffset = tileId * this->numTileData_;
                int32_t tileDataStart = tileId * ubHoldDataCount;
                int32_t remainTileDataNum = this->totalDataNum_ - tileDataStart;
                if (remainTileDataNum < 0) {
                    break;
                }
                int32_t currTileSize = SortGetMin<int32_t>(remainTileDataNum, ubHoldDataCount);
                // copy gm to ub
                if (round == 0) {
                    this->CopyDataIn(inputX[inputXUnsortedAxisOffset], tileOffset, currTileSize);
                    this->CopyIndexDataIn(presetIndices[inputXUnsortedAxisOffset], tileOffset, currTileSize);
                } else {
                    this->CopyDataIn(this->inputXDoubleBuffer_.Current()[inputXUnsortedAxisOffset], tileOffset,
                                    currTileSize);
                    this->CopyIndexDataIn(this->indexDoubleBufferGm_.Current()[inputXUnsortedAxisOffset], tileOffset,
                                        currTileSize);
                }

                LocalTensor<T> xLocal = this->inQueueX_.template DeQue<T>();
                LocalTensor<int32_t> xIndexLocal = this->inQueueIndex_.template DeQue<int32_t>();

                // convert singed data to unsinged
                LocalTensor<UNSINGED_TYPE> unsingedInputXData = this->PreProcess(xLocal, currTileSize);
                // get block hist/excusive
                this->GetBlockExcusiveSum(unsingedInputXData, inputX8BitValue, inputX8BitValueCopy, blockExcusive, round,
                                        currTileSize);
                // add hist aggragate mask
                this->ScatterBlockHist2Global(this->blockHist_, this->blockHistFlag_, this->globalHistGm_, tileId,
                                            tileCount);

                // get sort need buffer
                AscendC::LocalTensor<uint8_t> sortedValueLocal = this->sortedValueQueue_.template AllocTensor<uint8_t>();
                AscendC::LocalTensor<uint32_t> sortedValueIndexLocal = this->sortedValueIndexQueue_.template AllocTensor<uint32_t>();
                // need add static
                AscendC::Sort<uint8_t, false, sortConfigMuti_AAAAAA>(sortedValueLocal, sortedValueIndexLocal,
                                                                    inputX8BitValueCopy, shareTmpBuffer,
                                                                    static_cast<uint32_t>(currTileSize));
                PipeBarrier<PIPE_ALL>();
                this->sortedValueQueue_.template EnQue<uint8_t>(sortedValueLocal);
                this->sortedValueIndexQueue_.template EnQue<uint32_t>(sortedValueIndexLocal);
                // not first tile
                if (tileId > 0) {
                    // get key=xxx which block id less and equal to now
                    this->LookbackGlobal(this->blockHistFlag_, this->globalHistGm_, ubFlagTensor, tileId, tileCount);
                }
                // not last tile
                if (tileId < (tileCount - 1)) {
                    // add prefix mask to block hist
                    this->AddPrevfixMask(this->blockHistFlag_, this->globalHistGm_, tileId);
                }
                AscendC::LocalTensor<T> sortedValueTensor = this->sortedValueQueue_.template DeQue<T>();
                AscendC::LocalTensor<uint32_t> sortedIndexTensor = this->sortedValueIndexQueue_.template DeQue<uint32_t>();
                // scatter one sweep result to gm for next round
                this->ScatterKeysGlobal(xLocal, sortedIndexTensor, xIndexLocal, inputX8BitValue, blockExcusive,
                                        this->blockDataInGlobalPos_, round, tileDataStart, currTileSize);

                this->inQueueIndex_.FreeTensor(xIndexLocal);
                this->inQueueX_.FreeTensor(xLocal);
                // release buffer
                this->sortedValueQueue_.FreeTensor(sortedValueTensor);
                this->sortedValueIndexQueue_.FreeTensor(sortedIndexTensor);
            }
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        this->indexDoubleBufferGm_.selector_ ^= 1;
        this->inputXDoubleBuffer_.selector_ ^= 1;
        // core sync
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__
    void CopyOutGmWithIndex(uint32_t round, uint32_t tileDataStart, uint32_t cureTileSize,
                            uint32_t outputXUnsortedAxisOffset, uint32_t unsortedAxisIdOffset,
                            __ubuf__ uint16_t* blockExcusiveSumAddr, __gm__ volatile uint32_t* excusiveBinsGmAddr,
                            __ubuf__ int32_t* blockDataInGlobalPosAddr, __ubuf__ uint32_t* sortedIndexLocalAddr,
                            __ubuf__ int32_t* xInputIndexLocalAddr, __ubuf__ uint8_t* inputX8BitValueAddr,
                            __ubuf__ T* xInputValueLocalAddr, __ubuf__ uint32_t* blockHistFlagAddr,
                            __ubuf__ uint16_t* blockHistAddr, __gm__ volatile uint32_t* indexDoubleBufferGmAddr,
                            __gm__ volatile T* inputXDoubleBufferAddr)
{
    for (int i = Simt::GetThreadIdx(); i < RADIX_SORT_BIN_NUM; i += THREAD_DIM_NUM) {
        // how many data key = i and block id le to now block id
        uint32_t blockHistCumsumVal = blockHistFlagAddr[i];
        blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK;
        // block key=i excusive sum
        uint32_t blockExcusiveSumVal = blockExcusiveSumAddr[i];
        // block key=i num
        uint32_t blockHistVal = blockHistAddr[i];
        // global key<i num
        uint32_t globalKeyOffsetVal = excusiveBinsGmAddr[unsortedAxisIdOffset + i + RADIX_SORT_BIN_NUM * round];
        // now block key=i pos
        // real stand for block data in global pos which not have in block pos
        uint32_t finalpos = globalKeyOffsetVal + blockHistCumsumVal - blockHistVal - blockExcusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    Simt::ThreadBarrier();
    for (int i = Simt::GetThreadIdx(); i < cureTileSize; i += THREAD_DIM_NUM) {
        // i stand for pos
        // sorted lcoal index content  stand for data index
        uint32_t localDataIndex = sortedIndexLocalAddr[i];
        // blockDataInGlobalPos stand for one data in globa pos
        // i stand for data in now block pos
        uint32_t dataFinalGlobalPos = blockDataInGlobalPosAddr[inputX8BitValueAddr[localDataIndex]] + i;
        // store to gm
        inputXDoubleBufferAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = xInputValueLocalAddr[localDataIndex];
        indexDoubleBufferGmAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = xInputIndexLocalAddr[localDataIndex];
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void
RadixSortWithIndexSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ScatterKeysGlobal(
    LocalTensor<T> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<int32_t> xInputIndexLocal,
    LocalTensor<uint8_t> inputX8BitValue, LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<int32_t> blockDataInGlobalPos, uint32_t round, uint32_t tileDataStart, uint32_t cureTileSize)
{
    uint32_t unsortedAxisId = GetBlockIdx() / this->lastDimRealCore_;
    uint32_t outputXUnsortedAxisOffset = unsortedAxisId * this->totalDataNum_;
    uint32_t unsortedAxisIdOffset = unsortedAxisId * RADIX_SORT_BIN_NUM * NUM_PASS;
    Simt::VF_CALL<CopyOutGmWithIndex<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND>>(
        Simt::Dim3(THREAD_DIM_NUM), round, tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unsortedAxisIdOffset,
        (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ volatile uint32_t*)(this->excusiveBinsGm_.GetPhyAddr()),
        (__ubuf__ int32_t*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
        (__ubuf__ int32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(inputX8BitValue.GetPhyAddr()),
        (__ubuf__ T*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ uint32_t*)(this->blockHistFlag_.GetPhyAddr()),
        (__ubuf__ uint16_t*)(this->blockHist_.GetPhyAddr()),
        (__gm__ volatile uint32_t*)(this->indexDoubleBufferGm_.Alternate().GetPhyAddr()),
        (__gm__ volatile T*)(this->inputXDoubleBuffer_.Alternate().GetPhyAddr()));
}

#endif