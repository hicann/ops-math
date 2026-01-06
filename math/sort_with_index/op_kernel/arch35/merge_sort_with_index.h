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
 * \file merge_sort_with_index.h
 * \brief merge_sort_with_index entry
 */
#ifndef MERGE_SORT_WITH_INDEX_H
#define MERGE_SORT_WITH_INDEX_H

#include <cmath>
#include "kernel_operator.h"
#include "merge_sort.h"
#include "merge_sort_with_index_simd.h"

using namespace AscendC;

template <typename T, typename CONVERT_TYPE, typename TILING_DATA_TYPE, bool IS_LARGEST, typename INDEX_TYPE>
struct MergeSortWithIndex : public MergeSort<T, CONVERT_TYPE, TILING_DATA_TYPE, IS_LARGEST, INDEX_TYPE> {
    __aicore__ inline MergeSortWithIndex()
    {
    }
    __aicore__ inline void Init(GM_ADDR inputValue, GM_ADDR presetIndex, GM_ADDR value, GM_ADDR indices,
                                GM_ADDR workSpace, const TILING_DATA_TYPE* tilingData, TPipe* pipe);
    __aicore__ inline void ProcessSort();
    __aicore__ inline void ProcessSingleBlockSort(GlobalTensor<T> inputX, GlobalTensor<uint32_t> presetIndices);
    __aicore__ inline void CopyDataIn(GlobalTensor<T> inputX, GlobalTensor<uint32_t> presetIndices,
                                      LocalTensor<uint32_t> presetIndexLocal, uint64_t tileOffset,
                                      uint32_t currTileSize, uint32_t oneCoreRowNum);

public:
    GlobalTensor<uint32_t> presetIndexGm_;
    // merge sort kernel
    KernelVbsMergeSortWithIndex<T, CONVERT_TYPE, IS_LARGEST> vbsSortMe;
};

template <typename T, typename CONVERT_TYPE, typename TILING_DATA_TYPE, bool IS_LARGEST, typename INDEX_TYPE>
__aicore__ inline void MergeSortWithIndex<T, CONVERT_TYPE, TILING_DATA_TYPE, IS_LARGEST, INDEX_TYPE>::Init(
    GM_ADDR inputValue, GM_ADDR presetIndex, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
    const TILING_DATA_TYPE* tilingData, TPipe* pipe)
{
    this->InitTilingData(tilingData);
    this->InitBuffers(inputValue, value, indices, workSpace, pipe);
    this->presetIndexGm_.SetGlobalBuffer((__gm__ uint32_t*)(presetIndex));
    // vbs init
    vbsSortMe.SetPipe(pipe);
    vbsSortMe.MergeSortInitBuffer(this->numTileData_, this->oneCoreRowNum_, this->mergSortAcApiNeedBufferSize_);
}

template <typename T, typename CONVERT_TYPE, typename TILING_DATA_TYPE, bool IS_LARGEST, typename INDEX_TYPE>
__aicore__ inline void MergeSortWithIndex<T, CONVERT_TYPE, TILING_DATA_TYPE, IS_LARGEST, INDEX_TYPE>::ProcessSort()
{
    for (int32_t i = 0; i < this->sortLoopTimes_; i++) {
        this->sortLoopRound_ = i;
        uint64_t loopOffset = i * this->unsortedDimParallel_ * this->oneCoreRowNum_ * this->numTileData_;
        ProcessSingleBlockSort(this->inputValueGm_[loopOffset], this->presetIndexGm_[loopOffset]);
    }
}

template <typename T, typename CONVERT_TYPE, typename TILING_DATA_TYPE, bool IS_LARGEST, typename INDEX_TYPE>
__aicore__ inline void MergeSortWithIndex<T, CONVERT_TYPE, TILING_DATA_TYPE, IS_LARGEST, INDEX_TYPE>::ProcessSingleBlockSort(
    GlobalTensor<T> inputX, GlobalTensor<uint32_t> presetIndices)
{
    uint32_t tileId = GetBlockIdx();
    uint32_t unsortedDimIndex =
        (GetBlockIdx() + this->sortLoopRound_ * this->unsortedDimParallel_) * this->oneCoreRowNum_;
    if (unsortedDimIndex >= this->unsortedDimNum_) {
        return;
    }
    uint32_t nowCoreRealRowNum = SortGetMin<uint32_t>((this->unsortedDimNum_ - unsortedDimIndex), this->oneCoreRowNum_);
    // get buffer
    AscendC::LocalTensor<T> sortedValueLocal = this->outValueQueue_.template AllocTensor<T>();
    AscendC::LocalTensor<uint32_t> sortedValueIndexLocal = this->outIndexQueue_.template AllocTensor<uint32_t>();
    // offset
    uint64_t tileOffset = tileId * this->numTileData_ * this->oneCoreRowNum_;
    // copy data and presetIndices
    this->CopyDataIn(inputX, presetIndices, sortedValueIndexLocal, tileOffset, this->numTileData_, nowCoreRealRowNum);
    AscendC::LocalTensor<T> xLocal = this->inQueueX_.template DeQue<T>();
    if constexpr (is_same<bfloat16_t, T>::value) {
        this->vbsSortMe.VbsMergeSortBf16(xLocal, sortedValueLocal, sortedValueIndexLocal, this->numTileData_,
                                         nowCoreRealRowNum);
    } else {
        this->vbsSortMe.VbsMergeSort(xLocal, sortedValueLocal, sortedValueIndexLocal, this->numTileData_,
                                   nowCoreRealRowNum);
    }
    this->outValueQueue_.template EnQue<T>(sortedValueLocal);
    this->outIndexQueue_.template EnQue<uint32_t>(sortedValueIndexLocal);
    this->inQueueX_.FreeTensor(xLocal);
    // copy result out
    uint64_t gmOffset =
        this->sortLoopRound_ * this->unsortedDimParallel_ * this->outputLastDimValue_ * this->oneCoreRowNum_;
    uint64_t answerTileOffset = tileId * this->outputLastDimValue_ * this->oneCoreRowNum_;
    this->CopyValue2Gm(gmOffset, answerTileOffset, this->outputLastDimValue_, nowCoreRealRowNum);
}

template <typename T, typename CONVERT_TYPE, typename TILING_DATA_TYPE, bool IS_LARGEST, typename INDEX_TYPE>
__aicore__ inline void MergeSortWithIndex<T, CONVERT_TYPE, TILING_DATA_TYPE, IS_LARGEST, INDEX_TYPE>::CopyDataIn(
    GlobalTensor<T> inputX, GlobalTensor<uint32_t> presetIndices, LocalTensor<uint32_t> presetIndexLocal,
    uint64_t tileOffset, uint32_t currTileSize, uint32_t oneCoreRowNum)
{
    LocalTensor<T> xLocal = this->inQueueX_.template AllocTensor<T>();
    uint32_t aglinOneRowTileSize = ROUND_UP_AGLIN(currTileSize);
    uint32_t localTensorLen = aglinOneRowTileSize * oneCoreRowNum;
    T defaultValue = IS_LARGEST ? static_cast<T>(-INFINITY) : static_cast<T>(NAN);
    Duplicate(xLocal, defaultValue, localTensorLen);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T)) / sizeof(T);
    uint32_t dstStride = ((aglinOneRowTileSize - currTileSizeAlign) * sizeof(T)) / UB_AGLIN_VALUE;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.rightPadding = currTileSizeAlign - currTileSize;
    padParams.paddingValue = static_cast<T>(defaultValue);
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = oneCoreRowNum;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = dstStride;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);

    uint32_t currTileSizeAlign4Indice = ROUND_UP_AGLIN(currTileSize * sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t dstStride4Indice = ((aglinOneRowTileSize - currTileSizeAlign) * sizeof(uint32_t)) / UB_AGLIN_VALUE;
    DataCopyPadExtParams<uint32_t> padParams4Indice;
    padParams4Indice.isPad = true;
    padParams4Indice.rightPadding = currTileSizeAlign4Indice - currTileSize;
    padParams4Indice.paddingValue = static_cast<uint32_t>(0xffffffff);
    DataCopyExtParams dataCopyParam4Indice;
    dataCopyParam4Indice.blockCount = oneCoreRowNum;
    dataCopyParam4Indice.blockLen = currTileSize * sizeof(uint32_t);
    dataCopyParam4Indice.srcStride = 0;
    dataCopyParam4Indice.dstStride = dstStride4Indice;
    DataCopyPad(presetIndexLocal, presetIndices[tileOffset], dataCopyParam4Indice, padParams4Indice);

    this->inQueueX_.EnQue(xLocal);
}

#endif