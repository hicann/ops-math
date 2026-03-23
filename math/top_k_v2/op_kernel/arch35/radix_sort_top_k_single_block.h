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
 * \file radix_sort_top_k_single_block.h
 * \brief radix_sort_top_k_single_blcok impl
 */

#ifndef RADIX_SORT_TOP_K_SINGLE_BLOCK_H
#define RADIX_SORT_TOP_K_SINGLE_BLOCK_H

#include "kernel_operator.h"
#include "radix_sort_top_k_base.h"

#include <algorithm>

using namespace AscendC;

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
struct RadixSortTopKSingleBlock : public RadixSortTopKBase<T, T_INDEX, T_INDEX_TO> {
    __aicore__ inline RadixSortTopKSingleBlock() {};
    __aicore__ inline void Init(
        GM_ADDR inputValue,
        GM_ADDR k,
        GM_ADDR value,
        GM_ADDR indices,
        GM_ADDR workSpace,
        const TopKV2TilingDataSimd* tilingData,
        TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSingleTime(int32_t loopTime);
    __aicore__ inline bool ShouldCurrTimeReturn(int32_t loopTime);
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t coun, uint32_t parallelBatchNum);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t topKValue);

private:
    // 同时处理的行数
    uint32_t unsortedDimParallel_ = 0;
    // UB可以放的Batch轴个数
    uint32_t batchNumInUb_ = 1;
    // 最后一次循环每个核处理的行数
    uint32_t tailLoopBatchNum_ = 0;
    // 最后一次循环的尾行个数
    uint32_t tailBatchNum_ = 0;
    // 给sort用的TQue
    TQue<QuePosition::VECOUT, 1> valuesTmpForSortQue_;
 	TQue<QuePosition::VECOUT, 1> indicesTmpForSortQue_;
};

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlock<T, UNSIGNED_TYPE, IS_LARGEST, IS_SORT, T_INDEX, T_INDEX_TO>::Init(
    GM_ADDR inputValue,
    GM_ADDR k,
    GM_ADDR value,
    GM_ADDR indices,
    GM_ADDR workSpace,
    const TopKV2TilingDataSimd* tilingData,
    TPipe* tPipe)
{
    this->BaseInit(inputValue, k, value, indices, tilingData, tPipe);

    unsortedDimParallel_ = tilingData->unsortedDimParallel;
    batchNumInUb_ = tilingData->batchNumInUb;
    tailLoopBatchNum_ = tilingData->tailLoopBatchNum;
    tailBatchNum_ = tilingData->tailBatchNum;

    this->tPipe_->InitBuffer(this->inputXQue_, 1, batchNumInUb_ * ROUND_UP_AGLIN(this->numTileData_) * sizeof(T));
    this->tPipe_->InitBuffer(this->valuesQue_, 1, batchNumInUb_ * ROUND_UP_AGLIN(this->k_ * sizeof(T)));
    this->tPipe_->InitBuffer(this->indicesQue_, 1, batchNumInUb_ * ROUND_UP_AGLIN(this->k_ * sizeof(T_INDEX_TO)));
    this->tPipe_->InitBuffer(this->valuesTmpForSortQue_, 1, ROUND_UP_AGLIN(this->lastAxisNum_ * sizeof(T)));
 	this->tPipe_->InitBuffer(this->indicesTmpForSortQue_, 1, ROUND_UP_AGLIN(this->lastAxisNum_ * sizeof(uint32_t)));
    this->tPipe_->InitBuffer(this->indicesOutTbuf_, batchNumInUb_ * ROUND_UP_AGLIN(this->k_ * sizeof(int32_t)));
    this->tPipe_->InitBuffer(this->topKApiTmpTBuf_, ROUND_UP_AGLIN(this->topKApiTmpSize_));
}

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlock<T, UNSIGNED_TYPE, IS_LARGEST, IS_SORT, T_INDEX, T_INDEX_TO>::Process()
{
    for(int32_t i = 0; i < this->sortLoopTimes_; i++) {
        ProcessSingleTime(i);
    }
}

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlock<T, UNSIGNED_TYPE, IS_LARGEST, IS_SORT, T_INDEX, T_INDEX_TO>::CopyIn(uint64_t offset, uint32_t count, uint32_t parallelBatchNum)
{
    LocalTensor<T> xLocal = this->inputXQue_.template AllocTensor<T>();
    uint32_t countAlign = ROUND_UP_AGLIN(count);
    T defaultValue = IS_LARGEST ? GetTypeMinValue<T>() : GetTypeMaxValue<T>();
    Duplicate(xLocal, defaultValue, countAlign * parallelBatchNum);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    for (uint32_t i = 0; i < parallelBatchNum; i++) {
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = count * sizeof(T);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(xLocal[i * countAlign], this->inputXGm_[offset + i * count], dataCopyParam, padParams);
    }
    this->inputXQue_.template EnQue(xLocal);
}

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline bool RadixSortTopKSingleBlock<T, UNSIGNED_TYPE, IS_LARGEST, IS_SORT, T_INDEX, T_INDEX_TO>::ShouldCurrTimeReturn(int32_t loopTime) {
    if (this->blockIndex_ >= unsortedDimParallel_) {
        return true;
    }

    // 最后一次循环，没有用满核，直接返回
    if (loopTime == this->sortLoopTimes_ - 1 && tailLoopBatchNum_ == 0 && tailBatchNum_ != 0 && this->blockIndex_ >= tailBatchNum_) {
        return true;
    }

    return false;
} 

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlock<T, UNSIGNED_TYPE, IS_LARGEST, IS_SORT, T_INDEX, T_INDEX_TO>::ProcessSingleTime(int32_t loopTime)
{
    if (ShouldCurrTimeReturn(loopTime)) { return; }

    uint32_t parallelBatchNum = batchNumInUb_;
    uint64_t loopOffset = loopTime * unsortedDimParallel_ * batchNumInUb_ * this->lastAxisNum_;

    uint64_t tileOffset = this->blockIndex_ * parallelBatchNum * this->numTileData_;
    uint64_t outTileOffset = this->blockIndex_ * parallelBatchNum * this->k_;

    // 如果是最后一次循环且是带尾行的情况，循环的Batch轴个数需单独处理
    if (loopTime == this->sortLoopTimes_ - 1 && (tailLoopBatchNum_ != 0 || tailBatchNum_ != 0)) {
        parallelBatchNum = tailLoopBatchNum_;
        parallelBatchNum = this->blockIndex_ < tailBatchNum_ ? parallelBatchNum + 1 : parallelBatchNum;
        tileOffset = this->blockIndex_ < tailBatchNum_ ? this->blockIndex_ * parallelBatchNum * this->numTileData_ :
                                                        (this->blockIndex_ * parallelBatchNum + tailBatchNum_) * this->numTileData_;
        outTileOffset = this->blockIndex_ < tailBatchNum_ ? this->blockIndex_ * parallelBatchNum * this->k_ :
                                                        (this->blockIndex_ * parallelBatchNum + tailBatchNum_) * this->k_;
    }

    CopyIn(loopOffset + tileOffset, this->numTileData_, parallelBatchNum);
    AscendC::LocalTensor<T> xLocal = this->inputXQue_.template DeQue<T>();
    uint32_t aglinNum = ROUND_UP_AGLIN(this->numTileData_);
    uint32_t aglinValuesOffset = ROUND_UP_AGLIN(this->k_ * sizeof(T)) / sizeof(T);
    uint32_t aglinIndicesOffset = ROUND_UP_AGLIN(this->k_ * sizeof(int32_t)) / sizeof(int32_t);
    
    AscendC::LocalTensor<T> valuesLocal = this->valuesQue_.template AllocTensor<T>();
    AscendC::LocalTensor<T_INDEX_TO> indicesLocal = this->indicesQue_.template AllocTensor<T_INDEX_TO>();
    AscendC::LocalTensor<T> valuesTmpLocal = this->valuesTmpForSortQue_.template AllocTensor<T>();
 	AscendC::LocalTensor<uint32_t> indicesTmpLocal = this->indicesTmpForSortQue_.template AllocTensor<uint32_t>();
    AscendC::LocalTensor<uint8_t> tmpBuffer = this->topKApiTmpTBuf_.template Get<uint8_t>();

    bool needsCast = IsSameType<T_INDEX_TO, int64_t>::value;
    AscendC::LocalTensor<int32_t> indicesOutTmp = needsCast ? this->indicesOutTbuf_.template AllocTensor<int32_t>() :
                                                              indicesLocal.template ReinterpretCast<int32_t>();

    static constexpr AscendC::SortConfig config = {AscendC::SortType::RADIX_SORT, IS_LARGEST};
    for (int16_t i = 0; i < parallelBatchNum; i++) {
        AscendC::Sort<T, false, config>(valuesTmpLocal, indicesTmpLocal, xLocal[i * aglinNum], 
            tmpBuffer, static_cast<uint32_t>(this->lastAxisNum_));
        AscendC::Copy(valuesLocal[i * aglinValuesOffset], valuesTmpLocal, static_cast<uint32_t>(this->k_));
        AscendC::Copy(indicesOutTmp[i * aglinIndicesOffset], indicesTmpLocal.template ReinterpretCast<int32_t>(), 
            static_cast<uint32_t>(this->k_));
    }
    if (needsCast) {
        AscendC::Cast(indicesLocal, indicesOutTmp, RoundMode::CAST_NONE,
            static_cast<int32_t>(parallelBatchNum * aglinIndicesOffset));
        this->indicesOutTbuf_.FreeTensor(indicesOutTmp);
    }
    this->valuesQue_.template EnQue<T>(valuesLocal);
    this->indicesQue_.template EnQue<T_INDEX_TO>(indicesLocal);
    uint64_t gmOffset = loopTime * unsortedDimParallel_ * batchNumInUb_ * this->k_;
    CopyOut(gmOffset + outTileOffset, parallelBatchNum);
    this->inputXQue_.template FreeTensor(xLocal);
}

template <typename T, typename UNSIGNED_TYPE, bool IS_LARGEST, bool IS_SORT, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlock<T, UNSIGNED_TYPE, IS_LARGEST, IS_SORT, T_INDEX, T_INDEX_TO>::CopyOut(uint64_t offset, uint32_t parallelBatchNum)
{
    // copy sorted value
    AscendC::LocalTensor<T> valuesLocal = this->valuesQue_.template DeQue<T>();
    AscendC::DataCopyExtParams dataCopyParamValue{static_cast<uint16_t>(parallelBatchNum),
                                                static_cast<uint32_t>(this->k_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(this->valuesGm_[offset], valuesLocal, dataCopyParamValue);
    this->valuesQue_.template FreeTensor(valuesLocal);

    // copy sorted value index, UB_AGLIN_VALUE = 32
    uint32_t aglinIndicesOffset = ROUND_UP_AGLIN(this->k_ * sizeof(int32_t)) / sizeof(int32_t);
    uint32_t blockCastIntervalBytes = (aglinIndicesOffset - this->k_) * sizeof(T_INDEX_TO);
    bool needsCast = IsSameType<T_INDEX_TO, int64_t>::value;
    uint32_t srcStride = needsCast &&  blockCastIntervalBytes >= UB_AGLIN_VALUE && parallelBatchNum >= 2 ? 1 : 0;
    AscendC::LocalTensor<T_INDEX_TO> indicesLocal = this->indicesQue_.template DeQue<T_INDEX_TO>();
    AscendC::DataCopyExtParams dataCopyParamIndex{static_cast<uint16_t>(parallelBatchNum),
                                                static_cast<uint32_t>(this->k_ * sizeof(T_INDEX_TO)), srcStride, 0, 0};
    AscendC::DataCopyPad(this->indicesGm_[offset], indicesLocal, dataCopyParamIndex);
    this->indicesQue_.template FreeTensor(indicesLocal);
}
#endif // RADIX_SORT_TOP_K_SINGLE_BLOCK_H