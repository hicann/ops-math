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
 * \file radix_sort_simd_entry.h
 * \brief sort kernel entry
 */
#ifndef SORT_WITH_INDEX_RADIX_SORT_SIMD_ENTRY_H
#define SORT_WITH_INDEX_RADIX_SORT_SIMD_ENTRY_H
#include <cmath>
#include "kernel_operator.h"
#include "radix_block_sort_b8.h"
#include "radix_block_sort_b16.h"
#include "radix_block_sort_b32.h"
#include "radix_block_sort_b64.h"
#include "util_type_simd.h"
#include "constant_var_simd.h"
using namespace AscendC;
template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
struct RadixSortSimdEntry {
public:
    __aicore__ inline RadixSortSimdEntry() {}
    __aicore__ inline void Init(
        GM_ADDR inputValue,
        GM_ADDR sortedInputValue,
        GM_ADDR sortIndex,
        GM_ADDR workspace,
        const SortWithIndexTilingDataSimt* tilingData);
    __aicore__ inline void InitPara(
        GM_ADDR inputValue,
        GM_ADDR sortedInputValue,
        GM_ADDR sortIndex,
        GM_ADDR workspace,
        const SortWithIndexTilingDataSimt* tilingData);
    __aicore__ inline void CopyDataIn(
        GlobalTensor<T> inputX,
        uint64_t tileOffset,
        uint32_t currTileSize);
    __aicore__ inline void CopyIndexDataIn(
        GlobalTensor<T_INDEX> inputIndex,
        uint64_t tileOffset,
        uint32_t currTileSize);
    __aicore__ inline void CopyGlobalDataIn(
        GlobalTensor<uint32_t> inputX,
        uint64_t tileOffset,
        uint32_t currTileSize);
    __aicore__ inline void ProcessSort();
    __aicore__ inline void ProcessRadixSort(GlobalTensor<T> inputX);
    __aicore__ inline void ProcessRadixBlockSort(GlobalTensor<T> inputX);
protected:
    __aicore__ inline LocalTensor<UNSINGED_TYPE> PreProcess(
        LocalTensor<T> inputX,
        uint32_t numTileData);
    __aicore__ inline void GetGlobalExcusiveSum(
        GlobalTensor<T> inputX,
        uint32_t totalDataNum, uint32_t tileCount, uint32_t unsortedAxisId,
        uint32_t startTileId, uint32_t inputXUnsortedAxisOffset);
    __aicore__ inline void ProcessGlobalExcusiveSum(
        LocalTensor<UNSINGED_TYPE> inputX,
        LocalTensor<T_INDEX> blockExcusive,
        uint32_t numTileData);
    __aicore__ inline void GetBlockExcusiveSum(
        LocalTensor<UNSINGED_TYPE> inputX,
        LocalTensor<uint8_t> inputXBitValue,
        LocalTensor<uint8_t> inputXBitValueCopy,
        LocalTensor<uint16_t> blockExcusive,
        int32_t round,
        uint32_t numTileData);
    __aicore__ inline void ScatterBlockHist2Global(
        LocalTensor<uint16_t> blockHist,
        LocalTensor<uint32_t> blockHistWithFlag,
        GlobalTensor<uint32_t> allblockHistToGm,
        int32_t tileId,
        int32_t totalTileCount);
    __aicore__ inline void LookbackGlobal(
        LocalTensor<uint32_t> nowTileHistBuffer,
        GlobalTensor<uint32_t> allTileHistBuffer,
        LocalTensor<uint32_t> ubFlagTensor,
        int32_t tileId,
        uint32_t numTileData);
    __aicore__ inline void AddPrevfixMask(
        LocalTensor<uint32_t> blockHistWithFlag,
        GlobalTensor<uint32_t> blockHistToGm,
        int32_t tileId);
    template <typename T_INDEX_TMP>
    __aicore__ inline void ScatterKeysGlobal(
        LocalTensor<T> xInputValueLocal,
        LocalTensor<uint32_t> sortedIndexLocal,
        LocalTensor<T_INDEX> xInputIndexLocal,
        LocalTensor<uint8_t> inputX8BitValue,
        LocalTensor<uint16_t> blockExcusiveSum,
        LocalTensor<T_INDEX> blockDataInGlobalPos,
        uint32_t round,
        uint32_t tileDataStart,
        uint32_t cureTileSize);
    __aicore__ inline void ReverseInputData(
        LocalTensor<UNSINGED_TYPE> inputX,
        LocalTensor<UNSINGED_TYPE> reverseInputX,
        uint32_t numTileData);
    __aicore__ inline void ClearHistGmBuffer(
        uint32_t startTileId,
        uint32_t unsortedAxisId,
        uint32_t tileCount);
    __aicore__ inline void SortAndCopy(
        GlobalTensor<T> inputX,
        uint64_t tileOffset,
        uint64_t gmOffset,
        uint32_t currTileSize,
        LocalTensor<T> sortedValueLocal,
        LocalTensor<T_INDEX_TO> sortedValueIndexLocal);
public:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputXCopyTbuf_;
    TBuf<TPosition::VECCALC> blockExcusiveTbuf_;
    TBuf<TPosition::VECCALC> blockHistTbuf_;
    TBuf<TPosition::VECCALC> blockHistWithFlagTbuf_;
    TBuf<TPosition::VECCALC> inputX8BitValueTbuf_;
    TBuf<TPosition::VECCALC> inputX8BitValueCopyTbuf_;
    TBuf<TPosition::VECCALC> inputIndexU8ValueTbuf_;
    TBuf<TPosition::VECCALC> blockDataInGlobalPosTbuf_;
    TBuf<TPosition::VECCALC> blockUbFlagTbuf_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueIndex_;
    TQue<QuePosition::VECIN, 1> inQueueGlobalHist_;
    DoubleBufferSimd<T> inputXDoubleBuffer_;
    DoubleBufferSimd<T_INDEX_TO> indexDoubleBufferGm_;
    LocalTensor<uint32_t> sortedLocalIndex_;
    LocalTensor<UNSINGED_TYPE> inputXCopy_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T_INDEX_TO> sortedValueIndexGm_;
    GlobalTensor<T> sortedValueGm_;
    LocalTensor<uint16_t> blockHist_;
    LocalTensor<uint32_t> blockHistFlag_;
    LocalTensor<T_INDEX> blockDataInGlobalPos_;
    GlobalTensor<uint32_t> globalHistGm_;
    GlobalTensor<T_INDEX> excusiveBinsGm_;
    GlobalTensor<T_INDEX_TO> inputIndexCopyGm_;
    GlobalTensor<T> sortedValueCopyGm_;
    GM_ADDR workspace_;
    uint32_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    uint32_t sortLoopRound_ = 0;
    uint32_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t lastDimTileNum_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t lastDimRealCore_ = 0;
    uint32_t sortAcApiNeedTmpBufferSize_ = 0;
    uint32_t oneCoreRowNum_ = 0;
    // sort ac api buffer
    TBuf<TPosition::VECCALC> sortedShareMemTbuf_;
    TQue<QuePosition::VECOUT, 1> sortedValueQueue_;
    TQue<QuePosition::VECOUT, 1> sortedValueIndexQueue_;
    static constexpr SortConfig sortConfigMuti{SortType::RADIX_SORT, false};
    static constexpr SortConfig sortConfigSingle{SortType::RADIX_SORT, IS_DESCEND};
    static constexpr MicroAPI::CastTrait castTraitB82B16 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };
};

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::Init(
    GM_ADDR inputValue,
    GM_ADDR sortedInputValue,
    GM_ADDR sortIndex,
    GM_ADDR workspace,
    const SortWithIndexTilingDataSimt* tilingData)
{
    // init para
    InitPara(inputValue, sortedInputValue, sortIndex, workspace, tilingData);
    // resue with getGlobalexcusive
    pipe.InitBuffer(blockExcusiveTbuf_, ROUND_UP_AGLIN(RADIX_SORT_BIN_NUM * sizeof(T_INDEX) * NUM_PASS));
    pipe.InitBuffer(blockHistTbuf_, ROUND_UP_AGLIN(RADIX_SORT_BIN_NUM * sizeof(uint16_t)));
    pipe.InitBuffer(blockHistWithFlagTbuf_, ROUND_UP_AGLIN(RADIX_SORT_BIN_NUM * sizeof(uint32_t)));
    pipe.InitBuffer(inputX8BitValueTbuf_, ROUND_UP_AGLIN(numTileData_ * sizeof(uint8_t)));
    pipe.InitBuffer(inputX8BitValueCopyTbuf_, ROUND_UP_AGLIN(numTileData_ * sizeof(uint8_t)));
    pipe.InitBuffer(inputXCopyTbuf_, ROUND_UP_AGLIN(numTileData_ * sizeof(UNSINGED_TYPE)));
    pipe.InitBuffer(blockUbFlagTbuf_, ROUND_UP_AGLIN(RADIX_SORT_BIN_NUM * sizeof(uint32_t)));
    pipe.InitBuffer(blockDataInGlobalPosTbuf_, ROUND_UP_AGLIN(RADIX_SORT_BIN_NUM * sizeof(T_INDEX)));
    inputXCopy_ = inputXCopyTbuf_.Get<UNSINGED_TYPE>();
    blockHist_ = blockHistTbuf_.Get<uint16_t>();
    blockHistFlag_ = blockHistWithFlagTbuf_.Get<uint32_t>();
    blockDataInGlobalPos_ = blockDataInGlobalPosTbuf_.Get<T_INDEX>();
    // new ac api
    pipe.InitBuffer(sortedValueQueue_, 1, ROUND_UP_AGLIN(numTileData_ * sizeof(T)));
    pipe.InitBuffer(sortedValueIndexQueue_, 1, ROUND_UP_AGLIN(numTileData_ * sizeof(T_INDEX_TO)));
    pipe.InitBuffer(sortedShareMemTbuf_, ROUND_UP_AGLIN(sortAcApiNeedTmpBufferSize_));
    // excusive gm
    excusiveBinsGm_.SetGlobalBuffer((__gm__ T_INDEX*)workspace_,
                                    RADIX_SORT_BIN_NUM * NUM_PASS * unsortedDimParallel_);
    // global hist gm
    uint32_t workSpaceOffset = RADIX_SORT_BIN_NUM * NUM_PASS * unsortedDimParallel_ * sizeof(T_INDEX);
    globalHistGm_.SetGlobalBuffer((__gm__ uint32_t*)workspace_ + workSpaceOffset / sizeof(T_INDEX),
                                RADIX_SORT_BIN_NUM * lastDimTileNum_ * unsortedDimParallel_);
    // index copy buffer
    workSpaceOffset += RADIX_SORT_BIN_NUM * lastDimTileNum_ * unsortedDimParallel_ * sizeof(int32_t);
    inputIndexCopyGm_.SetGlobalBuffer((__gm__ T_INDEX_TO*)workspace_ + workSpaceOffset / sizeof(T_INDEX_TO),
                                    totalDataNum_ * unsortedDimParallel_);
    // input data copy
    workSpaceOffset += totalDataNum_ * unsortedDimParallel_ * sizeof(T_INDEX_TO);
    sortedValueCopyGm_.SetGlobalBuffer((__gm__ T*)workspace_ + workSpaceOffset / sizeof(T),
                                        totalDataNum_ * unsortedDimParallel_);
    // pip init
    pipe.InitBuffer(inQueueX_, 1, ROUND_UP_AGLIN(numTileData_ * sizeof(T)));
    pipe.InitBuffer(inQueueIndex_, 1, ROUND_UP_AGLIN(numTileData_ * sizeof(T_INDEX)));
    pipe.InitBuffer(inQueueGlobalHist_, 1, ROUND_UP_AGLIN(RADIX_SORT_BIN_NUM * sizeof(uint32_t)));
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::InitPara(
    GM_ADDR inputValue,
    GM_ADDR sortedInputValue,
    GM_ADDR sortIndex,
    GM_ADDR workspace,
    const SortWithIndexTilingDataSimt* tilingData)
{
    workspace_ = workspace;
    numTileData_ = tilingData->numTileDataSize;
    lastDimRealCore_ = tilingData->lastDimNeedCore;
    totalDataNum_ = tilingData->lastAxisNum;
    unsortedDimNum_ = tilingData->unsortedDimNum;
    sortLoopTimes_ = tilingData->sortLoopTimes;
    lastDimTileNum_  = tilingData->lastDimTileNum;
    unsortedDimParallel_ = tilingData->unsortedDimParallel;
    oneCoreRowNum_ = tilingData->oneCoreRowNum;
    sortAcApiNeedTmpBufferSize_ = tilingData->sortAcApiNeedBufferSize;
    inputValueGm_.SetGlobalBuffer((__gm__ T*)(inputValue));
    sortedValueGm_.SetGlobalBuffer((__gm__ T*)(sortedInputValue));
    sortedValueIndexGm_.SetGlobalBuffer((__gm__ T_INDEX_TO*)(sortIndex));
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessSort()
{
    for(int32_t i = 0; i < sortLoopTimes_; i++) {
        sortLoopRound_ = i;
        uint64_t loopOffset = i * unsortedDimParallel_ * totalDataNum_ * oneCoreRowNum_;
        if (lastDimRealCore_ == LAST_DIM_SINGLE_CORE_MOD) {
            ProcessRadixBlockSort(inputValueGm_[loopOffset]);
        } else {
            ProcessRadixSort(inputValueGm_[loopOffset]);
        }
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessRadixBlockSort(GlobalTensor<T> inputX)
{
    uint32_t currTileSize = numTileData_;
    uint32_t tileId = GetBlockIdx();
    uint32_t unsortedDimIndex = (GetBlockIdx() + sortLoopRound_ * unsortedDimParallel_) * oneCoreRowNum_;
    if (unsortedDimIndex >= unsortedDimNum_) {
        return;
    }
    // get buffer
    AscendC::LocalTensor<T> sortedValueLocal = sortedValueQueue_.AllocTensor<T>();
    AscendC::LocalTensor<T_INDEX_TO> sortedValueIndexLocal = sortedValueIndexQueue_.AllocTensor<T_INDEX_TO>();
    // offset
    uint64_t gmOffset = sortLoopRound_ * unsortedDimParallel_ * totalDataNum_ * oneCoreRowNum_;
    uint64_t tileOffset = tileId * numTileData_ * oneCoreRowNum_;
    SortAndCopy(inputX, tileOffset, gmOffset, currTileSize, sortedValueLocal, sortedValueIndexLocal);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::SortAndCopy(
    GlobalTensor<T> inputX,
    uint64_t tileOffset,
    uint64_t gmOffset,
    uint32_t currTileSize,
    LocalTensor<T> sortedValueLocal,
    LocalTensor<T_INDEX_TO> sortedValueIndexLocal)
{
    CopyDataIn(inputX, tileOffset, currTileSize);
    AscendC::LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    AscendC::LocalTensor<uint8_t> shareTmpBuffer = sortedShareMemTbuf_.Get<uint8_t>();
    if constexpr (IsSameType<T_INDEX_TO, int32_t>::value) {
        // need add static
        AscendC::LocalTensor<uint32_t> sortedValueIndexTmp = sortedValueIndexLocal.template ReinterpretCast<uint32_t>();
        AscendC::Sort<T, false, sortConfigSingle>(
                                            sortedValueLocal,
                                            sortedValueIndexTmp,
                                            xLocal, shareTmpBuffer, currTileSize);
        PipeBarrier<PIPE_ALL>();
    } else {
        AscendC::LocalTensor<uint32_t> sortedValueIndexTmp = inQueueIndex_.AllocTensor<uint32_t>();     // 复用作为临时空间
        AscendC::Sort<T, false, sortConfigSingle>(
                                            sortedValueLocal,
                                            sortedValueIndexTmp,
                                            xLocal, shareTmpBuffer, currTileSize);
        AscendC::LocalTensor<int32_t> sortedValueIndexForCast = sortedValueIndexTmp.template ReinterpretCast<int32_t>();
        AscendC::Cast<int64_t, int32_t>(sortedValueIndexLocal, sortedValueIndexForCast, RoundMode::CAST_NONE, currTileSize);
        inQueueIndex_.FreeTensor(sortedValueIndexTmp);
    }

    sortedValueQueue_.EnQue<T>(sortedValueLocal);
    sortedValueIndexQueue_.EnQue<T_INDEX_TO>(sortedValueIndexLocal);
    inQueueX_.FreeTensor(xLocal);
    // copy result out
    // copy sorted value
    AscendC::LocalTensor<T> sortedValueOutLocal = sortedValueQueue_.DeQue<T>();
    AscendC::DataCopyExtParams dataCopyParamValue{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(currTileSize * sizeof(T) * oneCoreRowNum_),
        0, 0, 0 };
    AscendC::DataCopyPad(sortedValueGm_[gmOffset + tileOffset], sortedValueOutLocal, dataCopyParamValue);
    sortedValueQueue_.FreeTensor(sortedValueOutLocal);
    // copy sorted value index
    AscendC::LocalTensor<T_INDEX_TO> sortedValueIndexOutLocal = sortedValueIndexQueue_.DeQue<T_INDEX_TO>();
    AscendC::DataCopyExtParams dataCopyParamIndex{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(currTileSize * sizeof(T_INDEX_TO) * oneCoreRowNum_),
        0, 0, 0 };
    AscendC::DataCopyPad(sortedValueIndexGm_[gmOffset + tileOffset], sortedValueIndexOutLocal, dataCopyParamIndex);
    sortedValueIndexQueue_.FreeTensor(sortedValueIndexOutLocal);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessRadixSort(GlobalTensor<T> inputX)
{
    // get global hist/excusive
    uint32_t tileCount = (totalDataNum_ + numTileData_ - 1) / numTileData_;
    uint32_t startTileId = GetBlockIdx() % lastDimRealCore_;
    uint32_t unsortedAxisId = GetBlockIdx() / lastDimRealCore_;
    uint32_t inputXUnsortedAxisOffset = unsortedAxisId * totalDataNum_;
    GetGlobalExcusiveSum(inputX, totalDataNum_, tileCount, unsortedAxisId, startTileId, inputXUnsortedAxisOffset);
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    // get tile num
    // get local use ub buffer
    LocalTensor<uint8_t> inputX8BitValue = inputX8BitValueTbuf_.Get<uint8_t>();
    LocalTensor<uint8_t> inputX8BitValueCopy = inputX8BitValueCopyTbuf_.Get<uint8_t>();
    LocalTensor<uint16_t> blockExcusive = blockExcusiveTbuf_.Get<uint16_t>();
    LocalTensor<uint32_t> ubFlagTensor = blockUbFlagTbuf_.Get<uint32_t>();
    // get sort need buffer
    AscendC::LocalTensor<uint8_t> shareTmpBuffer = sortedShareMemTbuf_.Get<uint8_t>();

    uint32_t unsortedDimIndex = unsortedAxisId + sortLoopRound_ * unsortedDimParallel_;
    uint32_t gmOffset = sortLoopRound_ * unsortedDimParallel_ * totalDataNum_;
    // gm double buffer
    if constexpr(sizeof(T) == sizeof(int8_t)) {
        inputXDoubleBuffer_.SetDoubleBuffer(sortedValueCopyGm_, sortedValueGm_[gmOffset]);
        indexDoubleBufferGm_.SetDoubleBuffer(inputIndexCopyGm_, sortedValueIndexGm_[gmOffset]);
    } else {
        inputXDoubleBuffer_.SetDoubleBuffer(sortedValueGm_[gmOffset], sortedValueCopyGm_);
        indexDoubleBufferGm_.SetDoubleBuffer(sortedValueIndexGm_[gmOffset], inputIndexCopyGm_);
    }
    // handle sort
    for(int32_t round = 0; round < NUM_PASS; round++) {
        // clear gm value
        ClearHistGmBuffer(startTileId, unsortedAxisId, tileCount);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        if (unsortedDimIndex < unsortedDimNum_) {
            for(uint32_t tileId = startTileId; tileId < tileCount; tileId += lastDimRealCore_) {
                // offset
                uint64_t tileOffset = tileId * numTileData_;
                int32_t tileDataStart = tileId * numTileData_;
                int32_t remainTileDataNum = totalDataNum_ - tileDataStart;
                if (remainTileDataNum < 0) {
                    break;
                }
                int32_t currTileSize = SortGetMin<int32_t>(remainTileDataNum, numTileData_);
                // copy gm to ub
                if (round == 0) {
                    CopyDataIn(inputX[inputXUnsortedAxisOffset], tileOffset, currTileSize);
                } else {
                    CopyDataIn(inputXDoubleBuffer_.Current()[inputXUnsortedAxisOffset],
                               tileOffset, currTileSize);
                }
                LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
                // convert singed data to unsinged
                LocalTensor<UNSINGED_TYPE> unsingedInputXData = PreProcess(xLocal, currTileSize);
                // get block hist/excusive
                GetBlockExcusiveSum(unsingedInputXData, inputX8BitValue, inputX8BitValueCopy,
                                    blockExcusive, round, currTileSize);
                // add hist aggragate mask
                ScatterBlockHist2Global(blockHist_, blockHistFlag_, globalHistGm_, tileId, tileCount);
                // need add static
                // get sort need buffer
                AscendC::LocalTensor<uint32_t> sortedValueIndexLocal = sortedValueIndexQueue_.AllocTensor<uint32_t>();
                AscendC::LocalTensor<uint8_t> sortedValueLocal = sortedValueQueue_.AllocTensor<uint8_t>();
                AscendC::Sort<uint8_t, false, sortConfigMuti>(
                                                    sortedValueLocal,
                                                    sortedValueIndexLocal,
                                                    inputX8BitValueCopy,
                                                    shareTmpBuffer,
                                                    static_cast<uint32_t>(currTileSize));
                PipeBarrier<PIPE_ALL>();
                sortedValueQueue_.FreeTensor(sortedValueLocal);
                sortedValueIndexQueue_.EnQue<uint32_t>(sortedValueIndexLocal);
                // not first tile
                if (tileId > 0) {
                    // get key=xxx which block id less and equal to now
                    LookbackGlobal(blockHistFlag_, globalHistGm_, ubFlagTensor, tileId, tileCount);
                }
                // not last tile
                if (tileId < (tileCount - 1)) {
                    // add prefix mask to block hist. 打上OK状态位，高2bit
                    AddPrevfixMask(blockHistFlag_, globalHistGm_, tileId);
                }
                LocalTensor<T_INDEX> xIndexLocal;
                if (round != 0) {
                    // copy input index
                    GlobalTensor<T_INDEX> doubleBufferTmp = (indexDoubleBufferGm_.Current()).template ReinterpretCast<T_INDEX>();
                    CopyIndexDataIn(doubleBufferTmp[inputXUnsortedAxisOffset],
                                    tileOffset, currTileSize);
                    xIndexLocal = inQueueIndex_.DeQue<T_INDEX>();
                }
                AscendC::LocalTensor<uint32_t> sortedIndexTensor = sortedValueIndexQueue_.DeQue<uint32_t>();
                // scatter one sweep result to gm for next round
                // 当尾轴长度在int32最大值范围内，中间的index相关计算依旧用int32，仅在最后一次搬出时转换为int64
                if (round < NUM_PASS - 1) {
                    ScatterKeysGlobal<T_INDEX>(
                        xLocal,
                        sortedIndexTensor,
                        xIndexLocal,
                        inputX8BitValue,
                        blockExcusive,
                        blockDataInGlobalPos_,
                        round,
                        tileDataStart,
                        currTileSize);
                } else {
                    ScatterKeysGlobal<T_INDEX_TO>(
                        xLocal,
                        sortedIndexTensor,
                        xIndexLocal,
                        inputX8BitValue,
                        blockExcusive,
                        blockDataInGlobalPos_,
                        round,
                        tileDataStart,
                        currTileSize);
                }
                if (round != 0) {
                    inQueueIndex_.FreeTensor(xIndexLocal);
                }
                inQueueX_.FreeTensor(xLocal);
                // release buffer
                sortedValueIndexQueue_.FreeTensor(sortedIndexTensor);
            }
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        indexDoubleBufferGm_.selector_ ^= 1;
        inputXDoubleBuffer_.selector_ ^= 1;
        // core sync
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ClearHistGmBuffer(
        uint32_t startTileId, uint32_t unsortedAxisId, uint32_t tileCount)
{
    uint32_t unsortedAxisIdOffset = unsortedAxisId * RADIX_SORT_BIN_NUM * lastDimTileNum_;
    for(uint32_t tileId = startTileId; tileId < tileCount; tileId += lastDimRealCore_) {
        InitOutput(globalHistGm_[unsortedAxisIdOffset + RADIX_SORT_BIN_NUM * tileId],
                   RADIX_SORT_BIN_NUM,
                   CLEAR_UB_VALUE);
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::CopyDataIn(
    GlobalTensor<T> inputX,
    uint64_t tileOffset,
    uint32_t currTileSize)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T)) / sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.rightPadding = currTileSizeAlign - currTileSize;
    padParams.paddingValue = static_cast<T>(0);
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::CopyIndexDataIn(
    GlobalTensor<T_INDEX> inputIndex,
    uint64_t tileOffset,
    uint32_t currTileSize)
{
    LocalTensor<T_INDEX> xLocal = inQueueIndex_.AllocTensor<T_INDEX>();
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T_INDEX)) / sizeof(T_INDEX);
    DataCopyPadExtParams<T_INDEX> padParams;
    padParams.isPad = true;
    padParams.rightPadding = currTileSizeAlign - currTileSize;
    padParams.paddingValue = static_cast<T_INDEX>(0);
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T_INDEX);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputIndex[tileOffset], dataCopyParam, padParams);
    inQueueIndex_.EnQue(xLocal);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::CopyGlobalDataIn(
    GlobalTensor<uint32_t> inputX,
    uint64_t tileOffset,
    uint32_t currTileSize)
{
    LocalTensor<uint32_t> xLocal = inQueueGlobalHist_.AllocTensor<uint32_t>();
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(uint32_t)) / sizeof(uint32_t);
    DataCopyPadExtParams<uint32_t> padParams;
    padParams.isPad = true;
    padParams.rightPadding = currTileSizeAlign - currTileSize;
    padParams.paddingValue = static_cast<uint32_t>(0);
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(uint32_t);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
    inQueueGlobalHist_.EnQue(xLocal);
}

template <int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__ void CopyExcusiveSumToGmB64(uint32_t excusiveBinOffset,
    __gm__ T_INDEX *excusiveBinsGm, __ubuf__ T_INDEX *blockExcusiveBuffer)
{
    for (int i = Simt::GetThreadIdx(); i < RADIX_SORT_BIN_NUM * NUM_PASS / 2; i += THREAD_DIM_NUM) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint32_t offset = i + j * RADIX_SORT_BIN_NUM * NUM_PASS / 2;
            T_INDEX srcData = blockExcusiveBuffer[offset];
            Simt::AtomicAdd<T_INDEX>(excusiveBinsGm + excusiveBinOffset + offset, srcData);
        }
    }
}

template <int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__ void CopyExcusiveSumToGmB8B16B32(uint32_t excusiveBinOffset,
    __gm__ T_INDEX *excusiveBinsGm, __ubuf__ T_INDEX *blockExcusiveBuffer)
{
    for (int i = Simt::GetThreadIdx(); i < RADIX_SORT_BIN_NUM * NUM_PASS; i += THREAD_DIM_NUM) {
        uint32_t offset = i;
        T_INDEX srcData = blockExcusiveBuffer[offset];
        Simt::AtomicAdd<T_INDEX>(excusiveBinsGm + excusiveBinOffset + offset, srcData);
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::GetGlobalExcusiveSum(
    GlobalTensor<T> inputX, uint32_t totalDataNum, uint32_t tileCount, uint32_t unsortedAxisId,
    uint32_t startTileId, uint32_t inputXUnsortedAxisOffset)
{
    int32_t clearGmCore = GetBlockNum() / lastDimRealCore_;
    // clear gm buffer
    if (GetBlockIdx() < clearGmCore) {
        uint32_t clearOffset = GetBlockIdx() * RADIX_SORT_BIN_NUM * NUM_PASS;
        InitOutput(excusiveBinsGm_[clearOffset], (RADIX_SORT_BIN_NUM * NUM_PASS), static_cast<T_INDEX>(CLEAR_UB_VALUE));
    }
    SyncAll();
    uint32_t excusiveBinOffset = unsortedAxisId * RADIX_SORT_BIN_NUM * NUM_PASS;
    LocalTensor<T_INDEX> blockExcusiveBuffer = blockExcusiveTbuf_.Get<T_INDEX>();
    // clear ub buffer
    Duplicate(blockExcusiveBuffer, static_cast<T_INDEX>(CLEAR_UB_VALUE), RADIX_SORT_BIN_NUM * NUM_PASS);
    for(uint32_t tileId = startTileId; tileId < tileCount; tileId += lastDimRealCore_) {
        // offset
        uint64_t tileOffset = tileId * numTileData_;
        int32_t tileDataStart = tileId * numTileData_;
        int32_t remainTileDataNum = totalDataNum - tileDataStart;
        if (remainTileDataNum < 0) {
            break;
        }
        int32_t currTileNum = SortGetMin<int32_t>(remainTileDataNum, numTileData_);
        // copy gm to ub
        CopyDataIn(inputX[inputXUnsortedAxisOffset], tileOffset, currTileNum);
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        // convert singed data to unsinged
        LocalTensor<UNSINGED_TYPE> unsingedInputXData = PreProcess(xLocal, currTileNum);
        // get global excusive
        ProcessGlobalExcusiveSum(unsingedInputXData, blockExcusiveBuffer, currTileNum);
        inQueueX_.FreeTensor(xLocal);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    if constexpr (IsSameType<T_INDEX, int64_t>::value) {
        if constexpr (NUM_PASS == B64_BITE_SIZE) {
            Simt::VF_CALL<CopyExcusiveSumToGmB64<NUM_PASS, T_INDEX, T_INDEX_TO>>(Simt::Dim3(THREAD_DIM_NUM), excusiveBinOffset,
                (__gm__ T_INDEX *)(excusiveBinsGm_.GetPhyAddr()),
                (__ubuf__ T_INDEX *)(blockExcusiveBuffer.GetPhyAddr()));
        } else {
            Simt::VF_CALL<CopyExcusiveSumToGmB8B16B32<NUM_PASS, T_INDEX, T_INDEX_TO>>(Simt::Dim3(THREAD_DIM_NUM), excusiveBinOffset,
                (__gm__ T_INDEX *)(excusiveBinsGm_.GetPhyAddr()),
                (__ubuf__ T_INDEX *)(blockExcusiveBuffer.GetPhyAddr()));
        }
    } else {
        SetAtomicAdd<int32_t>();
        // copy ub to gm
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_BIN_NUM * NUM_PASS * sizeof(uint32_t);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(excusiveBinsGm_[excusiveBinOffset], blockExcusiveBuffer, dataCopyParam);
        SetAtomicNone();
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ProcessGlobalExcusiveSum(
    LocalTensor<UNSINGED_TYPE> inputX,
    LocalTensor<T_INDEX> blockExcusive,
    uint32_t numTileData)
{
    if constexpr (is_same<int32_t, T>::value || is_same<uint32_t, T>::value
                  || is_same<float, T>::value) {
        RadixBlockSortSimdB32<T, uint32_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetGlobalExcusiveSum(inputX, blockExcusive, numTileData);
    } else if constexpr (is_same<half, T>::value || is_same<uint16_t, T>::value
                         || is_same<int16_t, T>::value || is_same<bfloat16_t, T>::value) {
        RadixBlockSortSimdB16<T, uint16_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetGlobalExcusiveSum(inputX, blockExcusive, numTileData);
    } else if constexpr (is_same<int8_t, T>::value || is_same<uint8_t, T>::value) {
        RadixBlockSortSimdB8<T, uint8_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetGlobalExcusiveSum(inputX, blockExcusive, numTileData);
    } else if constexpr (is_same<int64_t, T>::value || is_same<uint64_t, T>::value) {
        RadixBlockSortSimdB64<T, uint64_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetGlobalExcusiveSum(inputX, blockExcusive, numTileData);
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline LocalTensor<UNSINGED_TYPE> RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::PreProcess(
    LocalTensor<T> inputX,
    uint32_t numTileData)
{
    if constexpr (is_same<int32_t, T>::value) {
        RadixBlockSortSimdB32<T, uint32_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.TwiddleInB32(inputX, inputXCopy_, numTileData);
        return inputXCopy_;
    } else if constexpr (is_same<half, T>::value || is_same<bfloat16_t, T>::value) {
        RadixBlockSortSimdB16<T, uint16_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.TwiddleInFp16(inputX, inputXCopy_, numTileData);
        return inputXCopy_;
    } else if constexpr (is_same<float, T>::value) {
        RadixBlockSortSimdB32<T, uint32_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.TwiddleInFp32(inputX, inputXCopy_, numTileData);
        return inputXCopy_;
    } else if constexpr (is_same<int16_t, T>::value) {
        RadixBlockSortSimdB16<T, uint16_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.TwiddleInB16(inputX, inputXCopy_, numTileData);
        return inputXCopy_;
    } else if constexpr (is_same<int8_t, T>::value) {
        RadixBlockSortSimdB8<T, uint8_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.TwiddleInB8(inputX, inputXCopy_, numTileData);
        return inputXCopy_;
    } else if constexpr (is_same<int64_t, T>::value) {
        RadixBlockSortSimdB64<T, uint64_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.TwiddleInB64(inputX, inputXCopy_, numTileData);
        return inputXCopy_;
    } else {
        if (IS_DESCEND) {
            ReverseInputData(inputX, inputXCopy_, numTileData);
            return inputXCopy_;
        }
        return inputX;
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ReverseInputData(
    LocalTensor<UNSINGED_TYPE> inputX,
    LocalTensor<UNSINGED_TYPE> reverseInputX,
    uint32_t numTileData)
{
    if constexpr (is_same<uint32_t, T>::value) {
        RadixBlockSortSimdB32<T, uint32_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.ReverseInputData(inputX, reverseInputX, numTileData);
    } else if constexpr (is_same<uint16_t, T>::value) {
        RadixBlockSortSimdB16<T, uint16_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.ReverseInputData(inputX, reverseInputX, numTileData);
    } else if constexpr (is_same<uint8_t, T>::value) {
        RadixBlockSortSimdB8<T, uint8_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.ReverseInputData(inputX, reverseInputX, numTileData);
    } else if constexpr (is_same<uint64_t, T>::value) {
        RadixBlockSortSimdB64<T, uint64_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.ReverseInputData(inputX, reverseInputX, numTileData);
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::GetBlockExcusiveSum(
    LocalTensor<UNSINGED_TYPE> inputX,
    LocalTensor<uint8_t> inputXBitValue,
    LocalTensor<uint8_t> inputXBitValueCopy,
    LocalTensor<uint16_t> blockExcusive,
    int32_t round,
    uint32_t numTileData)
{
    if constexpr (is_same<int32_t, T>::value || is_same<uint32_t, T>::value
                  || is_same<float, T>::value) {
        RadixBlockSortSimdB32<T, uint32_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetBlockExcusiveSum(inputX, inputXBitValue, inputXBitValueCopy, blockExcusive,
                                           blockHist_, round, numTileData);
    }  else if constexpr (is_same<half, T>::value || is_same<uint16_t, T>::value
                          || is_same<int16_t, T>::value || is_same<bfloat16_t, T>::value) {
        RadixBlockSortSimdB16<T, uint16_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetBlockExcusiveSum(inputX, inputXBitValue, inputXBitValueCopy, blockExcusive,
                                           blockHist_, round, numTileData);
    } else if constexpr (is_same<int8_t, T>::value || is_same<uint8_t, T>::value ) {
        RadixBlockSortSimdB8<T, uint8_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetBlockExcusiveSum(inputX, inputXBitValue, inputXBitValueCopy, blockExcusive,
                                           blockHist_, round, numTileData);
    } else if constexpr (is_same<int64_t, T>::value || is_same<uint64_t, T>::value ) {
        RadixBlockSortSimdB64<T, uint64_t, NUM_PASS, IS_DESCEND, T_INDEX> radixBlockSort;
        radixBlockSort.GetBlockExcusiveSum(inputX, inputXBitValue, inputXBitValueCopy, blockExcusive,
                                           blockHist_, round, numTileData);
    }
}

template <typename T, typename T_INDEX, typename T_INDEX_TMP>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__ void CopyOutGm(
    uint32_t round,
    uint32_t tileDataStart,
    uint32_t cureTileSize,
    uint32_t outputXUnsortedAxisOffset,
    uint32_t unsortedAxisIdOffset,
    __ubuf__ uint16_t *blockExcusiveSumAddr,            // 当前tile块直方图局部前缀和
    __gm__ volatile T_INDEX *excusiveBinsGmAddr,                 // 全局前缀和（适配int64）
    __ubuf__ T_INDEX *blockDataInGlobalPosAddr,         // 临时UB空间，计算地址偏移（适配int64_t)
    __ubuf__ uint32_t *sortedIndexLocalAddr,            // 8bit排序后的idx
    __ubuf__ T_INDEX *xInputIndexLocalAddr,             // 8bit在workspace的idx
    __ubuf__ uint8_t *inputX8BitValueAddr,              // 8bit在workspace的value
    __ubuf__ T *xInputValueLocalAddr,                   // tile块的X值
    __ubuf__ uint32_t *blockHistFlagAddr,               // blockHistFlag_(适配int64_t)
    __ubuf__ uint16_t *blockHistAddr,                   // blockHist_, 当前tile块直方图统计
    __gm__ volatile T_INDEX_TMP *indexDoubleBufferGmAddr,        // 输出workspcae的idx
    __gm__ volatile T *inputXDoubleBufferAddr)                   // 输出workspace的value
{
    for (int i = Simt::GetThreadIdx(); i < RADIX_SORT_BIN_NUM; i += THREAD_DIM_NUM) {
        // how many data key = i and block id le to now block id
        uint32_t blockHistCumsumVal = blockHistFlagAddr[i];
        // 高2比特为状态位
        blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK;
        // block key=i excusive sum
        uint32_t blockExcusiveSumVal = blockExcusiveSumAddr[i];
        // block key=i num
        uint32_t blockHistVal = blockHistAddr[i];
        // global key<i num
        T_INDEX globalKeyOffsetVal = excusiveBinsGmAddr[unsortedAxisIdOffset + i + RADIX_SORT_BIN_NUM * round];
        // now block key=i pos
        // real stand for block data in global pos which not have in block pos
        T_INDEX finalpos = globalKeyOffsetVal + blockHistCumsumVal -  blockHistVal - blockExcusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    Simt::ThreadBarrier();
    for (int i = Simt::GetThreadIdx(); i < cureTileSize; i += THREAD_DIM_NUM) {
        // i stand for pos
        // sorted lcoal index content  stand for data index
        // 本地排序后的数据索引
        uint32_t localDataIndex = sortedIndexLocalAddr[i];
        T_INDEX_TMP dataInitIndex = 0;
        if (round != 0) {
            dataInitIndex = xInputIndexLocalAddr[localDataIndex];
        } else {
            dataInitIndex = tileDataStart + localDataIndex;
        }

        // blockDataInGlobalPos stand for one data in globa pos
        // i stand for data in now block pos
        T_INDEX dataFinalGlobalPos = blockDataInGlobalPosAddr[inputX8BitValueAddr[localDataIndex]] + i;
        // store to gm
        inputXDoubleBufferAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset]
            = xInputValueLocalAddr[localDataIndex];
        indexDoubleBufferGmAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = dataInitIndex;
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
template <typename T_INDEX_TMP>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ScatterKeysGlobal(
    LocalTensor<T> xInputValueLocal,
    LocalTensor<uint32_t> sortedIndexLocal,
    LocalTensor<T_INDEX> xInputIndexLocal,
    LocalTensor<uint8_t> inputX8BitValue,
    LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<T_INDEX> blockDataInGlobalPos,
    uint32_t round,
    uint32_t tileDataStart,
    uint32_t cureTileSize)
{
    uint32_t unsortedAxisId = GetBlockIdx() / lastDimRealCore_;             // 0
    uint32_t outputXUnsortedAxisOffset = unsortedAxisId * totalDataNum_;    // 0
    uint32_t unsortedAxisIdOffset = unsortedAxisId * RADIX_SORT_BIN_NUM * NUM_PASS;
    Simt::VF_CALL<CopyOutGm<T, T_INDEX, T_INDEX_TMP>>(Simt::Dim3(THREAD_DIM_NUM),
        round, tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unsortedAxisIdOffset,
        (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
        (__gm__ T_INDEX *)(excusiveBinsGm_.GetPhyAddr()),
        (__ubuf__ T_INDEX *)(blockDataInGlobalPos.GetPhyAddr()),
        (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()),
        (__ubuf__ T_INDEX *)(xInputIndexLocal.GetPhyAddr()),
        (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()),
        (__ubuf__ T *)(xInputValueLocal.GetPhyAddr()),
        (__ubuf__ uint32_t *)(blockHistFlag_.GetPhyAddr()),
        (__ubuf__ uint16_t *)(blockHist_.GetPhyAddr()),
        (__gm__ T_INDEX_TMP *)(indexDoubleBufferGm_.Alternate().GetPhyAddr()),
        (__gm__ T *)(inputXDoubleBuffer_.Alternate().GetPhyAddr()));
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::ScatterBlockHist2Global(
    LocalTensor<uint16_t> blockHist,
    LocalTensor<uint32_t> blockHistWithFlag,
    GlobalTensor<uint32_t> allblockHistToGm,
    int32_t tileId,
    int32_t totalTileCount)
{
    uint32_t unsortedAxisId = GetBlockIdx() / lastDimRealCore_;
    uint32_t unsortedAxisIdOffset = unsortedAxisId * lastDimTileNum_ * RADIX_SORT_BIN_NUM;
    __local_mem__ uint16_t* blockHistPtr = (__ubuf__ uint16_t*)blockHist.GetPhyAddr();
    __local_mem__ uint32_t* blockHistWithFlagPtr = (__ubuf__ uint32_t*)blockHistWithFlag.GetPhyAddr();
    // 软同步，写入状态位2bit
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint32_t> aggregateReadyMask;
        MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK, predicateDefault);
        MicroAPI::RegTensor<uint16_t> blockHistZero, blockHistOne;
        MicroAPI::RegTensor<uint16_t> lookaheadOutZero, lookaheadOutOne, lookaheadOutTwo, lookaheadOutThree;
        MicroAPI::RegTensor<uint16_t> zeroVector;
        MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistZero, blockHistPtr, ONE_TIMES_B16_NUM);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistOne, blockHistPtr, ONE_TIMES_B16_NUM);
        // cast b16 to B32
        MicroAPI::Interleave(lookaheadOutZero, lookaheadOutOne, blockHistZero, zeroVector);
        MicroAPI::Interleave(lookaheadOutTwo, lookaheadOutThree, blockHistOne, zeroVector);
        // add mask
        MicroAPI::RegTensor<uint32_t> lookaheadOutZeroMask, lookaheadOutOneMask,
                                      lookaheadOutTwoMask, lookaheadOutThreeMask;
        MicroAPI::Or(lookaheadOutZeroMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutZero,
                    aggregateReadyMask, predicateDefault);
        MicroAPI::Or(lookaheadOutOneMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutOne,
                    aggregateReadyMask, predicateDefault);
        MicroAPI::Or(lookaheadOutTwoMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutTwo,
                    aggregateReadyMask, predicateDefault);
        MicroAPI::Or(lookaheadOutThreeMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutThree,
                    aggregateReadyMask, predicateDefault);
        // vst
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                                                                              lookaheadOutZeroMask,
                                                                              ONE_TIMES_B32_NUM,
                                                                              predicateDefault);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                                                                              lookaheadOutOneMask,
                                                                              ONE_TIMES_B32_NUM,
                                                                              predicateDefault);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                                                                              lookaheadOutTwoMask,
                                                                              ONE_TIMES_B32_NUM,
                                                                              predicateDefault);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                                                                              lookaheadOutThreeMask,
                                                                              ONE_TIMES_B32_NUM,
                                                                              predicateDefault);
    }
    if (tileId < (totalTileCount - 1)) {
        // store block hist with flag value to gm
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        // copy ub to gm
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_BIN_NUM * sizeof(uint32_t);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(allblockHistToGm[unsortedAxisIdOffset + RADIX_SORT_BIN_NUM * tileId],
                                     blockHistWithFlag, dataCopyParam);
        event_t eventIdMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3);
    }
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::AddPrevfixMask(
    LocalTensor<uint32_t> blockHistWithFlag,
    GlobalTensor<uint32_t> blockHistToGm,
    int32_t tileId)
{
    uint32_t unsortedAxisId = GetBlockIdx() / lastDimRealCore_;
    uint32_t unsortedAxisIdOffset = unsortedAxisId * lastDimTileNum_ * RADIX_SORT_BIN_NUM;
    uint16_t repateTime = RADIX_SORT_BIN_NUM / ONE_TIMES_B32_NUM;
    __local_mem__ uint32_t* histCumsumPtr = (__ubuf__ uint32_t*)blockHistWithFlag.GetPhyAddr();
    __local_mem__ uint32_t* histCumsumPtrCopy = histCumsumPtr;
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint32_t> prefixReadyMask, prefixRemainMask;
        MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::Duplicate(prefixReadyMask, PREFIX_READY_MASK, predicateDefault);
        MicroAPI::Duplicate(prefixRemainMask, VALUE_MASK, predicateDefault);
        for (uint16_t repate = 0; repate < repateTime; repate++) {
            MicroAPI::RegTensor<uint32_t> keyCumsumValue;
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue,
                                                                                  histCumsumPtr,
                                                                                  ONE_TIMES_B32_NUM);
            // remove mask and get value
            // 清除状态位
            MicroAPI::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
            // add new mask
            // 添加OK状态位
            MicroAPI::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumPtrCopy,
                                                                                  keyCumsumValue,
                                                                                  ONE_TIMES_B32_NUM,
                                                                                  predicateDefault);
        }
    }
    // copy value to gm
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = RADIX_SORT_BIN_NUM * sizeof(uint32_t);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(blockHistToGm[unsortedAxisIdOffset + RADIX_SORT_BIN_NUM * tileId], blockHistWithFlag, dataCopyParam);
    event_t eventIdMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3);
}

template <typename T, typename CONVERT_TYPE, typename UNSINGED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortSimdEntry<T, CONVERT_TYPE, UNSINGED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX, T_INDEX_TO>::LookbackGlobal(
    LocalTensor<uint32_t> nowTileHistBuffer,
    GlobalTensor<uint32_t> allTileHistBuffer,
    LocalTensor<uint32_t> ubFlagTensor,
    int32_t tileId,
    uint32_t numTileData)
{
    uint32_t unsortedAxisId = GetBlockIdx() / lastDimRealCore_;
    uint32_t unsortedAxisIdOffset = unsortedAxisId * lastDimTileNum_ * RADIX_SORT_BIN_NUM;
    __local_mem__ uint32_t* nowTileHistBufferPtr = (__ubuf__ uint32_t*)nowTileHistBuffer.GetPhyAddr();
    for(int i = tileId - 1; i >= 0; --i) {
        int mode = -1;
        uint64_t histTileOffset = RADIX_SORT_BIN_NUM * i;
        uint16_t repateTime = RADIX_SORT_BIN_NUM / ONE_TIMES_B32_NUM;
        __local_mem__ uint32_t* ubFlagTensorPtr = (__ubuf__ uint32_t*)ubFlagTensor.GetPhyAddr();
        __local_mem__ uint32_t* ubFlagTensorPtrCopy = ubFlagTensorPtr;
        __local_mem__ uint32_t* tilePrevHistValuePtrCopy = nullptr;
        // 软同步，等待，读flag
        while(true) {
            ubFlagTensorPtrCopy = ubFlagTensorPtr;
            CopyGlobalDataIn(allTileHistBuffer[unsortedAxisIdOffset], histTileOffset, RADIX_SORT_BIN_NUM);
            LocalTensor<uint32_t> tilePrevHistValue = inQueueGlobalHist_.DeQue<uint32_t>();
            __local_mem__ uint32_t* tilePrevHistValuePtr = (__ubuf__ uint32_t*)tilePrevHistValue.GetPhyAddr();
            tilePrevHistValuePtrCopy = tilePrevHistValuePtr;
            // highest 2bit is state bits
            // 0 not inited, should retry
            // 1 aggragate ready accumlate and look i-1 block
            // 2 prefix ready accumlaye and stop
            // 3 undefined
            __VEC_SCOPE__ {
                MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint32_t>();
                // notInit/aggReady/prefixReady flag
                MicroAPI::RegTensor<uint32_t> notInitCount, aggReadyCount, prefixReadyCount;
                MicroAPI::Duplicate(notInitCount, 0, predicateDefault);
                MicroAPI::Duplicate(aggReadyCount, 0, predicateDefault);
                MicroAPI::Duplicate(prefixReadyCount, 0, predicateDefault);
                // flag tensor
                MicroAPI::RegTensor<uint32_t> onesVector, zerosVector;
                MicroAPI::Duplicate(onesVector, 1, predicateDefault);
                MicroAPI::Duplicate(zerosVector, 0, predicateDefault);
                // count notInit/aggReady/prefixReady number
                for (uint16_t i = 0; i < repateTime; i++) {
                    MicroAPI::RegTensor<uint32_t> prevTileHistValue;
                    // load pre tile hist
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(prevTileHistValue,
                                                                                          tilePrevHistValuePtr,
                                                                                          ONE_TIMES_B32_NUM);
                    // get highest 2bit
                    MicroAPI::RegTensor<uint32_t> stateBitValue;
                    MicroAPI::ShiftRights<uint32_t, int16_t>(stateBitValue, prevTileHistValue,
                                                             STATE_BIT_SHF_VALUE, predicateDefault);
                    // get not init cnt
                    MicroAPI::MaskReg maskNotInit;
                    MicroAPI::RegTensor<uint32_t> maskNotInitCount;
                    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskNotInit, stateBitValue,
                                                                   NOT_INIT_MODE, predicateDefault);
                    MicroAPI::Select(maskNotInitCount, onesVector, zerosVector, maskNotInit);
                    MicroAPI::Add(notInitCount, notInitCount, maskNotInitCount, maskNotInit);
                    // get aggrefate ready cnt
                    MicroAPI::MaskReg maskAggReady;
                    MicroAPI::RegTensor<uint32_t> maskAggReadyCount;
                    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskAggReady, stateBitValue,
                                                                   AGG_READY_MODE, predicateDefault);
                    MicroAPI::Select(maskAggReadyCount, onesVector, zerosVector, maskAggReady);
                    MicroAPI::Add(aggReadyCount, aggReadyCount, maskAggReadyCount, maskAggReady);
                    // get prefix ready cnt
                    MicroAPI::MaskReg maskPrefixReady;
                    MicroAPI::RegTensor<uint32_t> maskPrefixReadyCount;
                    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskPrefixReady, stateBitValue,
                                                                   PREFIX_READY_MODE, predicateDefault);
                    MicroAPI::Select(maskPrefixReadyCount, onesVector, zerosVector, maskPrefixReady);
                    MicroAPI::Add(prefixReadyCount, prefixReadyCount, maskPrefixReadyCount, maskPrefixReady);
                }
                // reduce sum noInitCount
                MicroAPI::ReduceSum(notInitCount, notInitCount, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                  MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr,
                                                                               notInitCount,
                                                                               HIST_MASK_OUT_LEN,
                                                                               predicateDefault);
                // reduce sum aggReadyCount
                MicroAPI::ReduceSum(aggReadyCount, aggReadyCount, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr,
                                                                                aggReadyCount,
                                                                                HIST_MASK_OUT_LEN,
                                                                                predicateDefault);
                // reduce sum prefixReadyCount
                MicroAPI::ReduceSum(prefixReadyCount,prefixReadyCount, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr,
                                                                                prefixReadyCount,
                                                                                HIST_MASK_OUT_LEN,
                                                                                predicateDefault);
            }
            // get (not init)/(aggregate ready)/(preifx ready) cnt
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
            uint32_t notInitCountScalar = ubFlagTensorPtr[NOT_INIT_COUNT_INDEX];
            uint32_t aggReadyCountScalar = ubFlagTensorPtr[AGG_READY_COUNT_INDEX];
            uint32_t PrefixReadyCountScalar = ubFlagTensorPtr[PREFIX_READY_COUNT_INDEX];
            if (aggReadyCountScalar == RADIX_SORT_BIN_NUM) {
                mode = AGGREGATE_READY_FLAG;
                inQueueGlobalHist_.FreeTensor( tilePrevHistValue);
                break;
            }
            if (PrefixReadyCountScalar == RADIX_SORT_BIN_NUM) {
                mode = PREFIX_READY_FLAG;
                inQueueGlobalHist_.FreeTensor( tilePrevHistValue);
                break;
            }
            inQueueGlobalHist_.FreeTensor( tilePrevHistValue);
        }
        __local_mem__ uint32_t* nowTileHistBufferPtrCopy = nowTileHistBufferPtr;
        event_t eventIdWaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdWaitV);
        WaitFlag<HardEvent::S_V>(eventIdWaitV);
        __VEC_SCOPE__ {
            MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint16_t>();
            MicroAPI::RegTensor<uint32_t> histMask;
            MicroAPI::Duplicate(histMask, VALUE_MASK, predicateDefault);
            for (uint16_t i = 0; i < repateTime; i++) {
                MicroAPI::RegTensor<uint32_t> nowTileHistVal, prevTileHistVal;
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(nowTileHistVal,
                                                                                      nowTileHistBufferPtr,
                                                                                      ONE_TIMES_B32_NUM);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(prevTileHistVal,
                                                                                      tilePrevHistValuePtrCopy,
                                                                                      ONE_TIMES_B32_NUM);
                // remove mask and get value
                MicroAPI::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
                MicroAPI::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
                // add prev hist value
                // get value key same which tile id less and equal now
                MicroAPI::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(nowTileHistBufferPtrCopy,
                                                                                      nowTileHistVal,
                                                                                      ONE_TIMES_B32_NUM,
                                                                                      predicateDefault);
            }
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        if (mode == PREFIX_READY_FLAG) {
            break;
        }
    }
}
#endif