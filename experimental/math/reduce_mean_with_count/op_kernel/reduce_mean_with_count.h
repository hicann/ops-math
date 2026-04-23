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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file reduce_mean_with_count.h
 * @brief ReduceMeanWithCount kernel class definition (arch35)
 *
 * Template parameters:
 *   - T: data type (float, half, bfloat16_t)
 *   - REDUCE_MODE: 0=AR full-load, 1=AR col-split, 2=ARA full-load
 *
 * Iteration 3: Full coverage - FP32 + FP16 + BF16 across all three modes (TK0..TK8).
 * FP16/BF16 use FP32 intermediate accumulation to avoid overflow:
 *   FP16 path: Cast<FP16->FP32, CAST_NONE> ... compute ... Cast<FP32->FP16, CAST_NONE>
 *   BF16 path: Cast<BF16->FP32, CAST_NONE> ... compute ... Cast<FP32->BF16, CAST_RINT>
 */
#ifndef REDUCE_MEAN_WITH_COUNT_H
#define REDUCE_MEAN_WITH_COUNT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_mean_with_count_tiling_data.h"
#include "reduce_mean_with_count_tiling_key.h"

namespace NsReduceMeanWithCount {

using namespace AscendC;

// -----------------------------------------------------------------------------
// Helper: Cast RoundMode for FP32 -> Low precision
// FP32 -> FP16: CAST_NONE (rounding-to-nearest-even by default)
// FP32 -> BF16: CAST_RINT (required on Ascend950)
// -----------------------------------------------------------------------------
template <typename T>
__aicore__ inline RoundMode GetCastDownRoundMode()
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return RoundMode::CAST_RINT;
    }
    return RoundMode::CAST_NONE;
}

template <typename T, int REDUCE_MODE>
class ReduceMeanWithCount {
    // Double buffer for AR full-load and ARA full-load; single buffer for AR col-split
    static constexpr int32_t BUFFER_NUM =
        (REDUCE_MODE == REDUCE_MODE_AR_COLSPLIT) ? 1 : 2;

    // Whether T requires precision-promotion for ReduceSum (FP16/BF16 -> FP32)
    static constexpr bool NEED_CAST = !std::is_same_v<T, float>;

public:
    __aicore__ inline ReduceMeanWithCount() {};

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR meanResult, GM_ADDR countResult,
                                 const ReduceMeanWithCountTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // AR full-load processing
    __aicore__ inline void ProcessARFullLoad();
    // AR col-split processing
    __aicore__ inline void ProcessARColSplit();
    // ARA full-load processing
    __aicore__ inline void ProcessARAFullLoad();

    // count_result fill
    __aicore__ inline void FillCountResult();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> meanResultGM_;
    GlobalTensor<int64_t> countResultGM_;

    // Tiling parameters
    uint64_t a1Length_;
    uint64_t rLength_;
    uint64_t a0Length_;
    uint64_t tilesPerCore_;
    uint64_t tailCoreTiles_;
    float invCount_;
    int64_t countResult_;
    uint64_t tmpBufSize_;
    uint64_t outputLength_;
    int32_t usedCoreNum_;

    // AR col-split specific
    uint64_t chunkR_;

    // ARA full-load specific
    uint64_t tileA0Len_;
    uint64_t alignedCols_;     // CeilAlign(tileA0Len * sizeof(T), 32) / sizeof(T) -- "source-typed" cols
    uint64_t alignedColsFP32_; // CeilAlign(tileA0Len * sizeof(float), 32) / sizeof(float)
    uint64_t a0Outer_;         // CeilDiv(a0Length, tileA0Len)

    // Derived parameters (common)
    uint64_t myStartTile_;
    uint64_t myTileCount_;
    uint64_t myStartRow_;      // alias to myStartTile_
    uint64_t myRowCount_;      // alias to myTileCount_
    uint64_t rLengthAlign_;    // rLength aligned to element boundary of T

    // Temporary buffer for ReduceSum
    TBuf<QuePosition::VECCALC> tmpBuf_;
    // FP32 cast buffer (only used for FP16/BF16)
    TBuf<QuePosition::VECCALC> castBuf_;
    // FP32 result buffer (only used for FP16/BF16 - holds ReduceSum/Mul result before cast back)
    TBuf<QuePosition::VECCALC> fp32ResultBuf_;
    // Count buffer for ARA mode (separate from inQueue_ to avoid conflicts)
    TBuf<QuePosition::VECCALC> countBuf_;
};

// ============================================================================
// Init
// ============================================================================
template <typename T, int REDUCE_MODE>
__aicore__ inline void ReduceMeanWithCount<T, REDUCE_MODE>::Init(
    GM_ADDR input, GM_ADDR meanResult, GM_ADDR countResult,
    const ReduceMeanWithCountTilingData* tilingData)
{
    a1Length_ = tilingData->a1Length;
    rLength_ = tilingData->rLength;
    a0Length_ = tilingData->a0Length;
    usedCoreNum_ = tilingData->usedCoreNum;
    tilesPerCore_ = tilingData->tilesPerCore;
    tailCoreTiles_ = tilingData->tailCoreTiles;
    invCount_ = tilingData->invCount;
    countResult_ = tilingData->countResult;
    tmpBufSize_ = tilingData->tmpBufSize;
    outputLength_ = tilingData->outputLength;
    chunkR_ = tilingData->chunkR;
    tileA0Len_ = tilingData->tileA0Len;

    // Align rLength to element boundary (8 elements for FP32, 16 elements for FP16/BF16)
    constexpr uint32_t elemPerBlock = 32 / sizeof(T);
    rLengthAlign_ = ((rLength_ + elemPerBlock - 1) / elemPerBlock) * elemPerBlock;

    // FP32 alignment of rLength (used for FP16/BF16 cast/reducesum buffers)
    constexpr uint32_t elemPerBlockFP32 = 32 / sizeof(float);  // 8
    uint64_t rLengthAlignFP32 = ((rLength_ + elemPerBlockFP32 - 1) / elemPerBlockFP32) * elemPerBlockFP32;

    uint32_t blockIdx = GetBlockIdx();
    if (static_cast<int32_t>(blockIdx) >= usedCoreNum_) {
        myTileCount_ = 0;
        myRowCount_ = 0;
        return;
    }

    bool isTailCore = (static_cast<int32_t>(blockIdx) == usedCoreNum_ - 1);
    myTileCount_ = isTailCore ? tailCoreTiles_ : tilesPerCore_;
    myStartTile_ = blockIdx * tilesPerCore_;
    // For AR modes, tiles = rows
    myRowCount_ = myTileCount_;
    myStartRow_ = myStartTile_;

    if constexpr (REDUCE_MODE == REDUCE_MODE_AR_FULLLOAD) {
        // Set GM tensors
        inputGM_.SetGlobalBuffer((__gm__ T*)input + myStartRow_ * rLength_, myRowCount_ * rLength_);
        meanResultGM_.SetGlobalBuffer((__gm__ T*)meanResult + myStartRow_, myRowCount_);
        countResultGM_.SetGlobalBuffer((__gm__ int64_t*)countResult + myStartRow_, myRowCount_);

        // Init buffers: double buffer for input
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, rLengthAlign_ * sizeof(T));
        // Output queue: type T, holds 1 element (>= 32 bytes)
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, 32);
        pipe_.InitBuffer(tmpBuf_, tmpBufSize_);

        if constexpr (NEED_CAST) {
            // Cast buffer: rLengthAlignFP32 * sizeof(float) - holds Cast<T->FP32>(input) result
            pipe_.InitBuffer(castBuf_, rLengthAlignFP32 * sizeof(float));
            // FP32 result buffer for ReduceSum + Muls (1 scalar but >= 32B)
            pipe_.InitBuffer(fp32ResultBuf_, 32);
        }
    } else if constexpr (REDUCE_MODE == REDUCE_MODE_AR_COLSPLIT) {
        // Set GM tensors (full input visible for strided access)
        inputGM_.SetGlobalBuffer((__gm__ T*)input, a1Length_ * rLength_);
        meanResultGM_.SetGlobalBuffer((__gm__ T*)meanResult, outputLength_);
        countResultGM_.SetGlobalBuffer((__gm__ int64_t*)countResult, outputLength_);

        // chunkR aligned to elemPerBlock
        uint64_t chunkRAlign = ((chunkR_ + elemPerBlock - 1) / elemPerBlock) * elemPerBlock;
        // FP32 aligned chunkR (for cast buffer)
        uint64_t chunkRAlignFP32 = ((chunkR_ + elemPerBlockFP32 - 1) / elemPerBlockFP32) * elemPerBlockFP32;

        // Single buffer mode (chunk data dependency prevents pipelining)
        pipe_.InitBuffer(inQueue_, 1, chunkRAlign * sizeof(T));
        pipe_.InitBuffer(outQueue_, 1, 32);
        pipe_.InitBuffer(tmpBuf_, tmpBufSize_);

        if constexpr (NEED_CAST) {
            pipe_.InitBuffer(castBuf_, chunkRAlignFP32 * sizeof(float));
            pipe_.InitBuffer(fp32ResultBuf_, 32);
        }
    } else if constexpr (REDUCE_MODE == REDUCE_MODE_ARA_FULLLOAD) {
        // Compute alignedCols for UB storage in source type T
        uint64_t tileA0Bytes = tileA0Len_ * sizeof(T);
        uint64_t alignedBytes = ((tileA0Bytes + 31) / 32) * 32;
        alignedCols_ = alignedBytes / sizeof(T);

        // FP32-aligned columns (for FP32 result buffer width)
        uint64_t tileA0BytesFP32 = tileA0Len_ * sizeof(float);
        uint64_t alignedBytesFP32 = ((tileA0BytesFP32 + 31) / 32) * 32;
        alignedColsFP32_ = alignedBytesFP32 / sizeof(float);

        a0Outer_ = ((a0Length_ + tileA0Len_ - 1) / tileA0Len_);

        // Set GM tensors (full input visible for strided access)
        inputGM_.SetGlobalBuffer((__gm__ T*)input, a1Length_ * rLength_ * a0Length_);
        meanResultGM_.SetGlobalBuffer((__gm__ T*)meanResult, outputLength_);
        countResultGM_.SetGlobalBuffer((__gm__ int64_t*)countResult, outputLength_);

        // Double buffer for input (source type T, R rows of alignedCols_)
        uint64_t inBufElemSize = rLength_ * alignedCols_;
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, inBufElemSize * sizeof(T));

        // Output queue (source type T, alignedCols_ wide)
        uint64_t outBufBytes = ((alignedCols_ * sizeof(T) + 31) / 32) * 32;
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, outBufBytes);

        pipe_.InitBuffer(tmpBuf_, tmpBufSize_);

        if constexpr (NEED_CAST) {
            // Cast buffer: holds R * alignedCols_ FP32 elements
            // NOTE: keep alignedCols_ (source-typed alignment) so element-wise Cast is layout-preserving.
            uint64_t castBufElems = rLength_ * alignedCols_;
            pipe_.InitBuffer(castBuf_, castBufElems * sizeof(float));
            // FP32 result buffer (alignedCols_ FP32 elements, >= 32B)
            uint64_t fp32ResBytes = ((alignedCols_ * sizeof(float) + 31) / 32) * 32;
            pipe_.InitBuffer(fp32ResultBuf_, fp32ResBytes);
        }

        // Count buffer for FillCountResult
        uint64_t minCountElems = (tileA0Len_ < 4) ? 4 : tileA0Len_;
        uint64_t countBufBytes = ((minCountElems * sizeof(int64_t) + 31) / 32) * 32;
        pipe_.InitBuffer(countBuf_, countBufBytes);
    }
}

// ============================================================================
// Process
// ============================================================================
template <typename T, int REDUCE_MODE>
__aicore__ inline void ReduceMeanWithCount<T, REDUCE_MODE>::Process()
{
    if (myTileCount_ == 0) {
        return;
    }

    if constexpr (REDUCE_MODE == REDUCE_MODE_AR_FULLLOAD) {
        ProcessARFullLoad();
    } else if constexpr (REDUCE_MODE == REDUCE_MODE_AR_COLSPLIT) {
        ProcessARColSplit();
    } else if constexpr (REDUCE_MODE == REDUCE_MODE_ARA_FULLLOAD) {
        ProcessARAFullLoad();
    }

    // Fill count_result
    FillCountResult();
}

// ============================================================================
// AR Full-Load Implementation (TK0 / TK3 / TK6)
// FP32: direct ReduceSum
// FP16/BF16: Cast(T->FP32) -> ReduceSum -> Muls -> Cast(FP32->T) -> CopyOut
// ============================================================================
template <typename T, int REDUCE_MODE>
__aicore__ inline void ReduceMeanWithCount<T, REDUCE_MODE>::ProcessARFullLoad()
{
    for (uint64_t row = 0; row < myRowCount_; row++) {
        // ---- CopyIn: GM[row, 0:R] -> UB inBuf (type T) ----
        LocalTensor<T> inLocal = inQueue_.template AllocTensor<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = static_cast<uint32_t>(rLength_ * sizeof(T));
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
        DataCopyPad(inLocal, inputGM_[row * rLength_], copyInParams, padParams);
        inQueue_.EnQue(inLocal);

        LocalTensor<T> inCompute = inQueue_.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();

        if constexpr (!NEED_CAST) {
            // ---- FP32 path: direct ReduceSum on T (= float) ----
            LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
            ReduceSum<float>(outLocal, inCompute, tmpLocal, static_cast<int32_t>(rLength_));
            float sumVal = outLocal.GetValue(0);
            float meanVal = sumVal * invCount_;
            outLocal.SetValue(0, meanVal);
        } else {
            // ---- FP16/BF16 path: Cast -> ReduceSum -> Muls -> Cast back ----
            LocalTensor<float> castLocal = castBuf_.Get<float>();
            // Cast<float, T> with CAST_NONE works for both FP16 and BF16 -> FP32
            Cast<float, T>(castLocal, inCompute, RoundMode::CAST_NONE, static_cast<int32_t>(rLength_));
            PipeBarrier<PIPE_V>();

            LocalTensor<float> sumLocal = fp32ResultBuf_.Get<float>();
            LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
            ReduceSum<float>(sumLocal, castLocal, tmpLocal, static_cast<int32_t>(rLength_));
            PipeBarrier<PIPE_V>();

            float sumVal = sumLocal.GetValue(0);
            // Guard against NaN/Inf propagation: arithmetic on Inf/NaN remains Inf/NaN, that's OK.
            float meanVal = sumVal * invCount_;
            sumLocal.SetValue(0, meanVal);
            PipeBarrier<PIPE_V>();

            // Cast back FP32 -> T (FP16: CAST_NONE; BF16: CAST_RINT)
            Cast<T, float>(outLocal, sumLocal, GetCastDownRoundMode<T>(), 1);
            PipeBarrier<PIPE_V>();
        }

        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inCompute);

        // ---- CopyOut: outLocal (1 element of T) -> GM meanResult[row] ----
        LocalTensor<T> outCopy = outQueue_.template DeQue<T>();
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(sizeof(T));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(meanResultGM_[row], outCopy, copyOutParams);
        outQueue_.FreeTensor(outCopy);
    }
}

// ============================================================================
// AR Col-Split Implementation (TK1 / TK4 / TK7)
// Single buffer mode: split row into chunks, cross-chunk scalar accumulation in FP32.
// ============================================================================
template <typename T, int REDUCE_MODE>
__aicore__ inline void ReduceMeanWithCount<T, REDUCE_MODE>::ProcessARColSplit()
{
    uint64_t numChunks = (rLength_ + chunkR_ - 1) / chunkR_;

    for (uint64_t row = 0; row < myRowCount_; row++) {
        uint64_t globalRow = myStartRow_ + row;
        float globalSum = 0.0f;  // FP32 accumulation across chunks (overflow-safe)

        for (uint64_t c = 0; c < numChunks; c++) {
            uint64_t colStart = c * chunkR_;
            uint64_t chunkSize = chunkR_;
            if (colStart + chunkSize > rLength_) {
                chunkSize = rLength_ - colStart;
            }

            // ---- CopyIn chunk: GM[globalRow, colStart:colStart+chunkSize] -> UB (type T) ----
            LocalTensor<T> inLocal = inQueue_.template AllocTensor<T>();
            uint64_t gmOffset = globalRow * rLength_ + colStart;
            DataCopyExtParams colCopyParams;
            colCopyParams.blockCount = 1;
            colCopyParams.blockLen = static_cast<uint32_t>(chunkSize * sizeof(T));
            colCopyParams.srcStride = 0;
            colCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> colPadParams{true, 0, 0, (T)0};
            DataCopyPad(inLocal, inputGM_[gmOffset], colCopyParams, colPadParams);
            inQueue_.EnQue(inLocal);
            LocalTensor<T> inCompute = inQueue_.template DeQue<T>();

            // ---- ReduceSum (chunk -> FP32 scalar) ----
            LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();

            if constexpr (!NEED_CAST) {
                // FP32 path: ReduceSum directly into out queue
                LocalTensor<T> sumLocal = outQueue_.template AllocTensor<T>();
                ReduceSum<float>(sumLocal, inCompute, tmpLocal, static_cast<int32_t>(chunkSize));
                PipeBarrier<PIPE_V>();
                globalSum += sumLocal.GetValue(0);
                outQueue_.FreeTensor(sumLocal);
            } else {
                // FP16/BF16 path: Cast then ReduceSum<FP32>
                LocalTensor<float> castLocal = castBuf_.Get<float>();
                Cast<float, T>(castLocal, inCompute, RoundMode::CAST_NONE, static_cast<int32_t>(chunkSize));
                PipeBarrier<PIPE_V>();

                LocalTensor<float> sumLocal = fp32ResultBuf_.Get<float>();
                ReduceSum<float>(sumLocal, castLocal, tmpLocal, static_cast<int32_t>(chunkSize));
                PipeBarrier<PIPE_V>();
                globalSum += sumLocal.GetValue(0);
            }

            inQueue_.FreeTensor(inCompute);
        }

        // ---- mean = globalSum * invCount, write to GM ----
        float meanVal = globalSum * invCount_;

        LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();
        if constexpr (!NEED_CAST) {
            outLocal.SetValue(0, meanVal);
        } else {
            // Use fp32ResultBuf_ as a temp scalar holder, then Cast to T into outLocal
            LocalTensor<float> tmpScalar = fp32ResultBuf_.Get<float>();
            tmpScalar.SetValue(0, meanVal);
            PipeBarrier<PIPE_V>();
            Cast<T, float>(outLocal, tmpScalar, GetCastDownRoundMode<T>(), 1);
            PipeBarrier<PIPE_V>();
        }
        outQueue_.EnQue(outLocal);

        LocalTensor<T> outCopy = outQueue_.template DeQue<T>();
        DataCopyExtParams colOutParams;
        colOutParams.blockCount = 1;
        colOutParams.blockLen = static_cast<uint32_t>(sizeof(T));
        colOutParams.srcStride = 0;
        colOutParams.dstStride = 0;
        DataCopyPad(meanResultGM_[myStartRow_ + row], outCopy, colOutParams);
        outQueue_.FreeTensor(outCopy);
    }
}

// ============================================================================
// ARA Full-Load Implementation (TK2 / TK5 / TK8)
// Non-innermost axis reduction using ReduceSum<float, Pattern::Reduce::RA>.
// FP16/BF16: layout after Cast preserves alignedCols_ (source-typed) columns;
//            srcShape uses alignedCols_ (NOT alignedColsFP32_) for ReduceSum.
// ============================================================================
template <typename T, int REDUCE_MODE>
__aicore__ inline void ReduceMeanWithCount<T, REDUCE_MODE>::ProcessARAFullLoad()
{
    for (uint64_t t = 0; t < myTileCount_; t++) {
        uint64_t tileIdx = myStartTile_ + t;
        uint64_t a1Idx = tileIdx / a0Outer_;
        uint64_t a0TileIdx = tileIdx % a0Outer_;
        uint64_t a0Start = a0TileIdx * tileA0Len_;
        uint64_t actualA0 = (a0Start + tileA0Len_ <= a0Length_) ? tileA0Len_ : (a0Length_ - a0Start);

        // ---- CopyIn: GM[a1Idx, 0:R, a0Start:a0Start+actualA0] -> UB inBuf (type T, R rows of alignedCols_) ----
        LocalTensor<T> inLocal = inQueue_.template AllocTensor<T>();
        uint64_t gmOffset = a1Idx * rLength_ * a0Length_ + a0Start;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = static_cast<uint16_t>(rLength_);
        copyInParams.blockLen = static_cast<uint32_t>(actualA0 * sizeof(T));
        copyInParams.srcStride = static_cast<int64_t>((a0Length_ - actualA0) * sizeof(T));
        copyInParams.dstStride = 0;

        DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
        DataCopyPad(inLocal, inputGM_[gmOffset], copyInParams, padParams);

        inQueue_.EnQue(inLocal);

        LocalTensor<T> inData = inQueue_.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();
        LocalTensor<uint8_t> tmpLocal = tmpBuf_.Get<uint8_t>();

        if constexpr (!NEED_CAST) {
            // ---- FP32 path ----
            // srcShape = {R, alignedCols}
            uint32_t srcShape[2] = {static_cast<uint32_t>(rLength_), static_cast<uint32_t>(alignedCols_)};
            ReduceSum<float, Pattern::Reduce::RA>(outLocal, inData, tmpLocal, srcShape, true);

            Muls(outLocal, outLocal, invCount_, static_cast<uint32_t>(alignedCols_));
        } else {
            // ---- FP16/BF16 path ----
            // 1) Cast all R*alignedCols_ low-precision elements to FP32, layout preserved.
            LocalTensor<float> castLocal = castBuf_.Get<float>();
            uint32_t totalCastElems = static_cast<uint32_t>(rLength_ * alignedCols_);
            Cast<float, T>(castLocal, inData, RoundMode::CAST_NONE, totalCastElems);
            PipeBarrier<PIPE_V>();

            // 2) ReduceSum<float, RA> with srcShape = {R, alignedCols_} (source-typed cols!)
            LocalTensor<float> fp32Result = fp32ResultBuf_.Get<float>();
            uint32_t srcShape[2] = {static_cast<uint32_t>(rLength_), static_cast<uint32_t>(alignedCols_)};
            ReduceSum<float, Pattern::Reduce::RA>(fp32Result, castLocal, tmpLocal, srcShape, true);
            PipeBarrier<PIPE_V>();

            // 3) Muls in FP32
            Muls(fp32Result, fp32Result, invCount_, static_cast<uint32_t>(alignedCols_));
            PipeBarrier<PIPE_V>();

            // 4) Cast back FP32 -> T (FP16: CAST_NONE; BF16: CAST_RINT)
            Cast<T, float>(outLocal, fp32Result, GetCastDownRoundMode<T>(), static_cast<uint32_t>(alignedCols_));
            PipeBarrier<PIPE_V>();
        }

        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inData);

        // ---- CopyOut: outLocal[0:actualA0] -> GM meanResult[a1Idx * A0 + a0Start] ----
        LocalTensor<T> result = outQueue_.template DeQue<T>();
        uint64_t meanOffset = a1Idx * a0Length_ + a0Start;
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(actualA0 * sizeof(T));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(meanResultGM_[meanOffset], result, copyOutParams);

        outQueue_.FreeTensor(result);
    }
}

// ============================================================================
// Fill count_result
// ============================================================================
template <typename T, int REDUCE_MODE>
__aicore__ inline void ReduceMeanWithCount<T, REDUCE_MODE>::FillCountResult()
{
    if constexpr (REDUCE_MODE == REDUCE_MODE_ARA_FULLLOAD) {
        // ARA mode: fill count_result for all output elements from core 0
        if (GetBlockIdx() != 0) {
            return;
        }

        // Ensure all mean computations are flushed
        PipeBarrier<PIPE_ALL>();

        LocalTensor<int64_t> countLocal = countBuf_.Get<int64_t>();
        uint32_t dupCount = (tileA0Len_ < 4) ? 4 : static_cast<uint32_t>(tileA0Len_);
        Duplicate<int64_t>(countLocal, countResult_, dupCount);
        PipeBarrier<PIPE_ALL>();

        // Write count_result for each output element
        uint64_t remaining = outputLength_;
        uint64_t offset = 0;
        while (remaining > 0) {
            uint64_t chunk = (remaining > tileA0Len_) ? tileA0Len_ : remaining;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(chunk * sizeof(int64_t));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(countResultGM_[offset], countLocal, copyParams);
            PipeBarrier<PIPE_ALL>();
            offset += chunk;
            remaining -= chunk;
        }
    } else {
        // AR modes: each core fills its portion of count_result
        constexpr uint32_t elemPerBlock = 32 / sizeof(int64_t); // 4 int64 per 32 bytes
        uint32_t alignedCount = ((myRowCount_ + elemPerBlock - 1) / elemPerBlock) * elemPerBlock;

        // Allocate from inQueue_ which is free after mean computation.
        // Note: inQueue_ template type is T; reinterpret as int64_t below.
        LocalTensor<int64_t> countLocal = inQueue_.template AllocTensor<int64_t>();

        // Fill with countResult_ value using Duplicate
        Duplicate<int64_t>(countLocal, countResult_, alignedCount);

        // Synchronize before DataCopyPad
        PipeBarrier<PIPE_ALL>();

        // For AR full-load, GM is already offset by myStartRow_ in SetGlobalBuffer, write at 0.
        // For AR col-split, GM starts from 0, write at myStartRow_.
        uint64_t countGMOffset = 0;
        if constexpr (REDUCE_MODE == REDUCE_MODE_AR_COLSPLIT) {
            countGMOffset = myStartRow_;
        }

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(myRowCount_ * sizeof(int64_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(countResultGM_[countGMOffset], countLocal, copyParams);

        PipeBarrier<PIPE_ALL>();

        inQueue_.FreeTensor(countLocal);
    }
}

} // namespace NsReduceMeanWithCount

#endif // REDUCE_MEAN_WITH_COUNT_H
