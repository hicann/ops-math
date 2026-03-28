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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file reduce_nansum_ara_rowsplit.h
 * \brief ReduceNansum ARA RowSplit Kernel 实现（TilingKey=3）
 *
 * ARA 模板（A0>1）分载模式：当 R > R_max 时，
 * 将 R 方向分 chunk 处理，每个 chunk 做 ReduceSum(Pattern::RA)
 * 归约为 1 行，然后跨 chunk 逐元素 Add 累加。
 *
 * 迭代三：支持 fp32/fp16/bf16 混合精度。
 * AtomicAdd 多核优化：当 totalTiles=1 时，按 R 维度切分多核并行，使用 AtomicAdd 汇聚结果。
 */
#ifndef REDUCE_NANSUM_ARA_ROWSPLIT_H
#define REDUCE_NANSUM_ARA_ROWSPLIT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_nansum_tiling_data.h"
#include "reduce_nansum_tiling_key.h"

namespace NsReduceNansum {

using namespace AscendC;

template <typename T>
class ReduceNansumAraRowsplit {
public:
    __aicore__ inline ReduceNansumAraRowsplit() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReduceNansumTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneTile(int64_t a1Idx, int64_t a0OuterIdx);
    __aicore__ inline void ProcessOneTileAtomicAdd(int64_t a1Idx, int64_t a0OuterIdx);

private:
    static constexpr bool IS_FP32 = sizeof(T) == sizeof(float);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> cleanBuf;
    TBuf<QuePosition::VECCALC> globalResultBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> castBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t a1Count_ = 0;
    int64_t rCount_ = 0;
    int64_t a0Count_ = 0;
    int64_t tileA0Len_ = 0;
    int64_t alignedCols_ = 0;
    int64_t a0Outer_ = 0;
    int64_t originalA0_ = 0;
    int64_t tmpBufSize_ = 0;
    int64_t rChunkSize_ = 0;
    int64_t numChunks_ = 0;
    int64_t lastChunkSize_ = 0;

    // 多核参数
    int64_t startTile_ = 0;
    int64_t tileCount_ = 0;

    // AtomicAdd 多核归约参数
    int64_t useAtomicAdd_ = 0;
    int64_t coreRStart_ = 0;      // 本核处理的 R 起始位置
    int64_t coreRCount_ = 0;      // 本核处理的 R 数量
    int64_t coreNumChunks_ = 0;   // 本核的 chunk 数
    int64_t coreLastChunkSize_ = 0; // 本核最后一个 chunk 大小

    // 非连续多轴归约参数（ARA strided）
    int64_t copyBlockCount_ = 0;
    int64_t copyBlockLen_ = 0;
    int64_t copySrcStride_ = 0;
    int64_t nonReduceDimCount_ = 0;
    int64_t nonReduceDimSizes_[8] = {0};
    int64_t nonReduceGmStrides_[8] = {0};
    // 归约维度信息（用于3+非连续轴的逐块GM偏移计算）
    int64_t reduceDimCount_ = 0;
    int64_t reduceDimSizes_[8] = {0};
    int64_t reduceGmStrides_[8] = {0};
    int64_t innerRowsPerBlock_ = 0;
    int64_t outerReduceDimCount_ = 0;
};

template <typename T>
__aicore__ inline void ReduceNansumAraRowsplit<T>::Init(GM_ADDR x, GM_ADDR y,
                                                         const ReduceNansumTilingData* tilingData)
{
    a1Count_ = tilingData->a1Count;
    rCount_ = tilingData->rCount;
    a0Count_ = tilingData->a0Count;
    tileA0Len_ = tilingData->tileA0Len;
    alignedCols_ = tilingData->alignedCols;
    a0Outer_ = tilingData->a0Outer;
    originalA0_ = tilingData->originalA0;
    tmpBufSize_ = tilingData->tmpBufSize;
    rChunkSize_ = tilingData->rChunkSize;
    numChunks_ = tilingData->numChunks;
    lastChunkSize_ = tilingData->lastChunkSize;

    // 多核切分
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t tilesPerCore = tilingData->tilesPerCore;
    int64_t tailCoreTiles = tilingData->tailCoreTiles;
    int64_t usedCoreNum = tilingData->usedCoreNum;

    // AtomicAdd 参数
    useAtomicAdd_ = tilingData->useAtomicAdd;

    if (useAtomicAdd_) {
        // AtomicAdd 模式：每核处理同一 tile（totalTiles=1），但处理不同的 R 区间
        startTile_ = 0;
        tileCount_ = 1;

        int64_t rPerCore = tilingData->rPerCore;
        coreRStart_ = blockIdx * rPerCore;
        coreRCount_ = rPerCore;
        if (coreRStart_ + coreRCount_ > rCount_) {
            coreRCount_ = rCount_ - coreRStart_;
        }
        if (coreRCount_ < 0) coreRCount_ = 0;

        if (coreRCount_ > 0) {
            coreNumChunks_ = (coreRCount_ + rChunkSize_ - 1) / rChunkSize_;
            coreLastChunkSize_ = coreRCount_ - (coreNumChunks_ - 1) * rChunkSize_;
            if (coreLastChunkSize_ <= 0) coreLastChunkSize_ = rChunkSize_;
        } else {
            coreNumChunks_ = 0;
            coreLastChunkSize_ = 0;
        }
    } else {
        startTile_ = blockIdx * tilesPerCore;
        if (blockIdx < usedCoreNum - 1) {
            tileCount_ = tilesPerCore;
        } else {
            tileCount_ = tailCoreTiles;
        }
    }

    // 设置 GM 指针
    copyBlockCount_ = tilingData->copyBlockCount;
    copyBlockLen_ = tilingData->copyBlockLen;
    copySrcStride_ = tilingData->copySrcStride;
    nonReduceDimCount_ = tilingData->nonReduceDimCount;
    for (int64_t i = 0; i < nonReduceDimCount_ && i < 8; i++) {
        nonReduceDimSizes_[i] = tilingData->nonReduceDimSizes[i];
        nonReduceGmStrides_[i] = tilingData->nonReduceGmStrides[i];
    }
    reduceDimCount_ = tilingData->reduceDimCount;
    for (int64_t i = 0; i < reduceDimCount_ && i < 8; i++) {
        reduceDimSizes_[i] = tilingData->reduceDimSizes[i];
        reduceGmStrides_[i] = tilingData->reduceGmStrides[i];
    }
    innerRowsPerBlock_ = 0;
    outerReduceDimCount_ = reduceDimCount_;
    if (copyBlockCount_ > 0 && a0Count_ > 0) {
        int64_t innerBlockElems = copyBlockLen_ / static_cast<int64_t>(sizeof(T));
        innerRowsPerBlock_ = innerBlockElems / a0Count_;
        // 确定外层归约维度数量
        int64_t innerProduct = 1;
        for (int64_t d = reduceDimCount_ - 1; d >= 0; d--) {
            innerProduct *= reduceDimSizes_[d];
            if (innerProduct == innerRowsPerBlock_) {
                outerReduceDimCount_ = d;
                break;
            }
        }
    }

    int64_t inputGmSize = a1Count_ * rCount_ * a0Count_;
    if (copyBlockCount_ > 0) {
        int64_t maxGmOffset = 0;
        for (int64_t d = 0; d < nonReduceDimCount_; d++) {
            maxGmOffset += (nonReduceDimSizes_[d] - 1) * nonReduceGmStrides_[d];
        }
        for (int64_t d = 0; d < reduceDimCount_; d++) {
            maxGmOffset += (reduceDimSizes_[d] - 1) * reduceGmStrides_[d];
        }
        maxGmOffset += a0Count_;
        inputGmSize = maxGmOffset;
    }
    inputGM.SetGlobalBuffer((__gm__ T*)x, inputGmSize);
    outputGM.SetGlobalBuffer((__gm__ T*)y, a1Count_ * a0Count_);

    // 初始化 Buffer（单缓冲，按 chunk 大小分配）
    int64_t inBufSize = rChunkSize_ * alignedCols_ * static_cast<int64_t>(sizeof(T));
    int64_t outBufSize = alignedCols_ * static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(inQueueX, 1, inBufSize);
    pipe.InitBuffer(outQueueY, 1, outBufSize);

    // maskBuf
    int64_t totalMaskElements = rChunkSize_ * alignedCols_;
    int64_t maskBufSize = ((totalMaskElements / 8 + 31) / 32) * 32;
    if (maskBufSize < 32) maskBufSize = 32;
    pipe.InitBuffer(maskBuf, maskBufSize);

    pipe.InitBuffer(zeroBuf, alignedCols_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(cleanBuf, rChunkSize_ * alignedCols_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(globalResultBuf, alignedCols_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(tmpBuf, tmpBufSize_);

    if constexpr (!IS_FP32) {
        pipe.InitBuffer(castBuf, rChunkSize_ * alignedCols_ * static_cast<int64_t>(sizeof(float)));
    }
}

// AtomicAdd 模式：每核处理自己的 R 区间，结果通过 AtomicAdd 写到 GM
template <typename T>
__aicore__ inline void ReduceNansumAraRowsplit<T>::ProcessOneTileAtomicAdd(int64_t a1Idx, int64_t a0OuterIdx)
{
    if (coreRCount_ <= 0) return;

    int64_t a0Start = a0OuterIdx * tileA0Len_;
    int64_t curA0Len = tileA0Len_;
    if (a0Start + curA0Len > a0Count_) {
        curA0Len = a0Count_ - a0Start;
    }

    // 初始化局部累加结果为 0
    LocalTensor<float> globalResult = globalResultBuf.Get<float>();
    Duplicate(globalResult, static_cast<float>(0.0f), static_cast<uint32_t>(alignedCols_));

#endif


    for (int64_t chunkIdx = 0; chunkIdx < coreNumChunks_; chunkIdx++) {
        int64_t rStart = coreRStart_ + chunkIdx * rChunkSize_;
        int64_t curRCount = (chunkIdx == coreNumChunks_ - 1) ? coreLastChunkSize_ : rChunkSize_;

        // ========== CopyIn ==========
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(rChunkSize_ * alignedCols_));
        } else {
            Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(rChunkSize_ * alignedCols_));
        }



        DataCopyParams copyInParams;
        int64_t blockLenBytes = curA0Len * static_cast<int64_t>(sizeof(T));
        int64_t paddedBlockLenBytes = ((blockLenBytes + 31) / 32) * 32;
        int64_t paddedA0Len = paddedBlockLenBytes / static_cast<int64_t>(sizeof(T));
        int64_t rightPadElems = paddedA0Len - curA0Len;
        int64_t dstGapBytes = (alignedCols_ - paddedA0Len) * static_cast<int64_t>(sizeof(T));
        bool usePadding = (rightPadElems > 0);

        if (copyBlockCount_ > 0) {
            // 非连续多轴归约: strided CopyIn
            int64_t gmBase = 0;
            int64_t remaining = a1Idx;
            for (int64_t d = nonReduceDimCount_ - 1; d >= 0; d--) {
                int64_t coord = remaining % nonReduceDimSizes_[d];
                remaining = remaining / nonReduceDimSizes_[d];
                gmBase += coord * nonReduceGmStrides_[d];
            }

            int64_t srcGapBytes = (a0Count_ - curA0Len) * static_cast<int64_t>(sizeof(T));

            int64_t ubRowWritten = 0;
            int64_t rRemain = curRCount;
            int64_t rPos = rStart;
            while (rRemain > 0) {
                int64_t blkIdx = rPos / innerRowsPerBlock_;
                int64_t rowInBlk = rPos % innerRowsPerBlock_;
                int64_t rowsThisBlk = innerRowsPerBlock_ - rowInBlk;
                if (rowsThisBlk > rRemain) rowsThisBlk = rRemain;

                int64_t gmReduceOffset = 0;
                int64_t blkRemaining = blkIdx;
                for (int64_t d = outerReduceDimCount_ - 1; d >= 0; d--) {
                    int64_t coord = blkRemaining % reduceDimSizes_[d];
                    blkRemaining = blkRemaining / reduceDimSizes_[d];
                    gmReduceOffset += coord * reduceGmStrides_[d];
                }
                int64_t gmBlockBase = gmBase + gmReduceOffset + rowInBlk * a0Count_ + a0Start;

                copyInParams.blockCount = static_cast<uint32_t>(rowsThisBlk);
                copyInParams.blockLen = static_cast<uint32_t>(blockLenBytes);
                copyInParams.srcStride = static_cast<uint32_t>(srcGapBytes);
                copyInParams.dstStride = static_cast<uint32_t>(dstGapBytes / 32);

                if (usePadding) {
                    DataCopyPad(xLocal[ubRowWritten * alignedCols_], inputGM[gmBlockBase], copyInParams,
                        {true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadElems), 0});
                } else {
                    DataCopyPad(xLocal[ubRowWritten * alignedCols_], inputGM[gmBlockBase], copyInParams,
                        {false, 0, 0, 0});
                }

                ubRowWritten += rowsThisBlk;
                rPos += rowsThisBlk;
                rRemain -= rowsThisBlk;
            }
        } else {
            // 连续模式
            int64_t srcGapBytes = (originalA0_ - curA0Len) * static_cast<int64_t>(sizeof(T));

            copyInParams.blockCount = static_cast<uint32_t>(curRCount);
            copyInParams.blockLen = static_cast<uint32_t>(blockLenBytes);
            copyInParams.srcStride = static_cast<uint32_t>(srcGapBytes);
            copyInParams.dstStride = static_cast<uint32_t>(dstGapBytes / 32);

            int64_t gmOffset = a1Idx * rCount_ * originalA0_ + rStart * originalA0_ + a0Start;
            if (usePadding) {
                DataCopyPad(xLocal, inputGM[gmOffset], copyInParams,
                    {true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadElems), 0});
            } else {
                DataCopyPad(xLocal, inputGM[gmOffset], copyInParams,
                    {false, 0, 0, 0});
            }
        }

        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.template DeQue<T>();

        // ========== Compute ==========
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
        LocalTensor<float> cleanLocal = cleanBuf.Get<float>();

        uint32_t totalCount = static_cast<uint32_t>(curRCount * alignedCols_);

        int64_t totalMaskElements = rChunkSize_ * alignedCols_;
        int64_t maskInt16Count = ((totalMaskElements / 8 + 31) / 32) * 32 / 2;
        if (maskInt16Count < 16) maskInt16Count = 16;
        LocalTensor<int16_t> maskInt16 = maskBuf.Get<int16_t>();
        Duplicate(maskInt16, static_cast<int16_t>(0), static_cast<uint32_t>(maskInt16Count));

        Duplicate(zeroLocal, static_cast<float>(0.0f), static_cast<uint32_t>(alignedCols_));

        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Compare(maskLocal, xFloat, xFloat, CMPMODE::EQ, totalCount);
            Duplicate(cleanLocal, static_cast<float>(0.0f), totalCount);
            Select(cleanLocal, maskLocal, xFloat, cleanLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, totalCount);
        } else {
            LocalTensor<float> castLocal = castBuf.Get<float>();
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, totalCount);
            Compare(maskLocal, castLocal, castLocal, CMPMODE::EQ, totalCount);
            Duplicate(cleanLocal, static_cast<float>(0.0f), totalCount);
            Select(cleanLocal, maskLocal, castLocal, cleanLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, totalCount);
        }

        // ReduceSum with Pattern::RA
        LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
        LocalTensor<float> chunkResult = yLocal.template ReinterpretCast<float>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(curRCount), static_cast<uint32_t>(alignedCols_)};
        LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();
        ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkResult, cleanLocal, tmpUint8, srcShape, true);






        // 跨 chunk 累加
        Add(globalResult, globalResult, chunkResult, static_cast<uint32_t>(alignedCols_));






        outQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // ========== AtomicAdd CopyOut ==========
    LocalTensor<T> yOut = outQueueY.template AllocTensor<T>();

    if constexpr (IS_FP32) {
        LocalTensor<float> yFloat = yOut.template ReinterpretCast<float>();
        DataCopy(yFloat, globalResult, static_cast<uint32_t>(alignedCols_));
    } else {
        Cast(yOut, globalResult, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedCols_));
    }



    outQueueY.EnQue(yOut);
    yOut = outQueueY.template DeQue<T>();

    // 使用 AtomicAdd 写出
    SetAtomicAdd<T>();
    DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(curA0Len * static_cast<int64_t>(sizeof(T)));
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int64_t outGmOffset = a1Idx * a0Count_ + a0Start;
    DataCopyPad(outputGM[outGmOffset], yOut, copyOutParams);
    SetAtomicNone();

    outQueueY.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void ReduceNansumAraRowsplit<T>::ProcessOneTile(int64_t a1Idx, int64_t a0OuterIdx)
{
    int64_t a0Start = a0OuterIdx * tileA0Len_;
    int64_t curA0Len = tileA0Len_;
    if (a0Start + curA0Len > a0Count_) {
        curA0Len = a0Count_ - a0Start;
    }

    // 初始化全局累加结果为 0
    LocalTensor<float> globalResult = globalResultBuf.Get<float>();
    Duplicate(globalResult, static_cast<float>(0.0f), static_cast<uint32_t>(alignedCols_));



    for (int64_t chunkIdx = 0; chunkIdx < numChunks_; chunkIdx++) {
        int64_t rStart = chunkIdx * rChunkSize_;
        int64_t curRCount = (chunkIdx == numChunks_ - 1) ? lastChunkSize_ : rChunkSize_;

        // ========== CopyIn ==========
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(rChunkSize_ * alignedCols_));
        } else {
            Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(rChunkSize_ * alignedCols_));
        }



        DataCopyParams copyInParams;
        int64_t blockLenBytes = curA0Len * static_cast<int64_t>(sizeof(T));
        int64_t paddedBlockLenBytes = ((blockLenBytes + 31) / 32) * 32;
        int64_t paddedA0Len = paddedBlockLenBytes / static_cast<int64_t>(sizeof(T));
        int64_t rightPadElems = paddedA0Len - curA0Len;
        int64_t dstGapBytes = (alignedCols_ - paddedA0Len) * static_cast<int64_t>(sizeof(T));
        bool usePadding = (rightPadElems > 0);

        if (copyBlockCount_ > 0) {
            // 非连续多轴归约: strided CopyIn (chunk within strided blocks)
            int64_t gmBase = 0;
            int64_t remaining = a1Idx;
            for (int64_t d = nonReduceDimCount_ - 1; d >= 0; d--) {
                int64_t coord = remaining % nonReduceDimSizes_[d];
                remaining = remaining / nonReduceDimSizes_[d];
                gmBase += coord * nonReduceGmStrides_[d];
            }

            int64_t srcGapBytes = (a0Count_ - curA0Len) * static_cast<int64_t>(sizeof(T));

            // rStart..rStart+curRCount-1 映射到 (block, row_within_block)
            int64_t ubRowWritten = 0;
            int64_t rRemain = curRCount;
            int64_t rPos = rStart;
            while (rRemain > 0) {
                int64_t blkIdx = rPos / innerRowsPerBlock_;
                int64_t rowInBlk = rPos % innerRowsPerBlock_;
                int64_t rowsThisBlk = innerRowsPerBlock_ - rowInBlk;
                if (rowsThisBlk > rRemain) rowsThisBlk = rRemain;

                // 通过外层归约维度坐标计算正确的 GM 偏移
                int64_t gmReduceOffset = 0;
                int64_t blkRemaining = blkIdx;
                for (int64_t d = outerReduceDimCount_ - 1; d >= 0; d--) {
                    int64_t coord = blkRemaining % reduceDimSizes_[d];
                    blkRemaining = blkRemaining / reduceDimSizes_[d];
                    gmReduceOffset += coord * reduceGmStrides_[d];
                }
                int64_t gmBlockBase = gmBase + gmReduceOffset + rowInBlk * a0Count_ + a0Start;

                copyInParams.blockCount = static_cast<uint32_t>(rowsThisBlk);
                copyInParams.blockLen = static_cast<uint32_t>(blockLenBytes);
                copyInParams.srcStride = static_cast<uint32_t>(srcGapBytes);
                copyInParams.dstStride = static_cast<uint32_t>(dstGapBytes / 32);

                if (usePadding) {
                    DataCopyPad(xLocal[ubRowWritten * alignedCols_], inputGM[gmBlockBase], copyInParams,
                        {true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadElems), 0});
                } else {
                    DataCopyPad(xLocal[ubRowWritten * alignedCols_], inputGM[gmBlockBase], copyInParams,
                        {false, 0, 0, 0});
                }

                ubRowWritten += rowsThisBlk;
                rPos += rowsThisBlk;
                rRemain -= rowsThisBlk;
            }
        } else {
            // 连续模式
            int64_t srcGapBytes = (originalA0_ - curA0Len) * static_cast<int64_t>(sizeof(T));

            copyInParams.blockCount = static_cast<uint32_t>(curRCount);
            copyInParams.blockLen = static_cast<uint32_t>(blockLenBytes);
            copyInParams.srcStride = static_cast<uint32_t>(srcGapBytes);
            copyInParams.dstStride = static_cast<uint32_t>(dstGapBytes / 32);

            int64_t gmOffset = a1Idx * rCount_ * originalA0_ + rStart * originalA0_ + a0Start;
            if (usePadding) {
                DataCopyPad(xLocal, inputGM[gmOffset], copyInParams,
                    {true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadElems), 0});
            } else {
                DataCopyPad(xLocal, inputGM[gmOffset], copyInParams,
                    {false, 0, 0, 0});
            }
        }

        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.template DeQue<T>();

        // ========== Compute ==========
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
        LocalTensor<float> cleanLocal = cleanBuf.Get<float>();

        uint32_t totalCount = static_cast<uint32_t>(curRCount * alignedCols_);

        // 清零 mask
        int64_t totalMaskElements = rChunkSize_ * alignedCols_;
        int64_t maskInt16Count = ((totalMaskElements / 8 + 31) / 32) * 32 / 2;
        if (maskInt16Count < 16) maskInt16Count = 16;
        LocalTensor<int16_t> maskInt16 = maskBuf.Get<int16_t>();
        Duplicate(maskInt16, static_cast<int16_t>(0), static_cast<uint32_t>(maskInt16Count));

        Duplicate(zeroLocal, static_cast<float>(0.0f), static_cast<uint32_t>(alignedCols_));

        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Compare(maskLocal, xFloat, xFloat, CMPMODE::EQ, totalCount);
            Duplicate(cleanLocal, static_cast<float>(0.0f), totalCount);
            Select(cleanLocal, maskLocal, xFloat, cleanLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, totalCount);
        } else {
            // fp16/bf16: Cast 到 fp32 后再 Compare（bf16 不支持直接 Compare）
            LocalTensor<float> castLocal = castBuf.Get<float>();
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, totalCount);
            Compare(maskLocal, castLocal, castLocal, CMPMODE::EQ, totalCount);
            Duplicate(cleanLocal, static_cast<float>(0.0f), totalCount);
            Select(cleanLocal, maskLocal, castLocal, cleanLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, totalCount);
        }

        // ReduceSum with Pattern::RA
        LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
        LocalTensor<float> chunkResult = yLocal.template ReinterpretCast<float>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(curRCount), static_cast<uint32_t>(alignedCols_)};
        LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();
        ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkResult, cleanLocal, tmpUint8, srcShape, true);






        // 跨 chunk 累加
        Add(globalResult, globalResult, chunkResult, static_cast<uint32_t>(alignedCols_));






        outQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // ========== CopyOut ==========
    LocalTensor<T> yOut = outQueueY.template AllocTensor<T>();

    if constexpr (IS_FP32) {
        LocalTensor<float> yFloat = yOut.template ReinterpretCast<float>();
        DataCopy(yFloat, globalResult, static_cast<uint32_t>(alignedCols_));
    } else {
        // Cast fp32 -> 原始类型
        Cast(yOut, globalResult, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedCols_));
    }



    outQueueY.EnQue(yOut);
    yOut = outQueueY.template DeQue<T>();

    DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(curA0Len * static_cast<int64_t>(sizeof(T)));
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int64_t outGmOffset = a1Idx * a0Count_ + a0Start;
    DataCopyPad(outputGM[outGmOffset], yOut, copyOutParams);

    outQueueY.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void ReduceNansumAraRowsplit<T>::Process()
{
    if (useAtomicAdd_) {
        // AtomicAdd 模式：核0 初始化输出为0，全核同步后各核处理自己的 R 区间
        int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx == 0) {
            // 核0 初始化输出 GM 为0
            LocalTensor<T> yOut = outQueueY.template AllocTensor<T>();
            if constexpr (IS_FP32) {
                LocalTensor<float> yFloat = yOut.template ReinterpretCast<float>();
                Duplicate(yFloat, static_cast<float>(0.0f), static_cast<uint32_t>(alignedCols_));
            } else {
                Duplicate(yOut, static_cast<T>(0), static_cast<uint32_t>(alignedCols_));
            }


            outQueueY.EnQue(yOut);
            yOut = outQueueY.template DeQue<T>();
            // 非原子写0（覆盖所有输出元素）
            int64_t totalOutElems = a1Count_ * a0Count_;
            int64_t outWritten = 0;
            while (outWritten < totalOutElems) {
                int64_t writeLen = alignedCols_;
                if (outWritten + writeLen > totalOutElems) {
                    writeLen = totalOutElems - outWritten;
                }
                DataCopyParams initParams;
                initParams.blockCount = 1;
                initParams.blockLen = static_cast<uint32_t>(writeLen * static_cast<int64_t>(sizeof(T)));
                initParams.srcStride = 0;
                initParams.dstStride = 0;
                DataCopyPad(outputGM[outWritten], yOut, initParams);
                outWritten += writeLen;
            }
            outQueueY.FreeTensor(yOut);
        }
        SyncAll();

        // 各核处理自己的 R 区间（每核处理同一 tile，但不同 R 范围）
        for (int64_t t = 0; t < tileCount_; t++) {
            int64_t globalTileIdx = startTile_ + t;
            int64_t a1Idx = globalTileIdx / a0Outer_;
            int64_t a0OuterIdx = globalTileIdx % a0Outer_;
            ProcessOneTileAtomicAdd(a1Idx, a0OuterIdx);
        }
    } else {
        // 原始模式
        for (int64_t t = 0; t < tileCount_; t++) {
            int64_t globalTileIdx = startTile_ + t;
            int64_t a1Idx = globalTileIdx / a0Outer_;
            int64_t a0OuterIdx = globalTileIdx % a0Outer_;
            ProcessOneTile(a1Idx, a0OuterIdx);
        }
    }
}

} // namespace NsReduceNansum

