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
 * \file reduce_nansum_ar_colsplit.h
 * \brief ReduceNansum AR ColSplit Kernel 实现（TilingKey=1）
 *
 * AR 模板（A0=1）分载模式：当 R > fullLoadThreshold 时，
 * 将每行沿 R 方向分 chunk 处理，每个 chunk 独立完成
 * IsNan → Select → ReduceSum，最后累加所有 chunk 的结果。
 *
 * 迭代三：支持 fp32/fp16/bf16 混合精度。
 * 迭代三精度修复：使用 Kahan 补偿求和解决大 R 值下的浮点精度累积误差。
 * AtomicAdd 多核优化：当 A1=1 时，按 R 维度切分多核并行，使用 AtomicAdd 汇聚结果。
 */
#ifndef REDUCE_NANSUM_AR_COLSPLIT_H
#define REDUCE_NANSUM_AR_COLSPLIT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_nansum_tiling_data.h"
#include "reduce_nansum_tiling_key.h"

namespace NsReduceNansum {

using namespace AscendC;

template <typename T>
class ReduceNansumArColsplit {
public:
    __aicore__ inline ReduceNansumArColsplit() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReduceNansumTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneRow(int64_t globalRowIdx);
    __aicore__ inline void ProcessOneRowAtomicAdd(int64_t globalRowIdx);

private:
    static constexpr bool IS_FP32 = sizeof(T) == sizeof(float);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> cleanBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> chunkResultBuf;
    TBuf<QuePosition::VECCALC> globalResultBuf;
    TBuf<QuePosition::VECCALC> kahanCompBuf;   // Kahan 补偿项
    TBuf<QuePosition::VECCALC> kahanTmpBuf;    // Kahan 临时缓冲
    TBuf<QuePosition::VECCALC> castBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t a1Count_ = 0;
    int64_t rCount_ = 0;
    int64_t rLengthAlign_ = 0;
    int64_t tmpBufSize_ = 0;
    int64_t rChunkSize_ = 0;
    int64_t numChunks_ = 0;
    int64_t lastChunkSize_ = 0;

    // 多核参数
    int64_t startRow_ = 0;
    int64_t rowCount_ = 0;

    // AtomicAdd 多核归约参数
    int64_t useAtomicAdd_ = 0;
    int64_t coreRStart_ = 0;      // 本核处理的 R 起始位置
    int64_t coreRCount_ = 0;      // 本核处理的 R 数量
    int64_t coreNumChunks_ = 0;   // 本核的 chunk 数
    int64_t coreLastChunkSize_ = 0; // 本核最后一个 chunk 大小

    // 非连续多轴归约参数
    int64_t copyBlockCount_ = 0;
    int64_t copyBlockLen_ = 0;
    int64_t copySrcStride_ = 0;
    int64_t outputStride_ = 0;
    // 非归约维度信息
    int64_t nonReduceDimCount_ = 0;
    int64_t nonReduceDimSizes_[8] = {0};
    int64_t nonReduceGmStrides_[8] = {0};
    // 归约维度信息（用于3+非连续轴的逐块GM偏移计算）
    int64_t reduceDimCount_ = 0;
    int64_t reduceDimSizes_[8] = {0};
    int64_t reduceGmStrides_[8] = {0};
    int64_t outerReduceDimCount_ = 0;
    int64_t innerBlockSize_ = 0;  // innerBlockElems
};

template <typename T>
__aicore__ inline void ReduceNansumArColsplit<T>::Init(GM_ADDR x, GM_ADDR y,
                                                       const ReduceNansumTilingData* tilingData)
{
    a1Count_ = tilingData->a1Count;
    rCount_ = tilingData->rCount;
    rLengthAlign_ = tilingData->rLengthAlign;
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
        // AtomicAdd 模式：每核处理同一行（A1=1），但处理不同的 R 区间
        startRow_ = 0;
        rowCount_ = 1;

        int64_t rPerCore = tilingData->rPerCore;
        coreRStart_ = blockIdx * rPerCore;
        coreRCount_ = rPerCore;
        // 尾核处理剩余
        if (coreRStart_ + coreRCount_ > rCount_) {
            coreRCount_ = rCount_ - coreRStart_;
        }
        if (coreRCount_ < 0) coreRCount_ = 0;

        // 计算本核内的 chunk 参数
        if (coreRCount_ > 0) {
            coreNumChunks_ = (coreRCount_ + rChunkSize_ - 1) / rChunkSize_;
            coreLastChunkSize_ = coreRCount_ - (coreNumChunks_ - 1) * rChunkSize_;
            if (coreLastChunkSize_ <= 0) coreLastChunkSize_ = rChunkSize_;
        } else {
            coreNumChunks_ = 0;
            coreLastChunkSize_ = 0;
        }
    } else {
        // 原始模式：按 A1 行切分
        startRow_ = blockIdx * tilesPerCore;
        if (blockIdx < usedCoreNum - 1) {
            rowCount_ = tilesPerCore;
        } else {
            rowCount_ = tailCoreTiles;
        }
    }

    // 设置 GM 指针
    int64_t inputGmSize = a1Count_ * rCount_;

    // 非连续多轴归约参数
    copyBlockCount_ = tilingData->copyBlockCount;
    copyBlockLen_ = tilingData->copyBlockLen;
    copySrcStride_ = tilingData->copySrcStride;
    outputStride_ = tilingData->outputStride;
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
    innerBlockSize_ = 0;
    outerReduceDimCount_ = reduceDimCount_;
    if (copyBlockCount_ > 0) {
        innerBlockSize_ = copyBlockLen_ / static_cast<int64_t>(sizeof(T));
        int64_t innerProduct = 1;
        for (int64_t d = reduceDimCount_ - 1; d >= 0; d--) {
            innerProduct *= reduceDimSizes_[d];
            if (innerProduct == innerBlockSize_) {
                outerReduceDimCount_ = d;
                break;
            }
        }
    }

    if (copyBlockCount_ > 0) {
        int64_t maxGmOffset = 0;
        for (int64_t d = 0; d < nonReduceDimCount_; d++) {
            maxGmOffset += (nonReduceDimSizes_[d] - 1) * nonReduceGmStrides_[d];
        }
        for (int64_t d = 0; d < reduceDimCount_; d++) {
            maxGmOffset += (reduceDimSizes_[d] - 1) * reduceGmStrides_[d];
        }
        maxGmOffset += 1;
        inputGmSize = maxGmOffset;
    }
    inputGM.SetGlobalBuffer((__gm__ T*)x, inputGmSize);
    outputGM.SetGlobalBuffer((__gm__ T*)y, a1Count_);

    // 初始化 Buffer（单缓冲，按 chunk 大小分配）
    pipe.InitBuffer(inQueueX, 1, rLengthAlign_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outQueueY, 1, 32);
    // maskBuf
    int64_t maskBufSize = ((rLengthAlign_ / 8 + 31) / 32) * 32;
    if (maskBufSize < 32) maskBufSize = 32;
    pipe.InitBuffer(maskBuf, maskBufSize);
    pipe.InitBuffer(zeroBuf, rLengthAlign_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(cleanBuf, rLengthAlign_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(tmpBuf, tmpBufSize_);
    pipe.InitBuffer(chunkResultBuf, 32);
    pipe.InitBuffer(globalResultBuf, 32);
    pipe.InitBuffer(kahanCompBuf, 32);
    pipe.InitBuffer(kahanTmpBuf, 32);

    if constexpr (!IS_FP32) {
        pipe.InitBuffer(castBuf, rLengthAlign_ * static_cast<int64_t>(sizeof(float)));
    }
}

// AtomicAdd 模式：每核处理自己的 R 区间，结果通过 AtomicAdd 写到 GM
template <typename T>
__aicore__ inline void ReduceNansumArColsplit<T>::ProcessOneRowAtomicAdd(int64_t globalRowIdx)
{
    if (coreRCount_ <= 0) return;

    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    LocalTensor<float> cleanLocal = cleanBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    LocalTensor<float> globalResult = globalResultBuf.Get<float>();
    constexpr uint32_t SCALAR_ALIGN = 8;
    Duplicate(globalResult, static_cast<float>(0.0f), SCALAR_ALIGN);

#endif


    // 判断是否为非连续多轴归约
    bool isStrided = (copyBlockCount_ > 0);
    int64_t gmRowBase = 0;
    if (isStrided) {
        int64_t remaining = globalRowIdx;
        for (int64_t d = nonReduceDimCount_ - 1; d >= 0; d--) {
            int64_t coord = remaining % nonReduceDimSizes_[d];
            remaining = remaining / nonReduceDimSizes_[d];
            gmRowBase += coord * nonReduceGmStrides_[d];
        }
    } else {
        gmRowBase = globalRowIdx * rCount_;
    }

    for (int64_t chunkIdx = 0; chunkIdx < coreNumChunks_; chunkIdx++) {
        int64_t chunkStart = coreRStart_ + chunkIdx * rChunkSize_;
        int64_t curChunkSize = (chunkIdx == coreNumChunks_ - 1) ? coreLastChunkSize_ : rChunkSize_;

        // CopyIn
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(rLengthAlign_));
        } else {
            Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(rLengthAlign_));
        }



        DataCopyParams copyParams;
        if (isStrided) {
            int64_t startBlock = chunkStart / innerBlockSize_;
            int64_t endElem = chunkStart + curChunkSize;
            int64_t endBlock = (endElem + innerBlockSize_ - 1) / innerBlockSize_;
            int64_t blocksInChunk = endBlock - startBlock;

            for (int64_t blk = 0; blk < blocksInChunk; blk++) {
                int64_t outerBlkIdx = startBlock + blk;
                int64_t gmReduceOffset = 0;
                int64_t blkRemaining = outerBlkIdx;
                for (int64_t d = outerReduceDimCount_ - 1; d >= 0; d--) {
                    int64_t coord = blkRemaining % reduceDimSizes_[d];
                    blkRemaining = blkRemaining / reduceDimSizes_[d];
                    gmReduceOffset += coord * reduceGmStrides_[d];
                }

                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(copyBlockLen_);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(xLocal[blk * innerBlockSize_],
                            inputGM[gmRowBase + gmReduceOffset],
                            copyParams, {false, 0, 0, 0});
            }
        } else {
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curChunkSize * static_cast<int64_t>(sizeof(T)));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, inputGM[gmRowBase + chunkStart], copyParams, {false, 0, 0, 0});
        }

        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.template DeQue<T>();

        // Compute
        uint32_t count = static_cast<uint32_t>(rLengthAlign_);

        int64_t maskInt16Count = ((rLengthAlign_ / 8 + 31) / 32) * 32 / 2;
        if (maskInt16Count < 16) maskInt16Count = 16;
        LocalTensor<int16_t> maskInt16 = maskBuf.Get<int16_t>();
        Duplicate(maskInt16, static_cast<int16_t>(0), static_cast<uint32_t>(maskInt16Count));

        Duplicate(zeroLocal, static_cast<float>(0.0f), count);

        if constexpr (IS_FP32) {
            LocalTensor<float> xFp32 = xLocal.template ReinterpretCast<float>();
            Compare(maskLocal, xFp32, xFp32, CMPMODE::EQ, count);
            Select(cleanLocal, maskLocal, xFp32, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        } else {
            LocalTensor<float> castLocal = castBuf.Get<float>();
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, count);
            Compare(maskLocal, castLocal, castLocal, CMPMODE::EQ, count);
            Select(cleanLocal, maskLocal, castLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }

        LocalTensor<float> chunkResult = chunkResultBuf.Get<float>();
        ReduceSum(chunkResult, cleanLocal, tmpLocal, static_cast<uint32_t>(curChunkSize));



        // 简单累加（每核R小，不需要Kahan补偿）
        Add(globalResult, globalResult, chunkResult, SCALAR_ALIGN);



        inQueueX.FreeTensor(xLocal);
    }

    // AtomicAdd 写出：写 fp32 结果到 GM，32字节对齐（8个float = 32字节）
    LocalTensor<T> yOut = outQueueY.template AllocTensor<T>();

    if constexpr (IS_FP32) {
        LocalTensor<float> yFloat = yOut.template ReinterpretCast<float>();
        DataCopy(yFloat, globalResult, SCALAR_ALIGN);
    } else {
        LocalTensor<T> yTyped = yOut;
        Cast(yTyped, globalResult, RoundMode::CAST_ROUND, SCALAR_ALIGN);
    }



    outQueueY.EnQue(yOut);
    yOut = outQueueY.template DeQue<T>();

    // 使用 AtomicAdd 写出
    SetAtomicAdd<T>();
    DataCopyParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = static_cast<uint32_t>(sizeof(T));
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(outputGM[globalRowIdx], yOut, outParams);
    SetAtomicNone();

    outQueueY.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void ReduceNansumArColsplit<T>::ProcessOneRow(int64_t globalRowIdx)
{
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    LocalTensor<float> cleanLocal = cleanBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    LocalTensor<float> globalResult = globalResultBuf.Get<float>();
    LocalTensor<float> kahanComp = kahanCompBuf.Get<float>();
    LocalTensor<float> kahanTmp = kahanTmpBuf.Get<float>();
    constexpr uint32_t SCALAR_ALIGN = 8;
    Duplicate(globalResult, static_cast<float>(0.0f), SCALAR_ALIGN);
    Duplicate(kahanComp, static_cast<float>(0.0f), SCALAR_ALIGN);



    // 判断是否为非连续多轴归约
    bool isStrided = (copyBlockCount_ > 0);
    int64_t gmRowBase = 0;
    if (isStrided) {
        // 通过非归约维度坐标计算 GM 基地址
        int64_t remaining = globalRowIdx;
        for (int64_t d = nonReduceDimCount_ - 1; d >= 0; d--) {
            int64_t coord = remaining % nonReduceDimSizes_[d];
            remaining = remaining / nonReduceDimSizes_[d];
            gmRowBase += coord * nonReduceGmStrides_[d];
        }
    } else {
        gmRowBase = globalRowIdx * rCount_;
    }

    for (int64_t chunkIdx = 0; chunkIdx < numChunks_; chunkIdx++) {
        int64_t chunkStart = chunkIdx * rChunkSize_;
        int64_t curChunkSize = (chunkIdx == numChunks_ - 1) ? lastChunkSize_ : rChunkSize_;

        // CopyIn
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(rLengthAlign_));
        } else {
            Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(rLengthAlign_));
        }



        DataCopyParams copyParams;
        if (isStrided) {
            // 非连续多轴归约的 chunk 处理：逐块拷贝确保 UB 连续
            // 计算 chunk 中的块范围
            int64_t startBlock = chunkStart / innerBlockSize_;
            int64_t endElem = chunkStart + curChunkSize;
            int64_t endBlock = (endElem + innerBlockSize_ - 1) / innerBlockSize_;
            int64_t blocksInChunk = endBlock - startBlock;

            for (int64_t blk = 0; blk < blocksInChunk; blk++) {
                int64_t outerBlkIdx = startBlock + blk;
                // 通过外层归约维度坐标计算正确的 GM 偏移
                int64_t gmReduceOffset = 0;
                int64_t blkRemaining = outerBlkIdx;
                for (int64_t d = outerReduceDimCount_ - 1; d >= 0; d--) {
                    int64_t coord = blkRemaining % reduceDimSizes_[d];
                    blkRemaining = blkRemaining / reduceDimSizes_[d];
                    gmReduceOffset += coord * reduceGmStrides_[d];
                }

                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(copyBlockLen_);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(xLocal[blk * innerBlockSize_],
                            inputGM[gmRowBase + gmReduceOffset],
                            copyParams, {false, 0, 0, 0});
            }
        } else {
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curChunkSize * static_cast<int64_t>(sizeof(T)));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, inputGM[gmRowBase + chunkStart], copyParams, {false, 0, 0, 0});
        }

        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.template DeQue<T>();

        // Compute
        uint32_t count = static_cast<uint32_t>(rLengthAlign_);

        // 清零 mask
        int64_t maskInt16Count = ((rLengthAlign_ / 8 + 31) / 32) * 32 / 2;
        if (maskInt16Count < 16) maskInt16Count = 16;
        LocalTensor<int16_t> maskInt16 = maskBuf.Get<int16_t>();
        Duplicate(maskInt16, static_cast<int16_t>(0), static_cast<uint32_t>(maskInt16Count));

        Duplicate(zeroLocal, static_cast<float>(0.0f), count);

        if constexpr (IS_FP32) {
            LocalTensor<float> xFp32 = xLocal.template ReinterpretCast<float>();
            Compare(maskLocal, xFp32, xFp32, CMPMODE::EQ, count);
            Select(cleanLocal, maskLocal, xFp32, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        } else {
            // fp16/bf16: Cast 到 fp32 后再 Compare（bf16 不支持直接 Compare）
            LocalTensor<float> castLocal = castBuf.Get<float>();
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, count);
            Compare(maskLocal, castLocal, castLocal, CMPMODE::EQ, count);
            Select(cleanLocal, maskLocal, castLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }

        LocalTensor<float> chunkResult = chunkResultBuf.Get<float>();
        ReduceSum(chunkResult, cleanLocal, tmpLocal, static_cast<uint32_t>(curChunkSize));



        // Kahan 补偿求和：y = chunkResult - comp; t = globalResult + y; comp = (t - globalResult) - y; globalResult = t
        // Step 1: y = chunkResult - kahanComp (补偿后的增量)
        Sub(kahanTmp, chunkResult, kahanComp, SCALAR_ALIGN);


        // Step 2: t = globalResult + y (新的累加和)
        // 先保存旧 globalResult 到 chunkResult (临时复用)
        DataCopy(chunkResult, globalResult, SCALAR_ALIGN);


        Add(globalResult, globalResult, kahanTmp, SCALAR_ALIGN);


        // Step 3: comp = (t - oldGlobalResult) - y = (globalResult - chunkResult) - kahanTmp
        Sub(kahanComp, globalResult, chunkResult, SCALAR_ALIGN);


        Sub(kahanComp, kahanComp, kahanTmp, SCALAR_ALIGN);



        inQueueX.FreeTensor(xLocal);
    }

    // 写回结果
    LocalTensor<T> yOut = outQueueY.template AllocTensor<T>();

    if constexpr (IS_FP32) {
        LocalTensor<float> yFloat = yOut.template ReinterpretCast<float>();
        DataCopy(yFloat, globalResult, SCALAR_ALIGN);
    } else {
        // Cast fp32 -> 原始类型
        LocalTensor<T> yTyped = yOut;
        Cast(yTyped, globalResult, RoundMode::CAST_ROUND, SCALAR_ALIGN);
    }



    outQueueY.EnQue(yOut);
    yOut = outQueueY.template DeQue<T>();

    DataCopyParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = static_cast<uint32_t>(sizeof(T));
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(outputGM[globalRowIdx], yOut, outParams);

    outQueueY.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void ReduceNansumArColsplit<T>::Process()
{
    if (useAtomicAdd_) {
        // AtomicAdd 模式：核0 初始化输出为0，全核同步后各核处理自己的 R 区间
        int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx == 0) {
            // 核0 初始化输出 GM 为0
            LocalTensor<T> yOut = outQueueY.template AllocTensor<T>();
            if constexpr (IS_FP32) {
                LocalTensor<float> yFloat = yOut.template ReinterpretCast<float>();
                Duplicate(yFloat, static_cast<float>(0.0f), 8u);
            } else {
                Duplicate(yOut, static_cast<T>(0), 8u);
            }


            outQueueY.EnQue(yOut);
            yOut = outQueueY.template DeQue<T>();
            // 非原子写0（覆盖所有输出元素）
            for (int64_t i = 0; i < a1Count_; i++) {
                DataCopyParams initParams;
                initParams.blockCount = 1;
                initParams.blockLen = static_cast<uint32_t>(sizeof(T));
                initParams.srcStride = 0;
                initParams.dstStride = 0;
                DataCopyPad(outputGM[i], yOut, initParams);
            }
            outQueueY.FreeTensor(yOut);
        }
        SyncAll();

        // 各核处理自己的 R 区间
        for (int64_t i = 0; i < rowCount_; i++) {
            ProcessOneRowAtomicAdd(startRow_ + i);
        }
    } else {
        // 原始模式
        for (int64_t i = 0; i < rowCount_; i++) {
            ProcessOneRow(startRow_ + i);
        }
    }
}

} // namespace NsReduceNansum

