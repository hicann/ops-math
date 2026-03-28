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
 * \file reduce_nansum_ara_fullload.h
 * \brief ReduceNansum ARA 全载 Kernel 实现（TilingKey=2）
 *
 * ARA 模板（A0>1）全载模式：将 R 行全部载入 UB，
 * 执行 IsNan → Select → ReduceSum(Pattern::RA) 将 R 行归约为 1 行。
 *
 * 迭代三：支持 fp32/fp16/bf16 混合精度。
 */
#ifndef REDUCE_NANSUM_ARA_FULLLOAD_H
#define REDUCE_NANSUM_ARA_FULLLOAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_nansum_tiling_data.h"
#include "reduce_nansum_tiling_key.h"

namespace NsReduceNansum {

using namespace AscendC;

template <typename T>
class ReduceNansumAraFullload {
public:
    __aicore__ inline ReduceNansumAraFullload() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReduceNansumTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneTile(int64_t a1Idx, int64_t a0OuterIdx);

private:
    static constexpr bool IS_FP32 = sizeof(T) == sizeof(float);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> cleanBuf;
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

    // 多核参数
    int64_t startTile_ = 0;
    int64_t tileCount_ = 0;

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
    int64_t innerRowsPerBlock_ = 0;  // 最内层连续归约组包含的行数
};

template <typename T>
__aicore__ inline void ReduceNansumAraFullload<T>::Init(GM_ADDR x, GM_ADDR y,
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

    // 多核切分
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t tilesPerCore = tilingData->tilesPerCore;
    int64_t tailCoreTiles = tilingData->tailCoreTiles;
    int64_t usedCoreNum = tilingData->usedCoreNum;

    startTile_ = blockIdx * tilesPerCore;
    if (blockIdx < usedCoreNum - 1) {
        tileCount_ = tilesPerCore;
    } else {
        tileCount_ = tailCoreTiles;
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
    // 计算最内层连续归约组的行数
    innerRowsPerBlock_ = 0;
    if (copyBlockCount_ > 0 && a0Count_ > 0) {
        int64_t innerBlockElems = copyBlockLen_ / static_cast<int64_t>(sizeof(T));
        innerRowsPerBlock_ = innerBlockElems / a0Count_;
    }

    int64_t inputGmSize = a1Count_ * rCount_ * a0Count_;
    if (copyBlockCount_ > 0) {
        // 非连续多轴归约：计算 GM 最远元素位置
        int64_t maxGmOffset = 0;
        for (int64_t d = 0; d < nonReduceDimCount_; d++) {
            maxGmOffset += (nonReduceDimSizes_[d] - 1) * nonReduceGmStrides_[d];
        }
        for (int64_t d = 0; d < reduceDimCount_; d++) {
            maxGmOffset += (reduceDimSizes_[d] - 1) * reduceGmStrides_[d];
        }
        maxGmOffset += a0Count_;  // 最内层 A0 个元素
        inputGmSize = maxGmOffset;
    }
    inputGM.SetGlobalBuffer((__gm__ T*)x, inputGmSize);
    outputGM.SetGlobalBuffer((__gm__ T*)y, a1Count_ * a0Count_);

    // 初始化 Buffer
    int64_t inBufSize = rCount_ * alignedCols_ * static_cast<int64_t>(sizeof(T));
    int64_t outBufSize = alignedCols_ * static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(inQueueX, 1, inBufSize);
    pipe.InitBuffer(outQueueY, 1, outBufSize);

    // maskBuf
    int64_t totalMaskElements = rCount_ * alignedCols_;
    int64_t maskBufSize = ((totalMaskElements / 8 + 31) / 32) * 32;
    if (maskBufSize < 32) maskBufSize = 32;
    pipe.InitBuffer(maskBuf, maskBufSize);

    pipe.InitBuffer(zeroBuf, alignedCols_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(cleanBuf, rCount_ * alignedCols_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(tmpBuf, tmpBufSize_);

    if constexpr (!IS_FP32) {
        pipe.InitBuffer(castBuf, rCount_ * alignedCols_ * static_cast<int64_t>(sizeof(float)));
    }
}

template <typename T>
__aicore__ inline void ReduceNansumAraFullload<T>::ProcessOneTile(int64_t a1Idx, int64_t a0OuterIdx)
{
    int64_t a0Start = a0OuterIdx * tileA0Len_;
    int64_t curA0Len = tileA0Len_;
    if (a0Start + curA0Len > a0Count_) {
        curA0Len = a0Count_ - a0Start;
    }

    // ========== CopyIn ==========
    LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

    // 清零
    if constexpr (IS_FP32) {
        LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
        Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(rCount_ * alignedCols_));
    } else {
        Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(rCount_ * alignedCols_));
    }

#endif


    if (copyBlockCount_ > 0) {
        // 非连续多轴归约: strided CopyIn
        // 计算 a1Idx 对应的 GM 基地址（从非归约维度坐标计算）
        int64_t gmBase = 0;
        int64_t remaining = a1Idx;
        for (int64_t d = nonReduceDimCount_ - 1; d >= 0; d--) {
            int64_t coord = remaining % nonReduceDimSizes_[d];
            remaining = remaining / nonReduceDimSizes_[d];
            gmBase += coord * nonReduceGmStrides_[d];
        }

        // 确定外层归约维度数量（排除最内层连续归约组）
        // innerRowsPerBlock_ = product of innermost contiguous reduce dims
        // 从最后一个 reduce dim 向前数，直到乘积 = innerRowsPerBlock_
        int64_t outerReduceDimCount = reduceDimCount_;
        {
            int64_t innerProduct = 1;
            for (int64_t d = reduceDimCount_ - 1; d >= 0; d--) {
                innerProduct *= reduceDimSizes_[d];
                if (innerProduct == innerRowsPerBlock_) {
                    outerReduceDimCount = d;
                    break;
                }
            }
        }

        // 逐外层块搬运：通过外层归约维度坐标计算正确的 GM 偏移
        // 当 blockLen 不是 32 字节对齐时，使用 DataCopyPad 的 rightPadding 补齐
        int64_t blockLenBytes = curA0Len * static_cast<int64_t>(sizeof(T));
        int64_t paddedBlockLenBytes = ((blockLenBytes + 31) / 32) * 32;
        int64_t paddedA0Len = paddedBlockLenBytes / static_cast<int64_t>(sizeof(T));
        int64_t rightPadElems = paddedA0Len - curA0Len;
        int64_t dstGapBytes = (alignedCols_ - paddedA0Len) * static_cast<int64_t>(sizeof(T));
        int64_t srcGapBytes = (a0Count_ - curA0Len) * static_cast<int64_t>(sizeof(T));
        bool usePadding = (rightPadElems > 0);

        for (int64_t blk = 0; blk < copyBlockCount_; blk++) {
            int64_t ubRowStart = blk * innerRowsPerBlock_;

            // 将 blk 分解为外层归约维度坐标，计算 GM 偏移
            int64_t gmReduceOffset = 0;
            int64_t blkRemaining = blk;
            for (int64_t d = outerReduceDimCount - 1; d >= 0; d--) {
                int64_t coord = blkRemaining % reduceDimSizes_[d];
                blkRemaining = blkRemaining / reduceDimSizes_[d];
                gmReduceOffset += coord * reduceGmStrides_[d];
            }
            int64_t gmBlockBase = gmBase + gmReduceOffset + a0Start;

            DataCopyParams copyInParams;
            copyInParams.blockCount = static_cast<uint32_t>(innerRowsPerBlock_);
            copyInParams.blockLen = static_cast<uint32_t>(blockLenBytes);
            copyInParams.srcStride = static_cast<uint32_t>(srcGapBytes);
            copyInParams.dstStride = static_cast<uint32_t>(dstGapBytes / 32);

            if (usePadding) {
                DataCopyPad(xLocal[ubRowStart * alignedCols_], inputGM[gmBlockBase], copyInParams,
                    {true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadElems), 0});
            } else {
                DataCopyPad(xLocal[ubRowStart * alignedCols_], inputGM[gmBlockBase], copyInParams,
                    {false, 0, 0, 0});
            }
        }
    } else {
        // 连续模式: 标准 CopyIn
        int64_t blockLenBytes = curA0Len * static_cast<int64_t>(sizeof(T));
        int64_t paddedBlockLenBytes = ((blockLenBytes + 31) / 32) * 32;
        int64_t paddedA0Len = paddedBlockLenBytes / static_cast<int64_t>(sizeof(T));
        int64_t rightPadElems = paddedA0Len - curA0Len;
        int64_t dstGapBytes = (alignedCols_ - paddedA0Len) * static_cast<int64_t>(sizeof(T));
        int64_t srcGapBytes = (originalA0_ - curA0Len) * static_cast<int64_t>(sizeof(T));
        bool usePadding = (rightPadElems > 0);

        DataCopyParams copyInParams;
        copyInParams.blockCount = static_cast<uint32_t>(rCount_);
        copyInParams.blockLen = static_cast<uint32_t>(blockLenBytes);
        copyInParams.srcStride = static_cast<uint32_t>(srcGapBytes);
        copyInParams.dstStride = static_cast<uint32_t>(dstGapBytes / 32);

        int64_t gmOffset = a1Idx * rCount_ * originalA0_ + a0Start;
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

    uint32_t totalCount = static_cast<uint32_t>(rCount_ * alignedCols_);

    // 清零 mask
    int64_t totalMaskElements = rCount_ * alignedCols_;
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
    LocalTensor<float> dstFloat;
    if constexpr (IS_FP32) {
        dstFloat = yLocal.template ReinterpretCast<float>();
    } else {
        dstFloat = castBuf.Get<float>();
    }

    uint32_t srcShape[2] = {static_cast<uint32_t>(rCount_), static_cast<uint32_t>(alignedCols_)};
    LocalTensor<uint8_t> tmpUint8 = tmpBuf.Get<uint8_t>();
    ReduceSum<float, AscendC::Pattern::Reduce::RA>(dstFloat, cleanLocal, tmpUint8, srcShape, true);






    if constexpr (!IS_FP32) {
        LocalTensor<T> yTyped = yLocal;
        Cast(yTyped, dstFloat, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedCols_));


    }

    // ========== CopyOut ==========
    outQueueY.EnQue(yLocal);
    yLocal = outQueueY.template DeQue<T>();

    DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(curA0Len * static_cast<int64_t>(sizeof(T)));
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int64_t outGmOffset = a1Idx * a0Count_ + a0Start;
    DataCopyPad(outputGM[outGmOffset], yLocal, copyOutParams);

    outQueueY.FreeTensor(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ReduceNansumAraFullload<T>::Process()
{
    for (int64_t t = 0; t < tileCount_; t++) {
        int64_t globalTileIdx = startTile_ + t;
        int64_t a1Idx = globalTileIdx / a0Outer_;
        int64_t a0OuterIdx = globalTileIdx % a0Outer_;
        ProcessOneTile(a1Idx, a0OuterIdx);
    }
}

} // namespace NsReduceNansum

