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
 * \file reduce_nansum_ar_fullload.h
 * \brief ReduceNansum AR 全载 Kernel 实现（TilingKey=0）
 *
 * AR 模板（A0=1）全载模式：每行 R 个元素全部载入 UB 后执行
 * IsNan → Select → ReduceSum 流程。
 *
 * 迭代三：支持 fp32/fp16/bf16 混合精度。
 * 非连续多轴归约：逐块处理，累加每块的 ReduceSum 结果。
 */
#ifndef REDUCE_NANSUM_AR_FULLLOAD_H
#define REDUCE_NANSUM_AR_FULLLOAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_nansum_tiling_data.h"
#include "reduce_nansum_tiling_key.h"

namespace NsReduceNansum {

using namespace AscendC;

template <typename T>
class ReduceNansumArFullload {
public:
    __aicore__ inline ReduceNansumArFullload() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReduceNansumTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t rowIdx);
    __aicore__ inline void Compute(int64_t rowIdx);
    __aicore__ inline void CopyOut(int64_t rowIdx);
    __aicore__ inline void ProcessStridedRow(int64_t globalRowIdx);

private:
    static constexpr bool IS_FP32 = sizeof(T) == sizeof(float);
    static constexpr bool CAN_COMPARE_DIRECTLY = IS_FP32;

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECOUT, 2> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> cleanBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> castBuf;  // fp16/bf16 -> fp32 转换

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t a1Count_ = 0;
    int64_t rCount_ = 0;
    int64_t rLengthAlign_ = 0;
    int64_t tmpBufSize_ = 0;

    // 多核参数
    int64_t startRow_ = 0;
    int64_t rowCount_ = 0;

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
    int64_t outerReduceDimCount_ = 0;  // 外层归约维度数
};

template <typename T>
__aicore__ inline void ReduceNansumArFullload<T>::Init(GM_ADDR x, GM_ADDR y,
                                                        const ReduceNansumTilingData* tilingData)
{
    a1Count_ = tilingData->a1Count;
    rCount_ = tilingData->rCount;
    rLengthAlign_ = tilingData->rLengthAlign;
    tmpBufSize_ = tilingData->tmpBufSize;

    // 多核切分：计算当前核处理的行范围
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t tilesPerCore = tilingData->tilesPerCore;
    int64_t tailCoreTiles = tilingData->tailCoreTiles;
    int64_t usedCoreNum = tilingData->usedCoreNum;

    startRow_ = blockIdx * tilesPerCore;
    if (blockIdx < usedCoreNum - 1) {
        rowCount_ = tilesPerCore;
    } else {
        rowCount_ = tailCoreTiles;
    }

    // 读取非连续多轴归约参数
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
    // 确定外层归约维度数量
    outerReduceDimCount_ = reduceDimCount_;
    if (copyBlockCount_ > 0) {
        int64_t innerBlockElems = copyBlockLen_ / static_cast<int64_t>(sizeof(T));
        int64_t innerProduct = 1;
        for (int64_t d = reduceDimCount_ - 1; d >= 0; d--) {
            innerProduct *= reduceDimSizes_[d];
            if (innerProduct == innerBlockElems) {
                outerReduceDimCount_ = d;
                break;
            }
        }
    }

    // 设置 GM 指针
    int64_t inputGmSize = a1Count_ * rCount_;
    if (copyBlockCount_ > 0) {
        // 非连续多轴归约：计算 GM 最远元素位置
        int64_t maxGmOffset = 0;
        for (int64_t d = 0; d < nonReduceDimCount_; d++) {
            maxGmOffset += (nonReduceDimSizes_[d] - 1) * nonReduceGmStrides_[d];
        }
        for (int64_t d = 0; d < reduceDimCount_; d++) {
            maxGmOffset += (reduceDimSizes_[d] - 1) * reduceGmStrides_[d];
        }
        maxGmOffset += 1;  // A0=1 for AR template
        inputGmSize = maxGmOffset;
    }
    inputGM.SetGlobalBuffer((__gm__ T*)x, inputGmSize);
    outputGM.SetGlobalBuffer((__gm__ T*)y, a1Count_);

    // 非连续多轴归约：UB buffer 按 innerBlockSize 对齐分配（每次只搬一个块）
    // 连续模式：按 rLengthAlign_ 分配
    int64_t ubBlockElems = rLengthAlign_;
    if (copyBlockCount_ > 0) {
        // 每次只搬一个 block, 大小 = innerBlockSize 对齐到 256 字节
        int64_t innerBlockElems = copyBlockLen_ / static_cast<int64_t>(sizeof(T));
        int64_t computeTypeSize = static_cast<int64_t>(sizeof(float));
        // 对齐到 256 字节（Compare API 要求）
        int64_t alignedElems = ((innerBlockElems * computeTypeSize + 255) / 256) * 256 / computeTypeSize;
        if (alignedElems < 64) alignedElems = 64;  // 最小 64 个 fp32 元素
        ubBlockElems = alignedElems;
    }

    // 初始化 Buffer
    pipe.InitBuffer(inQueueX, 2, ubBlockElems * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outQueueY, 2, 32);

    // maskBuf: Compare 输出，bit-packed uint8_t 类型
    int64_t maskBufSize = ((ubBlockElems / 8 + 31) / 32) * 32;
    if (maskBufSize < 32) {
        maskBufSize = 32;
    }
    pipe.InitBuffer(maskBuf, maskBufSize);
    pipe.InitBuffer(zeroBuf, ubBlockElems * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(cleanBuf, ubBlockElems * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(tmpBuf, tmpBufSize_);

    if constexpr (!IS_FP32) {
        pipe.InitBuffer(castBuf, ubBlockElems * static_cast<int64_t>(sizeof(float)));
    }
}

template <typename T>
__aicore__ inline void ReduceNansumArFullload<T>::CopyIn(int64_t rowIdx)
{
    int64_t globalRowIdx = startRow_ + rowIdx;
    LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

    // 清零整个 buffer（确保 padding 区域为 0）
    if constexpr (IS_FP32) {
        LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
        Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(rLengthAlign_));
    } else {
        Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(rLengthAlign_));
    }



#endif



    DataCopyParams copyParams;
    // 连续行：标准 copy（非连续走 ProcessStridedRow）
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(rCount_ * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGM[globalRowIdx * rCount_], copyParams, {false, 0, 0, 0});

    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ReduceNansumArFullload<T>::Compute(int64_t rowIdx)
{
    LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();

    // 获取工作 buffer
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    LocalTensor<float> cleanLocal = cleanBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    uint32_t count = static_cast<uint32_t>(rLengthAlign_);

    // Step 0: 清零 mask buffer
    int64_t maskInt16Count = ((rLengthAlign_ / 8 + 31) / 32) * 32 / 2;
    if (maskInt16Count < 16) maskInt16Count = 16;
    LocalTensor<int16_t> maskInt16 = maskBuf.Get<int16_t>();
    Duplicate(maskInt16, static_cast<int16_t>(0), static_cast<uint32_t>(maskInt16Count));

    // Step 1: 填充零向量（fp32）
    Duplicate(zeroLocal, static_cast<float>(0.0f), count);

    if constexpr (IS_FP32) {
        LocalTensor<float> xFp32 = xLocal.template ReinterpretCast<float>();
        Compare(maskLocal, xFp32, xFp32, CMPMODE::EQ, count);
        Select(cleanLocal, maskLocal, xFp32, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        LocalTensor<float> dstLocal = yLocal.template ReinterpretCast<float>();
        ReduceSum(dstLocal, cleanLocal, tmpLocal, static_cast<uint32_t>(rCount_));
    } else {
        LocalTensor<float> castLocal = castBuf.Get<float>();
        Cast(castLocal, xLocal, RoundMode::CAST_NONE, count);
        Compare(maskLocal, castLocal, castLocal, CMPMODE::EQ, count);
        Select(cleanLocal, maskLocal, castLocal, zeroLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        LocalTensor<float> reduceDst = zeroBuf.Get<float>();
        ReduceSum(reduceDst, cleanLocal, tmpLocal, static_cast<uint32_t>(rCount_));
        LocalTensor<T> yTyped = yLocal;
        Cast(yTyped, reduceDst, RoundMode::CAST_ROUND, 8u);
    }

    outQueueY.template EnQue<T>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ReduceNansumArFullload<T>::CopyOut(int64_t rowIdx)
{
    int64_t globalRowIdx = startRow_ + rowIdx;
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM[globalRowIdx], yLocal, copyParams);

    outQueueY.FreeTensor(yLocal);
}

// 非连续多轴归约的专用处理：逐块搬运，逐块 NaN→0 归约，块间累加
template <typename T>
__aicore__ inline void ReduceNansumArFullload<T>::ProcessStridedRow(int64_t globalRowIdx)
{
    int64_t innerBlockElems = copyBlockLen_ / static_cast<int64_t>(sizeof(T));

    // 计算当前行（输出元素）对应的 GM 基地址
    int64_t gmBase = 0;
    int64_t remaining = globalRowIdx;
    for (int64_t d = nonReduceDimCount_ - 1; d >= 0; d--) {
        int64_t coord = remaining % nonReduceDimSizes_[d];
        remaining = remaining / nonReduceDimSizes_[d];
        gmBase += coord * nonReduceGmStrides_[d];
    }

    // 计算对齐后的块大小（与 Init 中一致）
    int64_t computeTypeSize = static_cast<int64_t>(sizeof(float));
    int64_t alignedElems = ((innerBlockElems * computeTypeSize + 255) / 256) * 256 / computeTypeSize;
    if (alignedElems < 64) alignedElems = 64;

    // 全局累加器（复用 outQueueY 的 buffer）
    LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
    LocalTensor<float> yFloat = yLocal.template ReinterpretCast<float>();
    constexpr uint32_t SCALAR_ALIGN = 8;
    Duplicate(yFloat, static_cast<float>(0.0f), SCALAR_ALIGN);



    for (int64_t blk = 0; blk < copyBlockCount_; blk++) {
        // 通过外层归约维度坐标计算正确的 GM 偏移
        int64_t gmReduceOffset = 0;
        int64_t blkRemaining = blk;
        for (int64_t d = outerReduceDimCount_ - 1; d >= 0; d--) {
            int64_t coord = blkRemaining % reduceDimSizes_[d];
            blkRemaining = blkRemaining / reduceDimSizes_[d];
            gmReduceOffset += coord * reduceGmStrides_[d];
        }

        // CopyIn: 搬一个块
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
        if constexpr (IS_FP32) {
            LocalTensor<float> xFloat = xLocal.template ReinterpretCast<float>();
            Duplicate(xFloat, static_cast<float>(0.0f), static_cast<uint32_t>(alignedElems));
        } else {
            Duplicate(xLocal, static_cast<T>(0), static_cast<uint32_t>(alignedElems));
        }



        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(copyBlockLen_);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(xLocal, inputGM[gmBase + gmReduceOffset], copyParams, {false, 0, 0, 0});

        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.template DeQue<T>();

        // Compute: NaN 检测 + Select + ReduceSum（块内）
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
        LocalTensor<float> cleanLocal = cleanBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        uint32_t count = static_cast<uint32_t>(alignedElems);

        // 清零 mask
        int64_t maskInt16Count = ((alignedElems / 8 + 31) / 32) * 32 / 2;
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

        // ReduceSum 块内结果到 zeroBuf 前 8 个元素（临时复用）
        LocalTensor<float> blockResult = zeroBuf.Get<float>();
        ReduceSum(blockResult, cleanLocal, tmpLocal, static_cast<uint32_t>(innerBlockElems));



        // 累加到全局结果
        Add(yFloat, yFloat, blockResult, SCALAR_ALIGN);



        inQueueX.FreeTensor(xLocal);
    }

    // Cast 回原始类型（如需）
    if constexpr (!IS_FP32) {
        LocalTensor<float> tmpResult = zeroBuf.Get<float>();
        Duplicate(tmpResult, static_cast<float>(0.0f), SCALAR_ALIGN);


        DataCopy(tmpResult, yFloat, SCALAR_ALIGN);


        LocalTensor<T> yTyped = yLocal;
        Cast(yTyped, tmpResult, RoundMode::CAST_ROUND, SCALAR_ALIGN);


    }

    // CopyOut
    outQueueY.template EnQue<T>(yLocal);
    yLocal = outQueueY.template DeQue<T>();

    DataCopyParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = static_cast<uint32_t>(sizeof(T));
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(outputGM[globalRowIdx], yLocal, outParams);

    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ReduceNansumArFullload<T>::Process()
{
    if (copyBlockCount_ > 0) {
        // 非连续多轴归约：逐行逐块处理
        for (int64_t i = 0; i < rowCount_; i++) {
            ProcessStridedRow(startRow_ + i);
        }
    } else {
        // 连续模式：标准 CopyIn → Compute → CopyOut
        for (int64_t i = 0; i < rowCount_; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
}

} // namespace NsReduceNansum

