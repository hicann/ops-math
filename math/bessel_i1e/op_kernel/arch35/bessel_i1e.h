/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BESSEL_I1E_H
#define BESSEL_I1E_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bessel_i1e_tiling_data.h"
#include <type_traits>

namespace NsBesselI1e {

using namespace AscendC;

constexpr float SEGMENT_POINT = 3.75f;
constexpr float INV_SEGMENT = 0.26666666666666666f;

constexpr float itrBefore[7] = {0.5000000008f, 0.8789061535f, 0.5149860539f, 0.1508606731f,
                                0.0265652742f, 0.0030351394f, 0.0003173337f};

constexpr float itrAfter[9] = {0.3989422302f, -0.0398905760f, -0.0034090932f, 0.0000697438f, -0.0044962120f,
                               0.0108902378f, -0.0151944387f, 0.0095376700f,  -0.0021325862f};

template <typename T>
class BesselI1e {
    static constexpr int BUFFER_NUM = 2;
    static constexpr bool NEED_CAST = !std::is_same_v<T, float>;
    static constexpr int64_t CMP_ALIGN = 64;
    static constexpr int64_t MASK_ELEM_PER_FLOAT = 8;
    static constexpr int64_t MASK_ALIGN = 32;

public:
    __aicore__ inline BesselI1e() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BesselI1eTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(LocalTensor<float> xInput, LocalTensor<float> yLocal, int64_t count);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void ProcessFp32(int64_t loopCount);
    __aicore__ inline void ProcessFp16Bf16(int64_t loopCount);

    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    TBuf<TPosition::VECCALC> absBuf_;
    TBuf<TPosition::VECCALC> tBuf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> polyBuf_;
    TBuf<TPosition::VECCALC> scratchBuf_;
    TBuf<TPosition::VECCALC> smallBuf_;
    TBuf<TPosition::VECCALC> largeBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;

    TBuf<TPosition::VECCALC> initBuf_;
    TBuf<TPosition::VECCALC> castInBuf_;
    TBuf<TPosition::VECCALC> resultFp32Buf_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignedUbLength_ = 0;
};

template <typename T>
__aicore__ inline void BesselI1e<T>::Init(GM_ADDR x, GM_ADDR y, const BesselI1eTilingData* tilingData)
{
    int64_t remainder = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
    ubLength_ = tilingData->ubFactor;

    alignedUbLength_ = ((ubLength_ + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;

    inputGM_.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    outputGM_.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * GetBlockIdx(), blockLength_);

    if constexpr (NEED_CAST) {
        // The double-buffered input queue is only used by the FP16/BF16 path
        // (ProcessFp16Bf16); the FP32 path streams via initBuf_ instead.
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, alignedUbLength_ * sizeof(T));
    }
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, alignedUbLength_ * sizeof(T));

    pipe_.InitBuffer(absBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(tBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(t2Buf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(polyBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(scratchBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(smallBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(largeBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(tmpBuf_, alignedUbLength_ * sizeof(float));
    pipe_.InitBuffer(maskBuf_, ((alignedUbLength_ / MASK_ELEM_PER_FLOAT + MASK_ALIGN - 1) / MASK_ALIGN) * MASK_ALIGN);

    if constexpr (std::is_same_v<T, float>) {
        pipe_.InitBuffer(initBuf_, alignedUbLength_ * sizeof(float));
    }

    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(castInBuf_, alignedUbLength_ * sizeof(float));
        pipe_.InitBuffer(resultFp32Buf_, alignedUbLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void BesselI1e<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue_.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGM_[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void BesselI1e<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue_.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM_[progress * ubLength_], yLocal, copyParams);
    outputQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void BesselI1e<T>::Compute(LocalTensor<float> xInput, LocalTensor<float> yLocal, int64_t count)
{
    LocalTensor<float> absX = absBuf_.Get<float>();
    LocalTensor<float> t = tBuf_.Get<float>();
    LocalTensor<float> t2 = t2Buf_.Get<float>();
    LocalTensor<float> poly = polyBuf_.Get<float>();
    LocalTensor<float> scratch = scratchBuf_.Get<float>();
    LocalTensor<float> smallResult = smallBuf_.Get<float>();
    LocalTensor<float> largeResult = largeBuf_.Get<float>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    int64_t n = ((count + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;

    LocalTensor<float>& result = yLocal;

    Abs(absX, xInput, n);

    Duplicate(t, INV_SEGMENT, n);
    Mul(t, absX, t, n);
    Mul(t2, t, t, n);

    Duplicate(poly, itrBefore[6], n);
    Mul(scratch, poly, t2, n);
    Adds(poly, scratch, itrBefore[5], n);
    Mul(scratch, poly, t2, n);
    Adds(poly, scratch, itrBefore[4], n);
    Mul(scratch, poly, t2, n);
    Adds(poly, scratch, itrBefore[3], n);
    Mul(scratch, poly, t2, n);
    Adds(poly, scratch, itrBefore[2], n);
    Mul(scratch, poly, t2, n);
    Adds(poly, scratch, itrBefore[1], n);
    Mul(scratch, poly, t2, n);
    Adds(poly, scratch, itrBefore[0], n);

    Muls(tmp, absX, -1.0f, n);
    Exp(tmp, tmp, n);

    Mul(smallResult, absX, poly, n);
    Mul(smallResult, smallResult, tmp, n);

    Duplicate(scratch, SEGMENT_POINT, n);
    Div(t, scratch, absX, n);

    Duplicate(poly, itrAfter[8], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[7], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[6], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[5], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[4], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[3], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[2], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[1], n);
    Mul(scratch, poly, t, n);
    Adds(poly, scratch, itrAfter[0], n);

    Sqrt(tmp, absX, n);
    Div(largeResult, poly, tmp, n);

    Duplicate(scratch, SEGMENT_POINT, n);
    Compare(mask, absX, scratch, CMPMODE::LT, n);
    Select(result, mask, smallResult, largeResult, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

    Duplicate(scratch, 0.0f, n);
    Compare(mask, xInput, scratch, CMPMODE::LT, n);
    Muls(tmp, result, -1.0f, n);
    Select(result, mask, tmp, result, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
}

template <typename T>
__aicore__ inline void BesselI1e<T>::ProcessFp32(int64_t loopCount)
{
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        int64_t alignedCount = ((currentNum + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;

        LocalTensor<float> xFp32 = initBuf_.Get<float>();
        Duplicate(xFp32, 0.0f, alignedCount);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(xFp32, inputGM_[i * ubLength_], copyParams, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        LocalTensor<float> yLocal = outputQueue_.template AllocTensor<float>();
        Compute(xFp32, yLocal, currentNum);
        outputQueue_.template EnQue<float>(yLocal);

        CopyOut(i, currentNum);
    }
}

template <typename T>
__aicore__ inline void BesselI1e<T>::ProcessFp16Bf16(int64_t loopCount)
{
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);

        LocalTensor<T> xInput = inputQueue_.template DeQue<T>();

        LocalTensor<float> xFp32 = castInBuf_.Get<float>();
        int64_t alignedCount = ((currentNum + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN;
        Duplicate(xFp32, 0.0f, alignedCount);
        Cast<float, T>(xFp32, xInput, RoundMode::CAST_NONE, currentNum);

        LocalTensor<float> resultFp32 = resultFp32Buf_.Get<float>();
        Compute(xFp32, resultFp32, currentNum);

        LocalTensor<T> yOutput = outputQueue_.template AllocTensor<T>();
        Cast<T, float>(yOutput, resultFp32, RoundMode::CAST_ROUND, currentNum);
        outputQueue_.template EnQue<T>(yOutput);

        inputQueue_.FreeTensor(xInput);
        CopyOut(i, currentNum);
    }
}

template <typename T>
__aicore__ inline void BesselI1e<T>::Process()
{
    if (ubLength_ == 0 || blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    if constexpr (std::is_same_v<T, float>) {
        ProcessFp32(loopCount);
    } else {
        ProcessFp16Bf16(loopCount);
    }
}

} // namespace NsBesselI1e
#endif // BESSEL_I1E_H
