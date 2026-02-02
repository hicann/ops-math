/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file acos.h
 * \brief
 */

#ifndef ACOS_H
#define ACOS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acos_tiling_data.h"
#include "acos_tiling_key.h"
namespace NsAcos {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_TO_BIT = 8;
template <typename T>
class Acos {
public:
    __aicore__ inline Acos(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AcosTilingData* tilingData);
    __aicore__ inline void Process();

    inline static constexpr T taylorCoefficients[] = {1.0,         1.0 / 6,     3.0 / 40,      5.0 / 112,
                                                      35.0 / 1152, 63.0 / 2816, 231.0 / 13312, 143.0 / 10240};
    inline static constexpr T halfPi = 1.5707963267948996192313216916398;
    inline static constexpr T Boudry = 0.70710678118654752440084436210485;

private:
    __aicore__ inline void CopyIn(uint64_t progress, uint64_t tileLength);
    __aicore__ inline void CopyOut(uint64_t progress, uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t progress, uint64_t tileLength);
    __aicore__ inline void ComputeArcSin(LocalTensor<T> yLocal, LocalTensor<T> xLocal, uint64_t tileLength);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;   // 占用BUFFER_NUM个tileBufferLen_内存
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY; // 占用BUFFER_NUM个tileBufferLen_内存
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    TBuf<TPosition::VECCALC> outputTempBuf;   // 占用1个tileBufferLen_内存
    TBuf<TPosition::VECCALC> xPowTempBuf;     // 占用1个tileBufferLen_内存
    TBuf<TPosition::VECCALC> calcTempBuf;     // 占用1个tileBufferLen_内存
    TBuf<TPosition::VECCALC> xBoudryMarkMask; // 占用1/8个tileBufferLen_内存
    TBuf<TPosition::VECCALC> xSignMask;       // 占用1/8个tileBufferLen_内存

    uint64_t loopCount_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t tileBufferLen_ = 0;
    uint64_t tailTileLen_ = 0;
};

template <typename T>
__aicore__ inline void Acos<T>::Init(GM_ADDR x, GM_ADDR y, const AcosTilingData* tilingData)
{
    auto blockIdx_ = GetBlockIdx();
    uint64_t offset;
    if (blockIdx_ >= tilingData->formerCoreNum) {
        blockLength_ = tilingData->tailCoreDataNum;
        loopCount_ = tilingData->tailCoreLoopCount;
        tileBufferLen_ = tilingData->tailCoreFormerDataNum;
        tailTileLen_ = tilingData->tailCoreTailDataNum;
        offset = tilingData->formerCoreNum * tilingData->formerCoreDataNum +
                 (blockIdx_ - tilingData->formerCoreNum) * blockLength_;
    } else {
        blockLength_ = tilingData->formerCoreDataNum;
        loopCount_ = tilingData->formerCoreLoopCount;
        tileBufferLen_ = tilingData->formerCoreFormerDataNum;
        tailTileLen_ = tilingData->formerCoreTailDataNum;
        offset = blockLength_ * blockIdx_;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(outputTempBuf, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(xPowTempBuf, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(calcTempBuf, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(xSignMask, (tileBufferLen_ + BYTE_TO_BIT - 1) / BYTE_TO_BIT);
    pipe.InitBuffer(xBoudryMarkMask, (tileBufferLen_ + BYTE_TO_BIT - 1) / BYTE_TO_BIT);
}

template <typename T>
__aicore__ inline void Acos<T>::CopyIn(uint64_t progress, uint64_t tileLength)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tileLength * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * tileBufferLen_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Acos<T>::CopyOut(uint64_t progress, uint64_t tileLength)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tileLength * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * tileBufferLen_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Acos<T>::ComputeArcSin(LocalTensor<T> yLocal, LocalTensor<T> xLocal, uint64_t tileLength)
{
    /* 按照泰勒公式计算arcsin(x) = C*x + C*x^3 + C*x^5 + C*x^7 + C*x^9 + C*x^11 + C*x^13 + C*x^15 .....
       其中C为泰勒系数，参考常量taylorCoefficients */
    LocalTensor<T> xPowTempTensor = xPowTempBuf.Get<T>(); // 保存x幂的计算结果
    LocalTensor<T> calcTempTensor = calcTempBuf.Get<T>(); // 保存每个泰勒展开项结果
    DataCopy(xPowTempTensor, xLocal, tileLength);
    DataCopy(yLocal, xLocal, tileLength);    // C0 = 1直接用拷贝效率更高
    Mul(xLocal, xLocal, xLocal, tileLength); // 注意xLocal平方后保存下来给后续使用
    for (auto i = 1; i < sizeof(taylorCoefficients) / sizeof(T); i++) {
        Mul(xPowTempTensor, xPowTempTensor, xLocal, tileLength);
        Muls(calcTempTensor, xPowTempTensor, (T)taylorCoefficients[i], tileLength);
        Add(yLocal, yLocal, calcTempTensor, tileLength);
    }
}

template <typename T>
__aicore__ inline void Acos<T>::Compute(uint64_t progress, uint64_t tileLength)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> outputTempTensor = outputTempBuf.Get<T>();
    LocalTensor<uint8_t> xSign = xSignMask.Get<uint8_t>();
    LocalTensor<uint8_t> xBoudryMark = xBoudryMarkMask.Get<uint8_t>();

    // 提取正负值标记，小于0标记符号位，然后取绝对值进行计算
    CompareScalar(xSign, xLocal, (T)0.0, AscendC::CMPMODE::LT, tileLength);
    Abs(xLocal, xLocal, tileLength);

    // 计算边界值标记掩码
    CompareScalar(xBoudryMark, xLocal, (T)Boudry, AscendC::CMPMODE::LT, tileLength);

    // 小于Boudry的按照泰勒展开公式直接计算arcsin(x)放到yLocal中
    ComputeArcSin(yLocal, xLocal, tileLength);

    // 大于Boudry按照arcsin(x) = halfPi - 1/2 sqrt(1-x^2)计算放到outputTempTensor中
    Muls(xLocal, xLocal, (T)(-1.0), tileLength); // ComputeArcSin里面已经对xLocal平方保存
    Adds(xLocal, xLocal, (T)1.0, tileLength);
    Sqrt(xLocal, xLocal, tileLength);
    ComputeArcSin(outputTempTensor, xLocal, tileLength);
    Muls(outputTempTensor, outputTempTensor, (T)(-1.0), tileLength);
    Adds(outputTempTensor, outputTempTensor, (T)halfPi, tileLength);

    // 根据边界值掩码小于Boudry取yLocal计算结果，大于Boudry取outputTempTensor结果
    Select(yLocal, xBoudryMark, yLocal, outputTempTensor, SELMODE::VSEL_CMPMASK_SPR, tileLength);

    // arcsin(-x) = -arcsin(x), 根据符号标记合并还原计算负值情况
    Muls(outputTempTensor, yLocal, (T)(-1.0), tileLength);
    Select(yLocal, xSign, outputTempTensor, yLocal, SELMODE::VSEL_CMPMASK_SPR, tileLength);

    // arccos(x) = halfPi - arcsin(x)
    Muls(yLocal, yLocal, (T)(-1.0), tileLength);
    Adds(yLocal, yLocal, (T)halfPi, tileLength);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Acos<T>::Process()
{
    for (auto i = 0; i < loopCount_ - 1; i++) {
        CopyIn(i, tileBufferLen_);
        Compute(i, tileBufferLen_);
        CopyOut(i, tileBufferLen_);
    }
    CopyIn(loopCount_ - 1, tailTileLen_);
    Compute(loopCount_ - 1, tailTileLen_);
    CopyOut(loopCount_ - 1, tailTileLen_);
}

} // namespace NsAcos
#endif // ACOS_H