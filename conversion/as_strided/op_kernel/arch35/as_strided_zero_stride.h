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
 * \file as_strided_zero_stride.h
 * \brief as_strided_zero_strides
 */

#ifndef STRIDED_IS_ZERO_H
#define STRIDED_IS_ZERO_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "as_strided_struct.h"

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T>
__aicore__ inline T FloorDiv(T x, T y) {
  return y == 0 ? x : x / y;
}

namespace AsStrided {
using namespace AscendC;
constexpr int64_t BUF_NUM = 2;
constexpr int64_t ALIGN_NUM = 32;
template<typename T>
class StridedIsZero {
public:
    __aicore__ inline StridedIsZero() {};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, 
                                const AsStridedZeroStrideTilingData* tiling);
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Process();

private:
    const AsStridedZeroStrideTilingData* tilingData_;
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xInQue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> tempInQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> offsetGm_;
    int64_t curCoreBaseIndex_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t blockIdx_ = 0;
    int64_t halfUbMaxElements_ = 0;
    int64_t curCoreLoops_ = 0;
    int64_t curCorePerLoopElements_ = 0;
    int64_t curCoreLastLoopElements_ = 0;
    T inX_ = 0;
};

template <typename T>
__aicore__ inline void StridedIsZero<T>::Init(GM_ADDR input, GM_ADDR output, 
                                                const AsStridedZeroStrideTilingData* tiling)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tiling;
    int64_t halfUb_ = (tilingData_->ubSizePlatForm - sizeof(T)) / 2;
    halfUb_ = FloorDiv(halfUb_, ALIGN_NUM) * ALIGN_NUM;
    int64_t blkFactor_ = tilingData_->mainBlockFactor;
    int64_t blkTailFactor_ = tilingData_->tailBlockFactor;
    int64_t usedCoreCnt_ = tilingData_->blockNum;
    curCoreElements_ = blkFactor_;
    if(blockIdx_ == usedCoreCnt_ - 1) {
        curCoreElements_ = blkTailFactor_;
    }
    curCoreBaseIndex_ = blockIdx_ * blkFactor_;
    halfUbMaxElements_ = halfUb_ / sizeof(T);
    xGm_.SetGlobalBuffer((__gm__ T*)input);
    yGm_.SetGlobalBuffer((__gm__ T*)output);
    pipe_.InitBuffer(xInQue_, BUF_NUM, halfUbMaxElements_ * sizeof(T));
    pipe_.InitBuffer(tempInQue_, BUF_NUM, 1 * sizeof(T));

    // 计算循环参数
    curCoreLoops_ = CeilDiv(curCoreElements_, halfUbMaxElements_);
    curCorePerLoopElements_ = halfUbMaxElements_;
    curCoreLastLoopElements_ = curCoreElements_ - (curCoreLoops_ - 1) * curCorePerLoopElements_;
}

template <typename T>
__aicore__ inline void StridedIsZero<T>::CopyIn()
{
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyExtParams dataCopyInParams{1, static_cast<uint32_t>(1 * sizeof(T)), 0, 0, 0};
    LocalTensor<T> tempTensor = tempInQue_.AllocTensor<T>();
    DataCopyPad(tempTensor, xGm_[tilingData_->storageOffset], dataCopyInParams, dataCopyPadParams);
    tempInQue_.EnQue<T>(tempTensor);
}

template <typename T>
__aicore__ inline void StridedIsZero<T>::Compute()
{
    LocalTensor<T> tempInTensor = tempInQue_.DeQue<T>();
    LocalTensor<T> xTensor = xInQue_.AllocTensor<T>();
    AscendC::Duplicate<T>(xTensor, tempInTensor[0], curCorePerLoopElements_);
    xInQue_.EnQue<T>(xTensor);
    tempInQue_.FreeTensor<T>(tempInTensor);
}

template <typename T>
__aicore__ inline void StridedIsZero<T>::CopyOut()
{
    LocalTensor<T> xDequeTensor = xInQue_.DeQue<T>();
    DataCopyExtParams dataCopyOutMainParams{1, static_cast<uint32_t>(curCorePerLoopElements_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams dataCopyOutLastParams{1, static_cast<uint32_t>(curCoreLastLoopElements_ * sizeof(T)), 0, 0, 0};
    for(int64_t loop = 0; loop < curCoreLoops_; loop++) {
        if(loop == curCoreLoops_ - 1) {
            DataCopyPad(yGm_[curCoreBaseIndex_ + loop * curCorePerLoopElements_], xDequeTensor, dataCopyOutLastParams);
        } else {
            DataCopyPad(yGm_[curCoreBaseIndex_ + loop * curCorePerLoopElements_], xDequeTensor, dataCopyOutMainParams);
        }
    }
    xInQue_.FreeTensor<T>(xDequeTensor);
}

template <typename T>
__aicore__ inline void StridedIsZero<T>::Process()
{
    CopyIn();
    int32_t eventIdMTE22V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE22V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE22V);
    Compute();
    int32_t eventIdV2MTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
    CopyOut();
}

} // namespace AsStrided
#endif