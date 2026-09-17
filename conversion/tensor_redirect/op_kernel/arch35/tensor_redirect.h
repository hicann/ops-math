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
 * \file tensor_redirect.h
 * \brief tensor_redirect kernel: identity copy GM -> UB -> GM
 */

#ifndef TENSOR_REDIRECT_H
#define TENSOR_REDIRECT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tensor_redirect_tiling_data.h"

namespace NsTensorRedirect {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2; // double buffer

template <typename T>
class TensorRedirectKernel {
public:
    __aicore__ inline TensorRedirectKernel(){};
    __aicore__ inline bool Init(GM_ADDR x, GM_ADDR outputX, const TensorRedirectTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const TensorRedirectTilingData* tilingData);
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);

private:
    // 输入输出共用同一块 UB（无 Compute 阶段，输入即输出），省去一次 UB->UB 拷贝。
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailBlockTailUbFactor_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    uint32_t bufferSize_ = 0;
    TPipe* pipe_ = nullptr;
};

template <typename T>
__aicore__ inline void TensorRedirectKernel<T>::ParseTilingData(const TensorRedirectTilingData* tilingData)
{
    usedCoreNum_ = tilingData->usedCoreNum;
    ubFactor_ = tilingData->ubFactor;
    tailBlockTailUbFactor_ = tilingData->tailBlockTailUbFactor;
    blockFactor_ = tilingData->blockFactor;
    tailBlockFactor_ = tilingData->tailBlockFactor;
}

template <typename T>
__aicore__ inline bool TensorRedirectKernel<T>::Init(GM_ADDR x, GM_ADDR outputX,
                                                     const TensorRedirectTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    ParseTilingData(tilingData);

    // host 保证分区起点小于 numel 且 numel * sizeof(T) <= INT64_MAX，以下乘法及 GM 字节偏移均可表示。
    blockOffset_ = blockIdx_ * blockFactor_ * ubFactor_;
    int64_t coreElementCount = blockFactor_ * ubFactor_;
    if (blockIdx_ == usedCoreNum_ - 1) {
        coreElementCount = (tailBlockFactor_ - 1) * ubFactor_ + tailBlockTailUbFactor_;
    }
    xGm_.SetGlobalBuffer((__gm__ T*)(x) + blockOffset_, static_cast<uint64_t>(coreElementCount));
    yGm_.SetGlobalBuffer((__gm__ T*)(outputX) + blockOffset_, static_cast<uint64_t>(coreElementCount));

    // host tiling 已保证 ubFactor * sizeof(T) <= DataCopyPad::blockLen 上限；
    // 保持与 InitBuffer 长度类型一致，避免隐式窄化。
    bufferSize_ = static_cast<uint32_t>(ubFactor_) * static_cast<uint32_t>(sizeof(T));
    return pipe_->InitBuffer(dataQueue_, DB_BUFFER, bufferSize_);
}

template <typename T>
__aicore__ inline void TensorRedirectKernel<T>::CopyIn(int64_t offset, int64_t dataLen)
{
    // blockCount=1；blockLen 单位=字节；srcStride/dstStride=0（单块连续，无间隔）
    const uint32_t dataBytes = static_cast<uint32_t>(dataLen) * static_cast<uint32_t>(sizeof(T));
    DataCopyExtParams inParams{static_cast<uint16_t>(1), dataBytes, static_cast<int64_t>(0), static_cast<int64_t>(0),
                               static_cast<uint32_t>(0)};
    // isPad=false：不使用 paddingValue；尾块由硬件自动 dummy 补齐到 32B（dummy 不写回 GM）
    DataCopyPadExtParams<T> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, xGm_[offset], inParams, padParams); // GM -> UB
    dataQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void TensorRedirectKernel<T>::CopyOut(int64_t offset, int64_t dataLen)
{
    const uint32_t dataBytes = static_cast<uint32_t>(dataLen) * static_cast<uint32_t>(sizeof(T));
    DataCopyExtParams outParams{static_cast<uint16_t>(1), dataBytes, static_cast<int64_t>(0), static_cast<int64_t>(0),
                                static_cast<uint32_t>(0)};
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, outParams); // UB -> GM
    dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TensorRedirectKernel<T>::Process()
{
    // 注意：offset 为核内相对偏移；跨核偏移已在 Init() 中折入 xGm_/yGm_ 基址，此处不再叠加。
    if (blockIdx_ >= usedCoreNum_) {
        return; // 冗余核直接退出，不触碰任何 GM
    }
    int64_t loopSize = (blockIdx_ == usedCoreNum_ - 1) ? tailBlockFactor_ : blockFactor_;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) { // 满块循环
        // idx < blockFactor <= uo，故 offset < numel，乘法不会越过 int64_t 上界。
        int64_t offset = idx * ubFactor_;
        CopyIn(offset, ubFactor_);
        CopyOut(offset, ubFactor_);
    }
    // 最后一块（尾核时可能是非整除尾块）
    int64_t offset = (loopSize - 1) * ubFactor_;
    int64_t dataLen = (blockIdx_ == usedCoreNum_ - 1) ? tailBlockTailUbFactor_ : ubFactor_;
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}

} // namespace NsTensorRedirect

#endif // TENSOR_REDIRECT_H
