/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace RandomStandardNormalV2 {
using namespace AscendC;

constexpr uint16_t BUFFER_NUM = 2;
constexpr uint16_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
constexpr uint16_t KEY_SIZE = 2;
constexpr uint16_t COUNTER_SIZE = 4;
constexpr uint16_t RESULT_ELEMENT_CNT = 4;
constexpr uint32_t INT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
constexpr uint16_t DOUBLE_UNIFORM_RESULT = 2;
constexpr float DOUBLE_MULTIPLE = 2.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr uint64_t K_Reserved_Per_Output = 256;
constexpr uint32_t RIGHT_SHIFT = 32;
constexpr uint64_t GROUP_SIZE = 2;
constexpr uint64_t USED_THREAD = 1024;
constexpr uint64_t THREAD_LAUNCH = 1024;

__aicore__ inline void BoxMullerFloat(const float x0, const float x1, float* f0, float* f1)
{
    const float eps = 1.0e-7f;
    float u1 = x0;
    if (u1 < eps) {
        u1 = eps;
    }
    float v1 = static_cast<float>(DOUBLE_MULTIPLE * PI * x1);
    float u2 = Simt::Sqrt(-DOUBLE_MULTIPLE * Simt::Log(u1));
    Simt::Sincos(v1, *f0, *f1);
    *f0 *= u2;
    *f1 *= u2;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void SimtBoxMuller(
    __ubuf__ float* yOutputTmp, const uint32_t calCount)
{
    int64_t groupOffset = Simt::GetThreadIdx() * GROUP_SIZE;

    while (groupOffset < calCount) {
        float uniformRes[GROUP_SIZE];
        BoxMullerFloat(yOutputTmp[groupOffset], yOutputTmp[groupOffset + 1], &uniformRes[0], &uniformRes[1]);

        yOutputTmp[groupOffset] = uniformRes[0];
        yOutputTmp[groupOffset + 1] = uniformRes[1];

        groupOffset += Simt::GetThreadNum() * GROUP_SIZE;
    }
}

template <typename T, typename OFFSET_T>
class RandomStandardNormalV2Op {
public:
    __aicore__ inline RandomStandardNormalV2Op(
        TPipe* pipe, const RandomStandardNormalV2TilingData4RegBase* __restrict tilingData)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR offset, GM_ADDR y, GM_ADDR count);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKeyAndCounter();
    __aicore__ inline uint32_t ROUND_UP32(const uint32_t x) const;
    __aicore__ inline void Skip(const int64_t count);
    __aicore__ inline void DataTypeHandle(LocalTensor<float>& yOutputTmp, const uint32_t calCount);
    __aicore__ inline void CopyOut();

private:
    TPipe* pipe_;
    const RandomStandardNormalV2TilingData4RegBase* tiling_;

    GlobalTensor<OFFSET_T> offsetGm_;
    GlobalTensor<T> outputGm_;
    TBuf<QuePosition::VECCALC> uniformResult_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;

    int64_t blockNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t blockIdx_ = 0;
    int64_t currUbTilingSize_ = 0;
    int64_t blockOffSet_ = 0;
    int64_t curOffSet_ = 0;
    uint32_t key_[KEY_SIZE] = {0};
    uint32_t counter_[COUNTER_SIZE] = {0};

    static constexpr MicroAPI::CastTrait castTraitTf = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING};
};

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::Init(GM_ADDR offset, GM_ADDR y, GM_ADDR count)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    blockOffSet_ = blockIdx_ * tiling_->perCoreHandleRandom;
    if (blockIdx_ == blockNum_ - 1) {
        blockFactor_ =  tiling_->tailBlockFactor;
        tailUbFactor_ = tiling_->tailBlockTailUbFactor;
    } else {
        blockFactor_ =  tiling_->blockFactor;
        tailUbFactor_ = tiling_->tailUbFactor;
    }
    ubFactor_ = tiling_->ubFactor;
    
    offsetGm_.SetGlobalBuffer((__gm__ OFFSET_T*)offset);
    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(uniformResult_, ubFactor_ * sizeof(float));
    pipe_->InitBuffer(outQue_, BUFFER_NUM, ubFactor_ * sizeof(float));
}

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::InitKeyAndCounter()
{
    key_[0] = static_cast<uint32_t>(tiling_->seed);
    key_[1] = static_cast<uint32_t>(tiling_->seed >> RIGHT_SHIFT);
    counter_[2] = static_cast<uint32_t>(tiling_->seed2);
    counter_[3] = static_cast<uint32_t>(tiling_->seed2 >> RIGHT_SHIFT);
}

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::Process()
{
    if (blockIdx_ > blockNum_) {
        return;
    }

    InitKeyAndCounter();

    if (offsetGm_(0) > 0) {
        Skip(offsetGm_(0));
    }

    int64_t groupCnt = blockOffSet_ / RESULT_ELEMENT_CNT;
    Skip(groupCnt);
    for (auto idx = 0; idx < blockFactor_; idx++) {
        currUbTilingSize_ = ubFactor_;
        if (idx == blockFactor_ - 1) {
            currUbTilingSize_ = tailUbFactor_;
        }
        curOffSet_ = blockOffSet_ + idx * ubFactor_;
        LocalTensor<float> yOutputTmp = uniformResult_.Get<float>();
        uint16_t uniformResCount = Ops::Base::CeilAlign(currUbTilingSize_, static_cast<OFFSET_T>(DOUBLE_UNIFORM_RESULT));
        PhiloxRandom<10>(
            yOutputTmp, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]}, uniformResCount);        
        AscendC::Simt::VF_CALL<SimtBoxMuller>(
            AscendC::Simt::Dim3{USED_THREAD}, (__ubuf__ float*)(yOutputTmp.GetPhyAddr()), uniformResCount);
        DataTypeHandle(yOutputTmp, currUbTilingSize_);
        CopyOut();
        groupCnt = currUbTilingSize_ / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }
    SyncAll();
    
    if (blockIdx_ == blockNum_ - 1) {
        offsetGm_(0) = offsetGm_(0) + tiling_->shapeSize * K_Reserved_Per_Output;
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(offsetGm_);
    }
}

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::Skip(const int64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> RIGHT_SHIFT);

    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[2] == 0) {
            ++counter_[3];
        }
    }
}

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::DataTypeHandle(
    LocalTensor<float>& yOutputTmp, const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQue_.AllocTensor<T>();
    if constexpr (AscendC::IsSameType<T, float>::value) {
        DataCopy(yOutput, yOutputTmp, ROUND_UP32(calCount));
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yOutput, yOutputTmp, RoundMode::CAST_NONE, calCount);
    } else {
        Cast(yOutput, yOutputTmp, RoundMode::CAST_RINT, calCount);
    }
    outQue_.EnQue(yOutput);
}

template <typename T, typename OFFSET_T>
__aicore__ inline void RandomStandardNormalV2Op<T, OFFSET_T>::CopyOut()
{
    LocalTensor<T> yOutput = outQue_.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currUbTilingSize_ * sizeof(T));
    DataCopyPad(outputGm_[curOffSet_], yOutput, copyParams);
    outQue_.FreeTensor(yOutput);
}

template <typename T, typename OFFSET_T>
__aicore__ inline uint32_t RandomStandardNormalV2Op<T, OFFSET_T>::ROUND_UP32(const uint32_t x) const
{
    if (x % BLOCK_SIZE != 0) {
        return (x / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }
    return x;
}

} // namespace RandomStandardNormalV2