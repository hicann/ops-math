/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stateless_drop_out_gen_mask_pt.h
 * \brief
 */
#ifndef STATELESS_DROP_OUT_GEN_MASK_PT_H
#define STATELESS_DROP_OUT_GEN_MASK_PT_H

#pragma once
#include "stateless_drop_out_gen_mask_base.h"

namespace StatelessDropOutGenMask {
using namespace AscendC;

template <typename T>
class StatelessDropOutGenMaskPt : public StatelessDropOutGenMaskBase<T> {
public:
    __aicore__ inline StatelessDropOutGenMaskPt(){};
    __aicore__ inline void Init(
        GM_ADDR shape, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
        const StatelessDropOutGenMaskTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const StatelessDropOutGenMaskTilingData* __restrict tilingData);
    __aicore__ inline void Uint32ToFloat(uint32_t calCount);
    __aicore__ inline void CompareMask(uint32_t calCount);
    __aicore__ inline void Skip(uint64_t count);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t calCount);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t calCount);

private:
    TPipe* pipe_;
    constexpr static int64_t BUFFER_NUM = 2;
    constexpr static uint32_t ALG_KEY_SIZE = 2;
    constexpr static uint32_t ALG_COUNTER_SIZE = 4;

    GlobalTensor<T> probInputGm_;
    GlobalTensor<uint8_t> outputGm_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TBuf<QuePosition::VECCALC> calcuDataBuf_;
    TBuf<QuePosition::VECCALC> tempDataBuf_;

    float prob_ = 0.0f;
    uint32_t currBlockTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0;
    uint32_t currLoopCount_ = 0;
    uint32_t blockOffset_ = 0;

    uint32_t blockNum_ = 0;
    uint32_t blockTilingSize_ = 0;
    uint32_t tailBlockTilingSize_ = 0;
    uint32_t blockLoopCount_ = 0;
    uint32_t tailBlockLoopCount_ = 0;
    uint32_t ubTilingSize_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    static constexpr MicroAPI::CastTrait castTraitPt = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::ParseTilingData(
    const StatelessDropOutGenMaskTilingData* __restrict tilingData)
{
    blockNum_ = tilingData->blockNum;
    blockTilingSize_ = tilingData->blockTilingSize;
    tailBlockTilingSize_ = tilingData->tailBlockTilingSize;
    blockLoopCount_ = tilingData->blockLoopCount;
    tailBlockLoopCount_ = tilingData->tailBlockLoopCount;
    ubTilingSize_ = tilingData->ubTilingSize;
    for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tilingData->key[i];
    }
    for (uint32_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tilingData->counter[i];
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Skip(uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> this->RIGHT_SHIFT_BIT);

    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[this->COUNTER_IDX2] == 0) {
            ++counter_[this->COUNTER_IDX3];
        }
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Init(
    GM_ADDR shape, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
    const StatelessDropOutGenMaskTilingData* __restrict tilingData, TPipe* pipeIn)
{
    // Init tiling data
    ParseTilingData(tilingData);
    auto blockIdx = GetBlockIdx();
    blockOffset_ = blockIdx * blockTilingSize_;

    if (blockIdx == blockNum_ - 1) {
        currBlockTilingSize_ = tailBlockTilingSize_;
        currLoopCount_ = tailBlockLoopCount_;
    } else {
        currBlockTilingSize_ = blockTilingSize_;
        currLoopCount_ = blockLoopCount_;
    }

    // ubTiling size less than block tiling size
    if (currBlockTilingSize_ < ubTilingSize_) {
        ubTilingSize_ = currBlockTilingSize_;
    }

    // SetBuffer
    probInputGm_.SetGlobalBuffer((__gm__ T*)prob);
    outputGm_.SetGlobalBuffer((__gm__ uint8_t*)y + (blockOffset_ / this->byteBitRatio));

    // InitBuffer
    pipe_ = pipeIn;
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, this->AlignUp256(ubTilingSize_ * sizeof(float)));
    pipe_->InitBuffer(philoxQueBuf_, this->AlignUp256(ubTilingSize_ * sizeof(uint32_t)));
    pipe_->InitBuffer(calcuDataBuf_, this->AlignUp256(ubTilingSize_ * sizeof(float)));
    pipe_->InitBuffer(tempDataBuf_, this->AlignUp256(ubTilingSize_ * sizeof(float)));

    // GetValue
    T currProb = probInputGm_.GetValue(0);
    if constexpr (AscendC::IsSameType<T, half>::value) {
        prob_ = static_cast<float>(currProb);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        prob_ = static_cast<T>(currProb);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        prob_ = AscendC::ToFloat(currProb);
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Uint32ToFloat(uint32_t calCount)
{
    /*
    [calculation formula]
    (x * CURAND_2POW32_INV) + (CURAND_2POW32_INV / 2.0f)
    */

    // philox result saved in philoxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int64_t* ubPhilox = (__ubuf__ int64_t*)philoxRes.GetPhyAddr();
    LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    __ubuf__ float* ubOut = (__ubuf__ float*)caluData.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(int64_t);
    uint32_t repeatTimes = this->CeilDiv(calCount, vfLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int64_t> vReg0;
        MicroAPI::RegTensor<float> vReg1;
        MicroAPI::RegTensor<float> vReg2;
        MicroAPI::RegTensor<float> vReg3;
        MicroAPI::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount) * this->gainCoeff;
        float sReg3 = static_cast<float>(this->CURAND_2POW32_INV);
        float sReg4 = static_cast<float>(this->CURAND_2POW32_INV / this->COEFF_2POW32_INV);
        int32_t offset = static_cast<int32_t>(vfLen);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<int32_t>(sReg1);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B32>(
                vReg0, ubPhilox, offset / this->gainCoeff);
            MicroAPI::Cast<float, int64_t, castTraitPt>(vReg1, vReg0, mask);
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg1, sReg3, mask);
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(vReg3, vReg2, sReg4, mask);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B64>(
                ubOut, vReg3, offset, mask);
        }
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::CompareMask(uint32_t calCount)
{
    LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    LocalTensor<float> tempData = tempDataBuf_.Get<float>();
    LocalTensor<uint8_t> yOutput = outQueY_.AllocTensor<uint8_t>();

    uint32_t countAlign256 = this->AlignUp256(calCount);
    AscendC::Duplicate(tempData, static_cast<float>(1.0), countAlign256);
    AscendC::Adds(tempData, caluData, static_cast<float>(0.0), calCount);
    AscendC::CompareScalar(yOutput, tempData, static_cast<float>(prob_), CMPMODE::LT, countAlign256);
    outQueY_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Compute(uint32_t loopIdx, uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    AscendC::PhiloxRandom<10>(
        philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]}, calCount);
    Uint32ToFloat(calCount);
    CompareMask(calCount);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::CopyOut(uint32_t loopIdx, uint32_t calCount)
{
    LocalTensor<uint8_t> yOutput = outQueY_.DeQue<uint8_t>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->CeilDiv(calCount, this->byteBitRatio) * sizeof(uint8_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[(loopIdx * ubTilingSize_) / this->byteBitRatio], yOutput, copyParams);
    outQueY_.FreeTensor(yOutput);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Process()
{
    if (GetBlockIdx() >= this->blockNum_) {
        return;
    }

    auto groupCnt = this->CeilDiv(blockOffset_, this->RESULT_ELEMENT_CNT);
    Skip(groupCnt);

    for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((idx == currLoopCount_ - 1) && (currBlockTilingSize_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = currBlockTilingSize_ % ubTilingSize_;
        }
        Compute(idx, currUbTilingSize_);
        CopyOut(idx, currUbTilingSize_);

        groupCnt = this->CeilDiv(currUbTilingSize_, this->RESULT_ELEMENT_CNT);
        Skip(groupCnt);
    }
}
} // namespace StatelessDropOutGenMask
#endif // STATELESS_DROP_OUT_GEN_MASK_PT_H