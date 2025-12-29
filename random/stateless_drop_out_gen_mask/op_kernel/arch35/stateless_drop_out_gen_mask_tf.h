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
 * \file stateless_drop_out_gen_mask_tf.h
 * \brief
 */

#ifndef STATELESS_DROP_OUT_GEN_MASK_TF_H
#define STATELESS_DROP_OUT_GEN_MASK_TF_H

#pragma once
#include "stateless_drop_out_gen_mask_base.h"

namespace StatelessDropOutGenMask {
using namespace AscendC;

template <typename T>
class StatelessDropOutGenMaskTf : public StatelessDropOutGenMaskBase<T> {
public:
    __aicore__ inline StatelessDropOutGenMaskTf(){};
    __aicore__ inline void Init(
        GM_ADDR shape, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
        const StatelessDropOutGenMaskTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const StatelessDropOutGenMaskTilingData* __restrict tilingData);
    __aicore__ inline void Skip(uint64_t count);
    __aicore__ inline void CompareMask(uint32_t calCount);
    __aicore__ inline void Uint32ToFloat(uint32_t calCount);
    __aicore__ inline void Uint16ToHalf(uint32_t calCount);
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

    T prob_ = 0.0;
    uint32_t currBlockTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0;
    uint32_t currLoopCount_ = 0;
    uint32_t blockOffset_ = 0;

    uint32_t blockNum_ = 0;
    uint32_t ubTilingSize_ = 0;
    uint32_t blockLoopCount_ = 0;
    uint32_t tailBlockLoopCount_ = 0;
    uint32_t blockTilingSize_ = 0;
    uint32_t tailBlockTilingSize_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    static constexpr MicroAPI::CastTrait castTraitTf = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING};
};

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::ParseTilingData(
    const StatelessDropOutGenMaskTilingData* __restrict tilingData)
{
    blockNum_ = tilingData->blockNum;
    ubTilingSize_ = tilingData->ubTilingSize;
    blockLoopCount_ = tilingData->blockLoopCount;
    tailBlockLoopCount_ = tilingData->tailBlockLoopCount;
    blockTilingSize_ = tilingData->blockTilingSize;
    tailBlockTilingSize_ = tilingData->tailBlockTilingSize;
    for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tilingData->key[i];
    }
    for (uint32_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tilingData->counter[i];
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::Skip(uint64_t count)
{
    uint32_t countHi = static_cast<uint32_t>(count >> 32);
    const uint32_t countLo = static_cast<uint32_t>(count);

    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[this->COUNTER_IDX2] == 0) {
            ++counter_[this->COUNTER_IDX3];
        }
    }
    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::Init(
    GM_ADDR shape, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
    const StatelessDropOutGenMaskTilingData* __restrict tilingData, TPipe* pipeIn)
{
    // Init tiling data
    ParseTilingData(tilingData);
    auto blockIdx = GetBlockIdx();
    blockOffset_ = blockIdx * blockTilingSize_;

    if (blockIdx == blockNum_ - 1) {
        currLoopCount_ = tailBlockLoopCount_;
        currBlockTilingSize_ = tailBlockTilingSize_;
    } else {
        currLoopCount_ = blockLoopCount_;
        currBlockTilingSize_ = blockTilingSize_;
    }

    // ubTiling size less than block tiling size
    if (currBlockTilingSize_ <= ubTilingSize_) {
        ubTilingSize_ = currBlockTilingSize_;
    }

    // SetBuffer
    probInputGm_.SetGlobalBuffer((__gm__ T*)prob);
    outputGm_.SetGlobalBuffer((__gm__ uint8_t*)y + (blockOffset_ / this->byteBitRatio));

    // InitBuffer
    pipe_ = pipeIn;
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, this->AlignUp256(ubTilingSize_ * sizeof(uint8_t)));
    pipe_->InitBuffer(philoxQueBuf_, this->AlignUp256(ubTilingSize_ * sizeof(uint32_t)));
    pipe_->InitBuffer(calcuDataBuf_, this->AlignUp256(ubTilingSize_ * sizeof(T)));

    // GetValue
    prob_ = probInputGm_.GetValue(0);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::Uint32ToFloat(uint32_t calCount)
{
    /*
    |1|_____8____|___________23___________|
    |s|exponent  | mantissa               |

    const uint32_t man = x & 0x7fffffu   // 23 bit mantissa
    const uint32_t exp = static_cast<uint32_t>(127)  // 7 bit exp
    const uint32_t val = (exp << 23) | man
    float result
    memcpy(&result, &val, sizeof(val))
    return  result - 1.0f
    */

    // philox result saved in philoxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    __ubuf__ float* ubOut = (__ubuf__ float*)caluData.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t repeatTimes = this->CeilDiv(calCount, vfLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> vReg0;
        MicroAPI::RegTensor<int32_t> vReg1;
        MicroAPI::RegTensor<int32_t> vReg2;
        MicroAPI::RegTensor<int32_t> vReg3;
        MicroAPI::RegTensor<int32_t> vReg4;
        MicroAPI::RegTensor<float> vReg5;
        MicroAPI::RegTensor<float> vReg6;
        MicroAPI::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        uint32_t sReg2 = static_cast<uint32_t>(0x7fffffu);
        uint32_t exp = static_cast<uint32_t>(127);
        uint32_t sReg3 = exp << 23;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<int32_t>();
        MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll);
        MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll);

        float sReg4 = static_cast<float>(-1.0);
        int32_t offset = static_cast<int32_t>(vfLen);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<float>(sReg1);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vReg0, ubPhilox, offset);
            MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            MicroAPI::Or<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (MicroAPI::RegTensor<float>&)vReg4;
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ubOut, vReg6, offset, mask);
        }
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::Uint16ToHalf(uint32_t calCount)
{
    /*
    |1|____5____|__________10___________|
    |s|exponent  | mantissa             |

    const uint16_t man = x & 0x3ffu  // 10 bit mantissa
    const uint16_t expr = staic_cast<uint16_t>(15)  // 5 bit exp
    const uint16_t val = (exp << 10) | man
    float16 result
    memcpy(&result, &val, sizeof(val))
    return  result - half(1.0)
    */

    // philox result saved in philoxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    LocalTensor<half> caluData = calcuDataBuf_.Get<half>();
    __ubuf__ half* ubOut = (__ubuf__ half*)caluData.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t repeatTimes = this->CeilDiv(calCount, vfLen);

    set_ctrl(sbitset0(get_ctrl(), this->SET_CTRL_VAL));
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> vRegT;
        MicroAPI::RegTensor<int16_t> vReg0;
        MicroAPI::RegTensor<int16_t> vReg1;
        MicroAPI::RegTensor<int16_t> vReg2;
        MicroAPI::RegTensor<int16_t> vReg3;
        MicroAPI::RegTensor<int16_t> vReg4;
        MicroAPI::RegTensor<half> vReg5;
        MicroAPI::RegTensor<half> vReg6;
        MicroAPI::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        uint16_t sReg2 = static_cast<uint16_t>(0x3ffu);
        uint16_t exp = static_cast<uint16_t>(15);
        uint16_t sReg3 = exp << 10;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<int32_t>();
        MicroAPI::Duplicate<int16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll);
        MicroAPI::Duplicate<int16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll);

        half sReg4 = static_cast<half>(-1.0);
        int32_t offset = static_cast<int32_t>(vfLen);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<float>(sReg1);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vRegT, ubPhilox, offset);
            MicroAPI::Cast<int16_t, int32_t, castTraitTf>(vReg0, vRegT, mask);
            MicroAPI::And<int16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            MicroAPI::Or<int16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (MicroAPI::RegTensor<half>&)vReg4;
            MicroAPI::Adds<half, half, MicroAPI::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            MicroAPI::DataCopy<half, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
                ubOut, vReg6, offset, mask);
        }
    }
    set_ctrl(sbitset1(get_ctrl(), this->SET_CTRL_VAL));
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::CompareMask(uint32_t calCount)
{
    LocalTensor<uint8_t> yOut = outQueY_.AllocTensor<uint8_t>();
    uint32_t countAlign256 = this->AlignUp256(calCount);

    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
        AscendC::CompareScalar(yOut, caluData, AscendC::ToFloat(prob_), CMPMODE::LT, countAlign256);
    } else {
        LocalTensor<T> caluData = calcuDataBuf_.Get<T>();
        AscendC::CompareScalar(yOut, caluData, static_cast<T>(prob_), CMPMODE::LT, countAlign256);
    }

    outQueY_.EnQue(yOut);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::Compute(uint32_t loopIdx, uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    AscendC::PhiloxRandom<10>(
        philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]}, calCount);

    if constexpr (AscendC::IsSameType<T, half>::value) {
        Uint16ToHalf(calCount);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        Uint32ToFloat(calCount);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Uint32ToFloat(calCount);
    }

    CompareMask(calCount);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::CopyOut(uint32_t loopIdx, uint32_t calCount)
{
    LocalTensor<uint8_t> yOut = outQueY_.DeQue<uint8_t>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->CeilDiv(calCount, this->byteBitRatio) * sizeof(uint8_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[(loopIdx * ubTilingSize_) / this->byteBitRatio], yOut, copyParams);
    outQueY_.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskTf<T>::Process()
{
    if (GetBlockIdx() >= this->blockNum_) {
        return;
    }

    auto groupCnt = this->CeilDiv(blockOffset_, this->RESULT_ELEMENT_CNT);
    Skip(groupCnt);

    for (uint32_t id = 0; id < currLoopCount_; id++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((id == currLoopCount_ - 1) && (currBlockTilingSize_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = currBlockTilingSize_ % ubTilingSize_;
        }
        Compute(id, currUbTilingSize_);
        CopyOut(id, currUbTilingSize_);

        groupCnt = this->CeilDiv(currUbTilingSize_, this->RESULT_ELEMENT_CNT);
        Skip(groupCnt);
    }
}
} // namespace StatelessDropOutGenMask
#endif // STATELESS_DROP_OUT_GEN_MASK_TF_H