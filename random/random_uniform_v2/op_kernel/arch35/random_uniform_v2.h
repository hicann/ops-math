/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace RandomUniformV2 {
using namespace AscendC;

template <typename T>
class RandomUniformV2Op
{
public:
    __aicore__ inline RandomUniformV2Op(
    TPipe* pipe, const RandomUniformV2TilingData4RegBase* __restrict tilingData)
    : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR offset);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void DataTypeHandle(const uint32_t calCount);
    __aicore__ inline void Uint32ToFloat(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void Uint16ToHalf(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void Uint16ToBfloat16(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t yCount);
    __aicore__ inline void InitKeyAndCounter();

private:
    TPipe* pipe_;
    const RandomUniformV2TilingData4RegBase* tiling_;

    static constexpr uint16_t BUFFER_NUM = 2;
    static constexpr uint16_t ALG_KEY_SIZE = 2;
    static constexpr uint16_t ALG_COUNTER_SIZE = 4;
    static constexpr uint16_t RESULT_ELEMENT_CNT = 4;
    static constexpr uint32_t INT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
    static constexpr uint64_t K_RESERVEED_PER_OUTPUT = 256;
    static constexpr uint32_t RIGHT_SHIFT = 32;

    GlobalTensor<T> outputGm_;
    GlobalTensor<int64_t> offsetGm_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;

    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    uint32_t blockIdx_;
    uint32_t curCoreProNum_ = 0;

    static constexpr MicroAPI::CastTrait castTraitTf = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING};
};

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Init(GM_ADDR y, GM_ADDR offset)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ > tiling_->blockNum) {
        return;
    }

    if (blockIdx_ == tiling_->blockNum - 1) {
        curCoreProNum_ = tiling_->tailCoreProNum;
    } else {
        curCoreProNum_ = tiling_->normalCoreProNum;
    }

    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    offsetGm_.SetGlobalBuffer((__gm__ int64_t *)offset);
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, tiling_->singleUbSize * sizeof(T));
    pipe_->InitBuffer(philoxQueBuf_, tiling_->singleUbSize * sizeof(uint32_t));
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::InitKeyAndCounter()
{
    key_[0] = static_cast<uint32_t>(tiling_->seed);
    key_[1] = static_cast<uint32_t>(tiling_->seed >> RIGHT_SHIFT);
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = static_cast<uint32_t>(tiling_->seed2);
    counter_[3] = static_cast<uint32_t>(tiling_->seed2 >> RIGHT_SHIFT);
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Process()
{
    if (blockIdx_ > tiling_->blockNum) {
        return;
    }

    InitKeyAndCounter();
    auto offsetValue = offsetGm_.GetValue(0);
    if (offsetValue > 0) {
        Skip(offsetValue);
    }

    SyncAll();
    auto blockOffSet = tiling_->normalCoreProNum * blockIdx_;
    auto groupCnt = (blockOffSet + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;

    Skip(groupCnt);
    int64_t philoxQueEleNum = tiling_->singleUbSize;
    int64_t ubRepeatimes = (curCoreProNum_ + philoxQueEleNum - 1) / philoxQueEleNum;
    for (auto idx = 0; idx < ubRepeatimes; idx++) {
        int64_t philoxNumPro = idx == (ubRepeatimes - 1) ? curCoreProNum_ - (ubRepeatimes - 1) * philoxQueEleNum : philoxQueEleNum;
        int64_t philoxNumOffset = idx * philoxQueEleNum;

        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        PhiloxRandom<10>(philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                         philoxNumPro);

        DataTypeHandle(philoxNumPro);
        int64_t yOffset = blockOffSet + philoxNumOffset;
        CopyOut(yOffset, philoxNumPro);
        groupCnt = (philoxNumPro + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }

    if (blockIdx_ == 0) {
        offsetValue = offsetValue + tiling_->outputSize * K_RESERVEED_PER_OUTPUT;
        offsetGm_.SetValue(0, offsetValue);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ATOMIC>(offsetGm_);
    }
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Skip(const uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> 32);

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

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::DataTypeHandle(const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQueY_.AllocTensor<T>();
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Uint16ToHalf(yOutput, calCount);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        Uint32ToFloat(yOutput, calCount);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Uint16ToBfloat16(yOutput, calCount);
    }
    outQueY_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::CopyOut(int64_t yOffset, int64_t yCount)
{
    LocalTensor<T> yOutput = outQueY_.DeQue<T>();
    __ubuf__ T* ubPhilox = (__ubuf__ T*)yOutput.GetPhyAddr();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(yCount * sizeof(T));
    DataCopyPad(outputGm_[yOffset], yOutput, copyParams);
    outQueY_.FreeTensor(yOutput);
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Uint32ToFloat(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ float* ubOut = (__ubuf__ float*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

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

        uint32_t sReg1 = static_cast<uint32_t>(calCount);   // x
        uint32_t sReg2 = static_cast<uint32_t>(0x7fffffu);  // 23 bit mantissa
        uint32_t exp = static_cast<uint32_t>(127);
        uint32_t sReg3 = exp << 23;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll);  //  1 = 0x7fffffu
        MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll);  // 3 = 127 << 23
        float sReg4 = static_cast<float>(-1.0);
        int32_t offSet = static_cast<int32_t>(INT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<float>(sReg1);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vReg0, ubPhilox, offSet);
            MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            MicroAPI::Or<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (MicroAPI::RegTensor<float>&)vReg4;
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ubOut, vReg6, offSet, mask);
        }
    }
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Uint16ToHalf(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ half* ubOut = (__ubuf__ half*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    SetCtrlSpr<60,60>(0);
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
        int32_t offset = static_cast<int32_t>(INT32_ONE_REPEAT);

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

    SetCtrlSpr<60,60>(1);
}

template <typename T>
__aicore__ inline void RandomUniformV2Op<T>::Uint16ToBfloat16(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ bfloat16_t* ubOut = (__ubuf__ bfloat16_t*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    SetCtrlSpr<60,60>(0);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> vRegT;
        MicroAPI::RegTensor<int16_t> vReg0;
        MicroAPI::RegTensor<int16_t> vReg1;
        MicroAPI::RegTensor<int16_t> vReg2;
        MicroAPI::RegTensor<int16_t> vReg3;
        MicroAPI::RegTensor<int16_t> vReg4;
        MicroAPI::RegTensor<bfloat16_t> vReg5;
        MicroAPI::RegTensor<bfloat16_t> vReg6;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        uint16_t sReg2 = static_cast<uint16_t>(0x7fu);  // 7 bit mantissa
        uint16_t exp = static_cast<uint16_t>(127);
        uint16_t sReg3 = exp << 7;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(vReg1, sReg2, maskAll);
        MicroAPI::Duplicate(vReg3, sReg3, maskAll);
        bfloat16_t sReg4 = static_cast<bfloat16_t>(-1.0);
        MicroAPI::MaskReg mask;
        int32_t offSet = static_cast<int32_t>(INT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<float>(sReg1);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vRegT, ubPhilox, offSet);
            MicroAPI::Cast<int16_t, int32_t, castTraitTf>(vReg0, vRegT, mask);
            MicroAPI::And<int16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            MicroAPI::Or<int16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (MicroAPI::RegTensor<bfloat16_t>&)vReg4;
            MicroAPI::Adds<bfloat16_t, bfloat16_t, MicroAPI::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
                ubOut, vReg6, offSet, mask);
        }
    }

    SetCtrlSpr<60,60>(1);
}

} // namespace RandomUniformV2