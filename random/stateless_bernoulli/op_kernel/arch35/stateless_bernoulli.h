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
 * \file stateless_bernoulli.h
 * \brief
 */
#ifndef STATELESS_BERNOULLI_H
#define STATELESS_BERNOULLI_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace StatelessBernoulli {

template <typename T, typename U>
class StatelessBernoulliKernel {
public:
    __aicore__ inline StatelessBernoulliKernel(){};
    __aicore__ inline void Init(GM_ADDR shape, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
        const StatelessBernoulliTilingData *__restrict tilingData, AscendC::TPipe *pipeIn);
    __aicore__ inline void Process(const StatelessBernoulliTilingData *__restrict tilingData);

private:
    __aicore__ inline void ParseTilingData(const StatelessBernoulliTilingData *__restrict tilingData);
    __aicore__ inline void RandUniformUint32(uint32_t calCount);
    __aicore__ inline void SelectValidMaskScalar(uint32_t calCount);
    __aicore__ inline void SelectValidMaskTensor(uint32_t calCount);
    __aicore__ inline void GenPhiloxRandom(uint32_t calCount);
    __aicore__ inline uint64_t RoundUp(uint64_t a, uint64_t b);
    __aicore__ inline uint64_t AlignUp256(uint64_t param);
    __aicore__ inline uint8_t PadAlignByte32(uint64_t param);
    __aicore__ inline void Skip(uint64_t count);
    __aicore__ inline void CopyIn(uint32_t loopIdx, uint32_t calCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t calCount, const StatelessBernoulliTilingData *__restrict tilingData);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t calCount);

private:
    AscendC::TPipe *pipe_;
    constexpr static int64_t BUFFER_NUM = 2;
    constexpr static uint32_t ALG_KEY_SIZE = 2;
    constexpr static uint32_t ALG_COUNTER_SIZE = 4;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t RESULT_ELEMENT_CNT = 4;
    constexpr static float CURAND_2POW32_INV = 2.3283064365386963e-10;
    constexpr static float COEFF_2POW32_INV = 2.0f;
    constexpr static uint32_t gainCoeff = 2;
    constexpr static uint32_t roundUpByte256 = 256;
    constexpr static uint32_t roundUpNum = AscendC::ONE_BLK_SIZE / sizeof(T);
    constexpr static uint32_t RIGHT_SHIFT_BIT = 32;
    constexpr static uint32_t COUNTER_IDX2 = 2;
    constexpr static uint32_t COUNTER_IDX3 = 3;

    AscendC::GlobalTensor<T> probInputGm_;
    AscendC::GlobalTensor<U> outputGm_;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> probQueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> philoxQueBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcuDataBuf_;

    float prob_ = 0.0f;
    uint64_t currBlockTilingSize_ = 0;
    uint64_t currUbTilingSize_ = 0;
    uint64_t currLoopCount_ = 0;
    uint64_t blockOffset_ = 0;

    uint64_t countIndex0_ = 0;
    uint64_t countIndex1_ = 1;
    uint64_t countIndex2_ = 2;
    uint64_t countIndex3_ = 3;

    uint64_t ubTilingSize_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    bool isPadding_ = false;
    uint8_t rightPadding_ = 0;

    static constexpr AscendC::MicroAPI::CastTrait castTraitB64ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT
    };

    static constexpr AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN
    };
};

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::ParseTilingData(
    const StatelessBernoulliTilingData *__restrict tilingData)
{
    ubTilingSize_ = tilingData->ubTilingSize;
    for (uint64_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tilingData->key[i];
    }
    for (uint64_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tilingData->counter[i];
    }
}

template <typename T, typename U>
__aicore__ inline uint64_t StatelessBernoulliKernel<T, U>::RoundUp(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
};

template <typename T, typename U>
__aicore__ inline uint64_t StatelessBernoulliKernel<T, U>::AlignUp256(uint64_t param)
{
    return (param + roundUpByte256 - 1) / roundUpByte256 * roundUpByte256;
};

template <typename T, typename U>
__aicore__ inline uint8_t StatelessBernoulliKernel<T, U>::PadAlignByte32(uint64_t param)
{
    return roundUpNum - param % roundUpNum;
};

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::Skip(uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> RIGHT_SHIFT_BIT);

    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[COUNTER_IDX2] == 0) {
            ++counter_[COUNTER_IDX3];
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::Init(GM_ADDR shape, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
    const StatelessBernoulliTilingData *__restrict tilingData, AscendC::TPipe *pipeIn)
{
    // Init tiling data
    ParseTilingData(tilingData);
    auto blockIdx = AscendC::GetBlockIdx();
    blockOffset_ = blockIdx * tilingData->blockTilingSize;

    if (blockIdx == tilingData->blockNum - 1) {
        currBlockTilingSize_ = tilingData->tailBlockTilingSize;
        currLoopCount_ = tilingData->tailBlockLoopCount;
    } else {
        currBlockTilingSize_ = tilingData->blockTilingSize;
        currLoopCount_ = tilingData->blockLoopCount;
    }

    // ubTiling size less than block tiling size
    if (currBlockTilingSize_ < ubTilingSize_) {
        ubTilingSize_ = currBlockTilingSize_;
    }

    // SetBuffer
    probInputGm_.SetGlobalBuffer((__gm__ T *)prob);
    outputGm_.SetGlobalBuffer((__gm__ U *)y);
    if (!tilingData->isProbScalar && tilingData->outputSize > tilingData->probTensorSize) {
        AscendC::InitGlobalMemory(outputGm_, tilingData->outputSize, static_cast<U>(0));
        AscendC::SyncAll();
    }

    // InitBuffer
    pipe_ = pipeIn;
    pipe_->InitBuffer(probQueX_, BUFFER_NUM, AlignUp256(ubTilingSize_ * sizeof(T)));
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, AlignUp256(ubTilingSize_ * sizeof(U)));
    pipe_->InitBuffer(philoxQueBuf_, AlignUp256(ubTilingSize_ * sizeof(uint32_t)));
    pipe_->InitBuffer(calcuDataBuf_, AlignUp256(ubTilingSize_ * sizeof(float)));

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

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::GenPhiloxRandom(uint32_t calCount)
{
    AscendC::LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    AscendC::PhiloxRandom<10>(philoxRes, { key_[0], key_[1] }, { counter_[countIndex0_], counter_[countIndex1_], counter_[countIndex2_], counter_[countIndex3_] }, calCount);
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::RandUniformUint32(uint32_t calCount)
{
    // philox result saved in philoxQueBuf
    AscendC::LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int64_t *ubPhilox = (__ubuf__ int64_t *)philoxRes.GetPhyAddr();
    AscendC::LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    __ubuf__ float *ubOut = (__ubuf__ float *)caluData.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint16_t repeatTimes = RoundUp(calCount, vfLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int64_t> vReg0;
        AscendC::MicroAPI::RegTensor<float> vReg1;
        AscendC::MicroAPI::RegTensor<float> vReg2;
        AscendC::MicroAPI::RegTensor<float> vReg3;
        AscendC::MicroAPI::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount) * gainCoeff;
        float sReg3 = static_cast<float>(CURAND_2POW32_INV);
        float sReg4 = static_cast<float>(CURAND_2POW32_INV / COEFF_2POW32_INV);
        int32_t offset = static_cast<int32_t>(vfLen);

        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(sReg1);
            AscendC::MicroAPI::DataCopy<int64_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B32>(vReg0, ubPhilox, offset / gainCoeff);
            AscendC::MicroAPI::Cast<float, int64_t, castTraitB64ToB32>(vReg1, vReg0, mask);
            AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vReg2, vReg1, sReg3, mask);
            AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vReg3, vReg2, sReg4, mask);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK_B64>(
                ubOut, vReg3, offset, mask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::SelectValidMaskScalar(uint32_t calCount)
{
    AscendC::LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    __ubuf__ float *ubCaluData = (__ubuf__ float *)caluData.GetPhyAddr();

    AscendC::LocalTensor<U> yOutput = outQueY_.AllocTensor<U>();
    __ubuf__ U *ubOut = (__ubuf__ U *)yOutput.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatTimes = RoundUp(calCount, vfLen);

    __VEC_SCOPE__
    {
        int32_t offset = static_cast<int32_t>(vfLen);
        float pScalar = static_cast<float>(prob_);
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg cmpMaskReg;
        AscendC::MicroAPI::RegTensor<float> vCaluReg;

        if constexpr (sizeof(U) == sizeof(int64_t)) {
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumTwo> vSrcReg0;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumTwo> vSrcReg1;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumTwo> vDstReg0;

            maskReg = AscendC::MicroAPI::UpdateMask<U, AscendC::MicroAPI::RegTraitNumTwo>(calCount);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg0, static_cast<U>(1), maskReg);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg1, static_cast<U>(0), maskReg);

            for (uint16_t i = 0; i < repeatTimes; ++i) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_NORM>(vCaluReg, ubCaluData, offset);
                AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LT>(cmpMaskReg, vCaluReg, pScalar, maskReg);
                AscendC::MicroAPI::Select<U>(vDstReg0, vSrcReg0, vSrcReg1, cmpMaskReg);
                AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM>(ubOut, vDstReg0, offset, maskReg);
            }
        } else {
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumOne> vSrcReg0;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumOne> vSrcReg1;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumOne> vDstReg0;

            maskReg = AscendC::MicroAPI::UpdateMask<int32_t, AscendC::MicroAPI::RegTraitNumOne>(calCount);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg0, static_cast<U>(1), maskReg);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg1, static_cast<U>(0), maskReg);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_NORM>(vCaluReg, ubCaluData, offset);
                AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LT>(cmpMaskReg, vCaluReg, pScalar, maskReg);
                AscendC::MicroAPI::Select<U>(vDstReg0, vSrcReg0, vSrcReg1, cmpMaskReg);
                if constexpr (AscendC::IsSameType<U, int32_t>::value || AscendC::IsSameType<U, uint32_t>::value || AscendC::IsSameType<U, float>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM>(ubOut, vDstReg0, offset, maskReg);
                } else if constexpr (AscendC::IsSameType<U, half>::value || AscendC::IsSameType<U, bfloat16_t>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(ubOut, vDstReg0, offset, maskReg);
                } else if constexpr (AscendC::IsSameType<U, int16_t>::value || AscendC::IsSameType<U, uint16_t>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(ubOut, vDstReg0, offset, maskReg);
                } else if constexpr (AscendC::IsSameType<U, int8_t>::value || AscendC::IsSameType<U, uint8_t>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(ubOut, vDstReg0, offset, maskReg);
                }
            }
        }
    }

    outQueY_.EnQue(yOutput);
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::SelectValidMaskTensor(uint32_t calCount)
{
    AscendC::LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    __ubuf__ float *ubCaluData = (__ubuf__ float *)caluData.GetPhyAddr();

    AscendC::LocalTensor<T> probInputUb = probQueX_.DeQue<T>();
    __ubuf__ T *ubProbIn = (__ubuf__ T *)probInputUb.GetPhyAddr();

    AscendC::LocalTensor<U> yOutput = outQueY_.AllocTensor<U>();
    __ubuf__ U *ubOut = (__ubuf__ U *)yOutput.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatTimes = RoundUp(calCount, vfLen);

    __VEC_SCOPE__
    {
        int32_t offset = static_cast<int32_t>(vfLen);
        AscendC::MicroAPI::MaskReg maskReg0;
        AscendC::MicroAPI::MaskReg cmpMaskReg0;
        AscendC::MicroAPI::RegTensor<float> vCaluReg;
        AscendC::MicroAPI::RegTensor<T> vProbRegT;
        AscendC::MicroAPI::RegTensor<float> vProbRegFp;

        if constexpr (sizeof(U) == sizeof(int64_t)) {
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumTwo> vSrcReg0;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumTwo> vSrcReg1;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumTwo> vDstReg0;

            maskReg0 = AscendC::MicroAPI::UpdateMask<U, AscendC::MicroAPI::RegTraitNumTwo>(calCount);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg0, static_cast<U>(1), maskReg0);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg1, static_cast<U>(0), maskReg0);

            for (uint16_t i = 0; i < repeatTimes; ++i) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_NORM>(vCaluReg, ubCaluData, offset);
                if constexpr (AscendC::IsSameType<T, float>::value) {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_NORM>(vProbRegT, ubProbIn, offset);
                    AscendC::MicroAPI::Adds<float, float>(vProbRegFp, vProbRegT, static_cast<float>(0.0), maskReg0);
                } else {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vProbRegT, ubProbIn, offset);
                    AscendC::MicroAPI::Cast<float, T, castTraitB16ToB32>(vProbRegFp, vProbRegT, maskReg0);
                }
                AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::LT>(cmpMaskReg0, vCaluReg, vProbRegFp, maskReg0);
                AscendC::MicroAPI::Select<U>(vDstReg0, vSrcReg0, vSrcReg1, cmpMaskReg0);
                AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM>(ubOut, vDstReg0, offset, maskReg0);
            }
        } else {
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumOne> vSrcReg0;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumOne> vSrcReg1;
            AscendC::MicroAPI::RegTensor<U, AscendC::MicroAPI::RegTraitNumOne> vDstReg;

            maskReg0 = AscendC::MicroAPI::UpdateMask<int32_t, AscendC::MicroAPI::RegTraitNumOne>(calCount);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg0, static_cast<U>(1), maskReg0);
            AscendC::MicroAPI::Duplicate<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vSrcReg1, static_cast<U>(0), maskReg0);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_NORM>(vCaluReg, ubCaluData, offset);
                if constexpr (AscendC::IsSameType<T, float>::value) {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_NORM>(vProbRegT, ubProbIn, offset);
                    AscendC::MicroAPI::Adds<float, float>(vProbRegFp, vProbRegT, static_cast<float>(0.0), maskReg0);
                } else {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vProbRegT, ubProbIn, offset);
                    AscendC::MicroAPI::Cast<float, T, castTraitB16ToB32>(vProbRegFp, vProbRegT, maskReg0);
                }
                AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::LT>(cmpMaskReg0, vCaluReg, vProbRegFp, maskReg0);
                AscendC::MicroAPI::Select<U>(vDstReg, vSrcReg0, vSrcReg1, cmpMaskReg0);
                if constexpr (AscendC::IsSameType<U, int32_t>::value || AscendC::IsSameType<U, uint32_t>::value || AscendC::IsSameType<U, float>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM>(ubOut, vDstReg, offset, maskReg0);
                } else if constexpr (AscendC::IsSameType<U, half>::value || AscendC::IsSameType<U, bfloat16_t>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(ubOut, vDstReg, offset, maskReg0);
                } else if constexpr (AscendC::IsSameType<U, int16_t>::value || AscendC::IsSameType<U, uint16_t>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(ubOut, vDstReg, offset, maskReg0);
                } else if constexpr (AscendC::IsSameType<U, int8_t>::value || AscendC::IsSameType<U, uint8_t>::value) {
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(ubOut, vDstReg, offset, maskReg0);
                }
            }
        }
    }
    
    probQueX_.FreeTensor(probInputUb);
    outQueY_.EnQue(yOutput);
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::Compute(uint32_t loopIdx, uint32_t calCount, const StatelessBernoulliTilingData *__restrict tilingData)
{
    GenPhiloxRandom(calCount);
    RandUniformUint32(calCount);
    if (tilingData->isProbScalar) {
        SelectValidMaskScalar(calCount);
    } else {
        SelectValidMaskTensor(calCount);
    }
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::CopyIn(uint32_t loopIdx, uint32_t calCount)
{
    AscendC::LocalTensor<T> probInputUb = probQueX_.AllocTensor<T>();
    if (calCount % roundUpNum) {
        isPadding_ = true;
        rightPadding_ = PadAlignByte32(calCount);
    }
    AscendC::DataCopyExtParams copyParams { 1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0 };
    AscendC::DataCopyPadExtParams<T> padParams { isPadding_, 0, rightPadding_, 0 };
    AscendC::DataCopyPad(probInputUb, probInputGm_[blockOffset_ + loopIdx * ubTilingSize_], copyParams, padParams);
    probQueX_.EnQue<T>(probInputUb);
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::CopyOut(uint32_t loopIdx, uint32_t calCount)
{
    AscendC::LocalTensor<U> yOutput = outQueY_.DeQue<U>();
    AscendC::DataCopyExtParams copyParams { 1, (uint32_t)(calCount * sizeof(U)), 0, 0, 0 };
    AscendC::DataCopyPad(outputGm_[blockOffset_ + loopIdx * ubTilingSize_], yOutput, copyParams);
    outQueY_.FreeTensor(yOutput);
}

template <typename T, typename U>
__aicore__ inline void StatelessBernoulliKernel<T, U>::Process(const StatelessBernoulliTilingData *__restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= tilingData->blockNum) {
        return;
    }

    auto groupCnt = RoundUp(blockOffset_, RESULT_ELEMENT_CNT);
    Skip(groupCnt);

    for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((idx == currLoopCount_ - 1) && (currBlockTilingSize_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = currBlockTilingSize_ % ubTilingSize_;
        }
        if (!tilingData->isProbScalar) {
            CopyIn(idx, currUbTilingSize_);
        }
        Compute(idx, currUbTilingSize_, tilingData);
        CopyOut(idx, currUbTilingSize_);

        groupCnt = RoundUp(currUbTilingSize_, RESULT_ELEMENT_CNT);
        Skip(groupCnt);
    }
}
} // namespace StatelessBernoulli
#endif // STATELESS_BERNOULLI_H