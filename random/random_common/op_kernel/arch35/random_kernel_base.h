/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANDOM_KERNEL_BASE_H
#define RANDOM_KERNEL_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "random_unified_tiling_data_arch35.h"

namespace RandomKernelBase {
using namespace AscendC;

static constexpr MicroAPI::CastTrait castTraitTf = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING};

static constexpr uint16_t ALG_KEY_SIZE = 2;
static constexpr uint16_t ALG_COUNTER_SIZE = 4;
static constexpr uint32_t INT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
static constexpr uint32_t RIGHT_SHIFT = 32;
static constexpr uint16_t BLOCK_SIZE = 32;
static constexpr uint16_t DOUBLE_UNIFORM_RESULT = 2;
static constexpr uint32_t INT32_FLOAT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
static constexpr float DOUBLE_MULTIPLE = 2.0f;

template <typename T>
__aicore__ inline void CopyOut(
    LocalTensor<T> yLocal, GlobalTensor<T> yGm, uint32_t burstNum, uint32_t busrtLength, int64_t gmOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = burstNum;
    copyParams.blockLen = busrtLength;
    DataCopyPad(yGm[gmOffset], yLocal, copyParams);
}

template <typename T>
__aicore__ inline void Uint32ToFloat(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
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

        uint32_t sReg1 = static_cast<uint32_t>(calCount);  // x
        uint32_t sReg2 = static_cast<uint32_t>(0x7fffffu); // 23 bit mantissa
        uint32_t exp = static_cast<uint32_t>(127);
        uint32_t sReg3 = exp << 23;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll); //  1 = 0x7fffffu
        MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll); // 3 = 127 << 23
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
__aicore__ inline void Uint16ToHalf(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ half* ubOut = (__ubuf__ half*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    SetCtrlSpr<60, 60>(0);
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

    SetCtrlSpr<60, 60>(1);
}

template <typename T>
__aicore__ inline void Uint16ToBfloat16(
    LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ bfloat16_t* ubOut = (__ubuf__ bfloat16_t*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    SetCtrlSpr<60, 60>(0);
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
        uint16_t sReg2 = static_cast<uint16_t>(0x7fu); // 7 bit mantissa
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

    SetCtrlSpr<60, 60>(1);
}

template <typename T>
__aicore__ inline void U32Conversion(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Uint16ToHalf(yOutput, philoxRes, calCount);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        Uint32ToFloat(yOutput, philoxRes, calCount);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Uint16ToBfloat16(yOutput, philoxRes, calCount);
    }
}

template <typename T>
__aicore__ inline void Float32Conversion(
    LocalTensor<T> yOutput, LocalTensor<float> normalFloatResult, const uint32_t calCount)
{
    if constexpr (AscendC::IsSameType<T, float>::value) {
        DataCopy(yOutput, normalFloatResult, Ops::Base::CeilAlign(calCount, static_cast<uint32_t>(BLOCK_SIZE)));
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yOutput, normalFloatResult, RoundMode::CAST_NONE, calCount);
    } else {
        Cast(yOutput, normalFloatResult, RoundMode::CAST_RINT, calCount);
    }
}

/*
 *  Formula: Box-Muller
 *   X = sqrt(-2 * ln(U1)) * cos(2 * PI * U2)
 *   Y = sqrt(-2 * ln(U1)) * sin(2 * PI * U2)
 */
template <typename T>
__aicore__ inline void BoxMullerFloatSIMD(
    LocalTensor<float>& yOutputTmp, LocalTensor<float> v1Result, LocalTensor<float> u2Result, const uint32_t calCount)
{
    __ubuf__ float* uniformRes = (__ubuf__ float*)yOutputTmp.GetPhyAddr();
    __ubuf__ float* ubV1Out = (__ubuf__ float*)v1Result.GetPhyAddr();
    __ubuf__ float* ubU2Out = (__ubuf__ float*)u2Result.GetPhyAddr();
    uint32_t repeatTimes =
        Ops::Base::CeilDiv(calCount / DOUBLE_UNIFORM_RESULT, static_cast<uint32_t>(INT32_FLOAT32_ONE_REPEAT));
    uint32_t sreg1 = calCount;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vreg0;
        MicroAPI::RegTensor<float> vreg1;
        MicroAPI::RegTensor<float> vreg2;
        MicroAPI::RegTensor<float> vreg3;
        MicroAPI::RegTensor<float> vreg4;
        MicroAPI::RegTensor<float> vreg5;
        MicroAPI::RegTensor<float> vreg6;
        MicroAPI::MaskReg mask;

        float epsScalar = 1.0e-7f;
        float doublePiScalar = DOUBLE_MULTIPLE * PI;
        int32_t offset = static_cast<int32_t>(INT32_FLOAT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<float>(sreg1);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg0, vreg1, uniformRes + i * offset * DOUBLE_UNIFORM_RESULT);
            MicroAPI::Maxs(vreg2, vreg0, epsScalar, mask);
            MicroAPI::Ln(vreg3, vreg2, mask);
            MicroAPI::Muls(vreg4, vreg1, doublePiScalar, mask);
            MicroAPI::Muls(vreg5, vreg3, -DOUBLE_MULTIPLE, mask);
            MicroAPI::Sqrt(vreg6, vreg5, mask);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubV1Out, vreg4, offset, mask);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubU2Out, vreg6, offset, mask);
        }
    }
}

template <typename T>
__aicore__ inline void BoxMullerMulSIMD(
    LocalTensor<float>& yOutputTmp, LocalTensor<float> v1Result, LocalTensor<float> u2Result,
    LocalTensor<float> yOutput, const uint32_t calCount)
{
    Cos<float, false>(yOutputTmp, v1Result);
    Sin<float, false>(yOutput, v1Result);
    uint32_t repeatTimes =
        Ops::Base::CeilDiv(calCount / DOUBLE_UNIFORM_RESULT, static_cast<uint32_t>(INT32_FLOAT32_ONE_REPEAT));

    __ubuf__ float* ubSinResult = (__ubuf__ float*)yOutput.GetPhyAddr();
    __ubuf__ float* ubCosResult = (__ubuf__ float*)yOutputTmp.GetPhyAddr();
    __ubuf__ float* ubU2Result = (__ubuf__ float*)u2Result.GetPhyAddr();
    __ubuf__ float* ubOut = (__ubuf__ float*)v1Result.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vreg0;
        MicroAPI::RegTensor<float> vreg1;
        MicroAPI::RegTensor<float> vreg2;
        MicroAPI::RegTensor<float> vreg3;
        MicroAPI::RegTensor<float> vreg4;
        MicroAPI::RegTensor<float> vreg5;
        MicroAPI::MaskReg mask;

        uint32_t sreg1 = static_cast<uint32_t>(calCount);
        int32_t offset = static_cast<int32_t>(INT32_FLOAT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = MicroAPI::UpdateMask<float>(sreg1);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, ubSinResult, offset);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, ubCosResult, offset);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg2, ubU2Result, offset);

            MicroAPI::Mul(vreg3, vreg0, vreg2, mask);
            MicroAPI::Mul(vreg4, vreg1, vreg2, mask);

            MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                reinterpret_cast<__ubuf__ int32_t*>(ubOut + i * offset * DOUBLE_UNIFORM_RESULT),
                (MicroAPI::RegTensor<int32_t>&)(vreg3), (MicroAPI::RegTensor<int32_t>&)(vreg4), mask);
        }
    }
}

class RandomKernelBaseOp {
public:
    __aicore__ inline RandomKernelBaseOp(const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : tiling_(tilingData){};
    __aicore__ inline void VarsInit();

    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void GenRandomSIMD(LocalTensor<uint32_t> randomLocal, const uint64_t count);

    const RandomUnifiedTilingDataStruct* tiling_;

    // static constexpr uint16_t ALG_KEY_SIZE = 2;
    // static constexpr uint16_t ALG_COUNTER_SIZE = 4;
    // static constexpr uint32_t INT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
    // static constexpr uint32_t RIGHT_SHIFT = 32;

    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    uint32_t blockIdx_;
    int64_t curCoreProNum_ = 0;
    int64_t ubRepeatimes_ = 0;
};

__aicore__ inline void RandomKernelBaseOp::VarsInit()
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ > tiling_->usedCoreNum) {
        return;
    }

    if (blockIdx_ == tiling_->usedCoreNum - 1) {
        curCoreProNum_ = tiling_->tailCoreProNum;
    } else {
        curCoreProNum_ = tiling_->normalCoreProNum;
    }

    ubRepeatimes_ = Ops::Base::CeilDiv(curCoreProNum_, tiling_->singleBufferSize);
    // InitKeyAndCounter
    for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tiling_->key[i];
    }
    for (uint32_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tiling_->counter[i];
    }

    /*    算子自管理
        GlobalTensor   SetGlobalBuffer
        Que  InitBuffer
    */
}

__aicore__ inline void RandomKernelBaseOp::Skip(const uint64_t count)
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

__aicore__ inline void RandomKernelBaseOp::GenRandomSIMD(LocalTensor<uint32_t> randomLocal, const uint64_t count)
{
    PhiloxRandom<10>(randomLocal, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]}, count);
}

} // namespace RandomKernelBase
#endif 