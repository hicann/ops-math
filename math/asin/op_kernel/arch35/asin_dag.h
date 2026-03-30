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
 * \file asin_dag.h
 * \brief Asin dag
 */

#ifndef ASIN_DAG_H
#define ASIN_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AsinDag {
using namespace Ops::Base;

constexpr int CastModeToFp32 = 0;
constexpr int CastModeToBf16 = 1;

constexpr float NUM_ONE = 1.0f;
constexpr float NEG_ONE = -1.0f;
constexpr float HALF_PI = 1.5707963267948966192313216916398f;
constexpr float BOUNDARY = 0.70710678118654752440084436210485f;

constexpr float kCOEF[8] = {
    1.0f,
    0.16666666666666666666666666666667f,
    0.075f,
    0.04464285714285714285714285714286f,
    0.03038194444444444444444444444444f,
    0.02237215909090909090909090909091f,
    0.01735276442307692307692307692308f,
    0.01396484375f,
};

#ifdef __CCE_AICORE__
using namespace AscendC;

constexpr static AscendC::MicroAPI::CastTrait kAsinCastTraitNone = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_NONE};
constexpr static AscendC::MicroAPI::CastTrait kAsinCastTraitFloor = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_FLOOR};
constexpr static AscendC::MicroAPI::CastTrait kAsinCastTraitRint = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

template <class T>
__aicore__ inline void AsinTaylorComputeInner(
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& dstReg, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& srcReg,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::Muls(dstReg, dstReg, kCOEF[7], mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[6], mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[5], mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[4], mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[3], mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[2], mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[1], mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    AscendC::MicroAPI::Adds(dstReg, dstReg, kCOEF[0], mask);
}

template <class T>
__aicore__ inline void AsinTaylorCompute(
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& dstReg, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& srcReg,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> tmpReg;
    AscendC::MicroAPI::Mul(dstReg, srcReg, srcReg, mask);
    AscendC::MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
    AsinTaylorComputeInner<T>(dstReg, tmpReg, mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
}

template <class T>
__aicore__ inline void AsinTaylorComputeBySquareValue(
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& dstReg, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& srcReg,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::Muls(dstReg, srcReg, NUM_ONE, mask);
    AsinTaylorComputeInner<T>(dstReg, srcReg, mask);
    AscendC::MicroAPI::Sqrt(srcReg, srcReg, mask);
    AscendC::MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
}

template <class T>
__aicore__ inline void CalRes2(
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& resReg, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& srcReg,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> tmpReg;
    AscendC::MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
    AscendC::MicroAPI::Muls(tmpReg, tmpReg, NEG_ONE, mask);
    AscendC::MicroAPI::Adds(tmpReg, tmpReg, NUM_ONE, mask);
    AscendC::MicroAPI::Sqrt(tmpReg, tmpReg, mask);
    AsinTaylorCompute<T>(resReg, tmpReg, mask);
    AscendC::MicroAPI::Muls(resReg, resReg, NEG_ONE, mask);
    AscendC::MicroAPI::Adds(resReg, resReg, HALF_PI, mask);
}

template <class T>
__aicore__ inline void ProcessBranch(
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& resReg1,
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& resReg2, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& tmpReg,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::RegTensor<int32_t, AscendC::MicroAPI::RegTraitNumOne> s32Reg;
    AscendC::MicroAPI::Mins(tmpReg, tmpReg, BOUNDARY, mask);
    AscendC::MicroAPI::Adds(tmpReg, tmpReg, -BOUNDARY, mask);
    AscendC::MicroAPI::Cast<int32_t, T, kAsinCastTraitFloor>(s32Reg, tmpReg, mask);
    AscendC::MicroAPI::Cast<T, int32_t, kAsinCastTraitRint>(tmpReg, s32Reg, mask);
    AscendC::MicroAPI::Muls(tmpReg, tmpReg, NEG_ONE, mask);
    AscendC::MicroAPI::Mul(resReg1, resReg1, tmpReg, mask);
    AscendC::MicroAPI::Muls(tmpReg, tmpReg, NEG_ONE, mask);
    AscendC::MicroAPI::Adds(tmpReg, tmpReg, NUM_ONE, mask);
    AscendC::MicroAPI::Mul(resReg2, resReg2, tmpReg, mask);
    AscendC::MicroAPI::Add(resReg1, resReg1, resReg2, mask);
}

template <class T>
__aicore__ inline void GetSign(
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& dstReg, AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne>& srcReg,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> denominatorReg;
    constexpr float FP16_MAX = 32768.0f;
    constexpr float FP16_MIN = 3.0517578125e-05f;
    constexpr float FP32_MAX = 4611686018427387904.0f;
    constexpr float FP32_MIN = 2.168404344971009e-19f;
    constexpr float kFpMax = sizeof(T) == sizeof(float) ? FP32_MAX : FP16_MAX;
    constexpr float kFpMin = sizeof(T) == sizeof(float) ? FP32_MIN : FP16_MIN;
    AscendC::MicroAPI::Muls(dstReg, srcReg, kFpMax, mask);
    AscendC::MicroAPI::Abs(denominatorReg, dstReg, mask);
    AscendC::MicroAPI::Adds(denominatorReg, denominatorReg, kFpMin, mask);
    AscendC::MicroAPI::Div(dstReg, dstReg, denominatorReg, mask);
}

#endif // __CCE_AICORE__

template <class T>
struct AsinCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline AsinCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t stride = VL;

        __VEC_SCOPE__
        {
            __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> srcReg;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> dstReg;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> resReg1;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> resReg2;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> signReg;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> tmpReg;
            AscendC::MicroAPI::MaskReg mask;

            for (uint16_t i = 0; i < loopNum; ++i) {
                mask = AscendC::MicroAPI::UpdateMask<T>(count);
                AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * stride);

                CalRes2<T>(resReg2, srcReg, mask);

                AscendC::MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
                AsinTaylorComputeBySquareValue<T>(resReg1, tmpReg, mask);

                ProcessBranch<T>(resReg1, resReg2, tmpReg, mask);
                GetSign<T>(signReg, srcReg, mask);
                AscendC::MicroAPI::Mul(dstReg, resReg1, signReg, mask);

                AscendC::MicroAPI::StoreAlign(dstAddr + i * stride, dstReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct AsinOpDirect {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Y = Bind<AsinDag::AsinCustom<T>, InputX>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AsinOpWithCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, AsinDag::CastModeToFp32>, OpCopyIn>;
    using OpAsin = Bind<AsinDag::AsinCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, AsinDag::CastModeToBf16>, OpAsin>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace AsinDag
#endif // ASIN_DAG_H
