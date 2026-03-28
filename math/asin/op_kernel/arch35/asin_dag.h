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
 * \brief Asin dag - 使用自定义 AsinCustom Vf
 *
 * \note Asin 使用 Taylor 展开近似计算，统一在 float 下计算后 cast 回原类型
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
constexpr int CastModeToFp16 = 2;

// AsinCustom: 自定义 Asin Vf，使用 Taylor 展开近似计算（仅 float 版本）
// asin(x) = arcsin(x)
// 当 x ∈ (-2^(-0.5), 2^(-0.5)) 时: asin(x) = x + 1/6*x^3 + 3/40*x^5 + ...
// 当 x ∈ (-1, -2^(-0.5)) 时: asin(x) = PI/2 - arcsin(sqrt(1-x^2))
// 当 x ∈ (2^(-0.5), 1) 时: asin(x) = arcsin(sqrt(1-x^2)) - PI/2
template <class T>
struct AsinCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline AsinCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;

        __VEC_SCOPE__
        {
            __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregInput;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregTmp1;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregTmp2;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregRes1;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregRes2;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregSign;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregOutput;

            AscendC::MicroAPI::MaskReg mask;

            // Taylor 展开系数: asin(x) = x * (1 + x²/6 + 3x⁴/40 + 5x⁶/112 + ...)
            // 使用 Horner 形式: res = (((kCOEF[7])*x² + kCOEF[6])*x² + kCOEF[5])*x² + ...)*x
            constexpr float kCOEF[8] = {
                1.0f,            // kCOEF[0] = 1  (常数项)
                0.166666667f,   // kCOEF[1] = 1/6
                0.075f,         // kCOEF[2] = 3/40
                0.044642859f,   // kCOEF[3] = 5/112
                0.030381944f,   // kCOEF[4] = 35/1152
                0.022369318f,   // kCOEF[5] = 63/2816
                0.017353673f,   // kCOEF[6] = 231/13312
                0.017912578f    // kCOEF[7] = 1431/79872
            };
            constexpr float NEG_ONE = -1.0f;
            constexpr float HALF_PI = 1.57079632679f;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::MicroAPI::UpdateMask<T, AscendC::MicroAPI::RegTraitNumOne>(count);

                // Load input
                AscendC::MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                // 计算 res2 = PI/2 - asin(sqrt(1 - x^2)) -> vregRes2
                // tmp1 = x^2
                AscendC::MicroAPI::Mul(vregTmp1, vregInput, vregInput, mask);
                // tmp2 = 1 - x^2
                AscendC::MicroAPI::Muls(vregTmp2, vregTmp1, NEG_ONE, mask);
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, 1.0f, mask);
                // tmp1 = sqrt(1 - x^2)
                AscendC::MicroAPI::Sqrt(vregTmp1, vregTmp2, mask);

                // Taylor 展开计算 asin(sqrt(1-x^2)) -> vregTmp2
                // t = sqrt(1-x^2), t2 = t*t = 1-x^2 (already in vregTmp2)
                // Horner: asin(t) = t * (1 + t²/6 + 3t⁴/40 + ...) = (((kCOEF[7]*t² + kCOEF[6])*t² + ...)*t
                // 使用 t2 (=1-x²=vregTmp2) 进行 Horner 求值, 最后乘以 t (=vregTmp1)
                AscendC::MicroAPI::Muls(vregTmp2, vregTmp2, kCOEF[7], mask);   // tmp2 = (1-x²) * kCOEF[7]
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[6], mask);  // tmp2 = (1-x²)*kCOEF[7] + kCOEF[6]
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp2, mask);    // tmp2 = ((1-x²)*kCOEF[7] + kCOEF[6]) * (1-x²)
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[5], mask);
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[4], mask);
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[3], mask);
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[2], mask);
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[1], mask);
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregTmp2, vregTmp2, kCOEF[0], mask);
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp2, vregTmp1, mask);  // 最后乘以 t = sqrt(1-x²)
                AscendC::MicroAPI::Muls(vregTmp2, vregTmp2, NEG_ONE, mask);
                AscendC::MicroAPI::Adds(vregRes2, vregTmp2, HALF_PI, mask);

                // 计算 res1 = asin(|x|) Taylor 展开 -> vregRes1
                // Horner: asin(x) = x * (1 + x²/6 + 3x⁴/40 + 5x⁶/112 + ...)
                // = x * (((kCOEF[7]*x² + kCOEF[6])*x² + kCOEF[5])*x² + ...)*x
                // = (((((kCOEF[7])*x² + kCOEF[6])*x² + kCOEF[5])*x² + ...)*x
                AscendC::MicroAPI::Abs(vregTmp1, vregInput, mask);  // tmp1 = |x|
                AscendC::MicroAPI::Mul(vregTmp2, vregTmp1, vregTmp1, mask);  // tmp2 = |x|²
                // Horner evaluation from high index to low, multiplying by x² each time
                AscendC::MicroAPI::Muls(vregRes1, vregTmp2, kCOEF[7], mask);  // res1 = x² * kCOEF[7]
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[6], mask);  // res1 = x²*kCOEF[7] + kCOEF[6]
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp2, mask);    // res1 = (x²*kCOEF[7] + kCOEF[6]) * x²
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[5], mask);  // res1 = (x²*kCOEF[7] + kCOEF[6])*x² + kCOEF[5]
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp2, mask);    // res1 = ...
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[4], mask);
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[3], mask);
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[2], mask);
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[1], mask);
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp2, mask);
                AscendC::MicroAPI::Adds(vregRes1, vregRes1, kCOEF[0], mask);
                AscendC::MicroAPI::Mul(vregRes1, vregRes1, vregTmp1, mask);  // 最后乘以 |x|

                // 计算 sign(x) -> vregSign
                AscendC::MicroAPI::Abs(vregSign, vregInput, mask);
                AscendC::MicroAPI::Adds(vregSign, vregSign, 1e-10f, mask);
                AscendC::MicroAPI::Div(vregSign, vregInput, vregSign, mask);

                // output = sign(x) * res1
                AscendC::MicroAPI::Mul(vregOutput, vregRes1, vregSign, mask);

                // Store output
                AscendC::MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

// float 版本：直接计算
template <typename T>
struct AsinOpDirect {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Y = Bind<AsinDag::AsinCustom<T>, InputX>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// half/bfloat16 版本：cast to float -> compute -> cast back
template <typename T>
struct AsinOpWithCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, CastModeToFp32>, OpCopyIn>;
    using OpAsin = Bind<AsinDag::AsinCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, CastModeToFp16>, OpAsin>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace AsinDag
#endif  // ASIN_DAG_H
