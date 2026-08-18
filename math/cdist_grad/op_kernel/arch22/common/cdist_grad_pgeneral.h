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
 * \file cdist_grad_pgeneral.h
 * \brief CdistGrad general p (0 < p < inf, p != 1, p != 2) — matches PyTorch
 *        cdist_backward / pdist_backward general-p logic (arch22).
 *
 * PyTorch splits general p into two backward forms:
 *   - p < 2  (lttdist_calc / lt_two):
 *         sign * |diff|^(p-1) * grad / cdist^(p-1)
 *         zero when cdist==0, or when (|diff|==0 && p<1)
 *   - p > 2  (pdist_calc / p):
 *         diff * |diff|^(p-2) * grad / cdist^(p-1)
 *         zero when cdist==0 only (no |diff|==0 guard)
 *
 * Power via AscendC::Power (float scalar exponent), separate dst (no in-place).
 * Select masks are always written at the aligned BASE of a mask buffer (VSEL requires an
 * aligned mask address) — per-row Compare, never offset-indexed chunk masks.
 */

#ifndef CDIST_GRAD_PGENERAL_H
#define CDIST_GRAD_PGENERAL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradPGeneral : public CdistGradBase<T, CdistGradPGeneral<T>> {
public:
    using Base = CdistGradBase<T, CdistGradPGeneral<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeForJ(int64_t j);
};

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // per-row masks computed in ComputeForJ
}

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::ComputeForJ(int64_t j)
{
    uint32_t count = static_cast<uint32_t>(this->mAligned_);
    int64_t rowOff = j * this->mAligned_;
    LocalTensor<float> diff = this->diffBuf.template Get<float>();
    LocalTensor<float> sign = this->signBuf.template Get<float>();
    LocalTensor<uint8_t> maskDiffZero = this->maskBuf2.template Get<uint8_t>();
    LocalTensor<uint8_t> maskDistZero = this->maskBuf.template Get<uint8_t>();
    LocalTensor<uint8_t> powTmp = this->tmpBuf.template Get<uint8_t>();
    LocalTensor<float> powDst = this->powDstBuf.template Get<float>();
    float pMinus1 = this->pValueF_ - 1.0f;
    float pMinus2 = this->pValueF_ - 2.0f;

    // diff = x1 - x2[j]
    AscendC::Sub(diff, this->x1Row_, this->x2Chunk_[rowOff], count);

    if (this->pValueF_ < 2.0f) {
        // ---- PyTorch lttdist_calc / lt_two (p < 2) ----
        // result = sign * |diff|^(p-1) * grad / cdist^(p-1),
        //          zero when cdist==0 or (|diff|==0 && p<1).
        // sign(diff): hard decision
        AscendC::Compare(maskDiffZero, diff, this->zero_, AscendC::CMPMODE::GT, count);
        AscendC::Select(sign, maskDiffZero, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        AscendC::Compare(maskDiffZero, diff, this->zero_, AscendC::CMPMODE::LT, count);
        AscendC::Select(sign, maskDiffZero, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

        // |diff|, remember |diff|==0
        AscendC::Abs(diff, diff, count);
        AscendC::Compare(maskDiffZero, diff, this->zero_, AscendC::CMPMODE::EQ, count);
        // |diff|^(p-1) — Power to a separate dst
        AscendC::Power(powDst, diff, pMinus1, powTmp, count);
        // sign * |diff|^(p-1)
        AscendC::Mul(diff, sign, powDst, count);
        // * grad
        AscendC::Mul(diff, this->gradChunk_[rowOff], diff, count);
        // cdist^(p-1) — reuse powDst (consumed above)
        AscendC::Power(powDst, this->distChunk_[rowOff], pMinus1, powTmp, count);
        // / cdist^(p-1)
        AscendC::Div(diff, diff, powDst, count);
        // SelectZero(cdist==0)
        AscendC::Compare(maskDistZero, this->distChunk_[rowOff], this->zero_, AscendC::CMPMODE::EQ, count);
        AscendC::Select(diff, maskDistZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        // SelectZero(|diff|==0): only for p<1 (0^(p-1)=inf). For 1<=p<2 the guard is a
        // mathematical no-op — kept to mirror PyTorch blendv (diff==0) && (p<1).
        if (this->pValueF_ < 1.0f) {
            AscendC::Select(diff, maskDiffZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
    } else {
        // ---- PyTorch pdist_calc / p (p > 2) ----
        // result = diff * |diff|^(p-2) * grad / cdist^(p-1),
        //          zero when cdist==0 only (no |diff|==0 guard, matches PyTorch).
        // |diff| into sign buffer (diff keeps its sign for the multiply)
        AscendC::Abs(sign, diff, count);
        // |diff|^(p-2)
        AscendC::Power(powDst, sign, pMinus2, powTmp, count);
        // diff * |diff|^(p-2)
        AscendC::Mul(diff, diff, powDst, count);
        // * grad
        AscendC::Mul(diff, this->gradChunk_[rowOff], diff, count);
        // cdist^(p-1) — reuse powDst (consumed above)
        AscendC::Power(powDst, this->distChunk_[rowOff], pMinus1, powTmp, count);
        // / cdist^(p-1)
        AscendC::Div(diff, diff, powDst, count);
        // SelectZero(cdist==0)
        AscendC::Compare(maskDistZero, this->distChunk_[rowOff], this->zero_, AscendC::CMPMODE::EQ, count);
        AscendC::Select(diff, maskDistZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }
    // accum += result
    AscendC::Add(this->accum_, this->accum_, diff, count);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_PGENERAL_H
