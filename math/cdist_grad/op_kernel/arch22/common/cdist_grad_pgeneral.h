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
 * \brief CdistGrad general p (0 < p < inf, p != 1, p != 2) — replicates 950 CdistGradDag /
 *        CdistGradLargePDag.
 *
 *   result = sign * |diff|^(p-1) * grad / cdist^(p-1)
 *            then SelectZero(cdist==0), SelectZero(|diff|==0)
 *
 * High-precision reformulation for fp32:
 *   result = sign * grad * r^q,   q = |p-1|,   r = |diff|/cdist (p>1) or cdist/|diff| (p<1).
 * This collapses the two transcendental `Power` calls into a single power of a bounded ratio,
 * and replaces the low-precision arch22 `Div` (reciprocal-table) with a 2-step Newton-Raphson
 * reciprocal refinement. The remaining power r^q is computed as exp(q * ln(r)) with a
 * Newton-refined ln (2 iterations) and the Taylor-series Exp, driving fp32 relative error
 * from ~1e-4 (bare vln+vexp) down to ~1e-7.
 *
 * Select masks are always written at the aligned BASE of a mask buffer (VSEL requires an
 * aligned mask address) — per-row Compare, never offset-indexed chunk masks.
 */

#ifndef CDIST_GRAD_PGENERAL_H
#define CDIST_GRAD_PGENERAL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/math/log.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

// Largest finite fp32. Used to detect a cdist that overflowed to +inf, which happens for
// every fp16 case whose true distance exceeds 65504 (the reference computes cdist in fp64
// and rounds it to the operator dtype).
constexpr float MAX_FINITE_F32 = 3.4028235e38f;

// Number of low significand bits cleared by SplitHead. fp32 carries 24 significand bits, so
// splitting at half that width leaves two 12-bit heads whose product is still exact in fp32.
constexpr int32_t SPLIT_HEAD_SHIFT_BITS = 12;

// Largest integer exponent still served by PowIntExp. Up to this many repeated multiplies stay
// cheaper than the exp/ln path and, unlike it, are exact; beyond it the multiply chain wins on
// neither count.
constexpr int64_t POW_INT_FASTPATH_MAX_EXP = 16;

// dst = src^expInt via repeated multiplication for small integer expInt (exact, no exp/ln).
__aicore__ inline void PowIntExp(LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t expInt, uint32_t count)
{
    AscendC::Adds(dst, src, 0.0f, count); // dst = src
    for (int64_t k = 1; k < expInt; ++k) {
        AscendC::Mul(dst, src, dst, count); // dst = src * dst
    }
}

// hi = src with the low 12 significand bits cleared, leaving a 12-bit head. The product of
// two such heads is exact in fp32 (12 + 12 = 24 bits), which is what makes the two-product
// below exact. src must be non-negative -- every operand here is a magnitude -- so the shift
// pair acts as a plain mantissa mask. The textbook Dekker split (src * 4097) is deliberately
// avoided: it overflows to +inf, and then to NaN, for |src| > FLT_MAX/4097 ~ 8.3e34.
// hiI is an int32 scratch row; the returned head is its float view.
__aicore__ inline void SplitHead(LocalTensor<int32_t>& hiI, const LocalTensor<float>& src, uint32_t count)
{
    LocalTensor<float> s = src;
    AscendC::ShiftRight<int32_t>(hiI, s.template ReinterpretCast<int32_t>(), SPLIT_HEAD_SHIFT_BITS, count);
    AscendC::ShiftLeft<int32_t>(hiI, hiI, SPLIT_HEAD_SHIFT_BITS, count);
}

// dst = num / den, correctly rounded to fp32 (<= 0.5 ulp) and EXACT whenever num == den.

// Two Newton-Raphson reciprocal refinements on their own leave up to ~3 ulp and, in
// particular, return num/num == 1.0f only about 60% of the time. That matters far more than
// the raw ulp count suggests, because the quotient feeds ln(): its RELATIVE error lands as
// an ABSOLUTE error on ln(r) and is then multiplied by q in exp(q*ln r), so at q ~ 4.8 a
// single ulp of division error becomes ~3e-7 of relative error per term -- and the sum over
// j cancels by up to 1e6:1, which lifts that straight into the output. It is worst exactly
// where the reference is perfect: whenever the feature dim is 1, cdist is |diff| bit for
// bit, so the true factor is 1 and the CPU benchmark (exp(q*log(1))) gets every such term
// exactly right while we contribute a fresh ulp on each one.

// So the quotient is finished with one residual correction, q1 = q0 + (num - den*q0)/den,
// where den*q0 is formed exactly as a two-product (head/tail split above). num - den*q0 is
// then exact by Sterbenz, and the correction only needs the low-precision reciprocal we
// already have.

// tmp supplies 9 fp32 rows of `count` elements; dst may alias num or den (only the final
// Add writes to it).
__aicore__ inline void DivHighPrec(LocalTensor<float>& dst, const LocalTensor<float>& num,
                                   const LocalTensor<float>& den, LocalTensor<float>& tmp, uint32_t count)
{
    uint32_t e = count;
    LocalTensor<float> rcp = tmp;         // 1/den
    LocalTensor<float> q0 = tmp[e];       // first estimate of num/den
    LocalTensor<float> ph = tmp[2 * e];   // fl(den*q0), the high half of the product
    LocalTensor<float> acc = tmp[3 * e];  // den*q0 - ph exactly, then the residual
    LocalTensor<float> prod = tmp[4 * e]; // partial products
    LocalTensor<float> denLo = tmp[5 * e];
    LocalTensor<float> q0Lo = tmp[6 * e];
    LocalTensor<int32_t> denHiI = tmp[7 * e].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> q0HiI = tmp[8 * e].template ReinterpretCast<int32_t>();
    LocalTensor<float> denHi = tmp[7 * e];
    LocalTensor<float> q0Hi = tmp[8 * e];

    AscendC::Reciprocal(rcp, den, count); // r0 = 1/den
    AscendC::Mul(ph, den, rcp, count);    // den*r0
    AscendC::Muls(ph, ph, -1.0f, count);  // -den*r0
    AscendC::Adds(ph, ph, 2.0f, count);   // 2 - den*r0
    AscendC::Mul(rcp, rcp, ph, count);    // r1 = r0*(2 - den*r0)
    AscendC::Mul(ph, den, rcp, count);    // den*r1
    AscendC::Muls(ph, ph, -1.0f, count);  // -den*r1
    AscendC::Adds(ph, ph, 2.0f, count);   // 2 - den*r1
    AscendC::Mul(rcp, rcp, ph, count);    // r2 = r1*(2 - den*r1)
    AscendC::Mul(q0, num, rcp, count);    // q0 = fl(num * r2)

    // Two-product: den*q0 == ph + acc exactly.
    AscendC::Mul(ph, den, q0, count);
    SplitHead(denHiI, den, count);
    AscendC::Sub(denLo, den, denHi, count);
    SplitHead(q0HiI, q0, count);
    AscendC::Sub(q0Lo, q0, q0Hi, count);
    AscendC::Mul(acc, denHi, q0Hi, count);
    AscendC::Sub(acc, acc, ph, count);
    AscendC::Mul(prod, denHi, q0Lo, count);
    AscendC::Add(acc, acc, prod, count);
    AscendC::Mul(prod, denLo, q0Hi, count);
    AscendC::Add(acc, acc, prod, count);
    AscendC::Mul(prod, denLo, q0Lo, count);
    AscendC::Add(acc, acc, prod, count); // acc = den*q0 - ph

    // residual = num - den*q0 = (num - ph) - acc, exact; q1 = q0 + residual/den.
    AscendC::Sub(prod, num, ph, count);
    AscendC::Sub(prod, prod, acc, count);
    AscendC::Mul(prod, prod, rcp, count);
    AscendC::Add(dst, q0, prod, count);
}

// dst = exp(x) to ~1 ulp of fp32, via Cody-Waite range reduction:
//   m = floor(x*log2e);  c = (x - m*LN2_HI) - m*LN2_LO;  exp(x) = 2^m * e^c, c in [0, ln2).
//   ln2 is split into a 14-significant-bit head and the remainder so that m*LN2_HI is EXACT
//   in fp32 (|m| < 512 here), which makes c carry no error from the reduction. The earlier
//   `c = frac(x*log2e)*ln2` form instead inherited the rounding of x*log2e as an ABSOLUTE
//   error of |x|*2^-24 on c -- 1.2e-7 already at |x| = 2, and ~5e-6 at |x| = 80. That error
//   showed up as a relative error on exp, and LnHighPrec's Newton step turns exp's relative
//   error directly into ln's absolute error, so it was then multiplied by q in exp(q*ln r):
//   it dominated the fp32 accuracy of the whole r^q chain.
//   e^c is evaluated as 1 + expm1(c) with expm1 in Horner form from the tail, so the leading
//   1 absorbs a single rounding instead of the 11 sequential ones of a forward Taylor sum.
//   Truncating after c^9/9! leaves 7.7e-9 relative, well below one fp32 ulp.
// z / m / g are distinct fp32 scratch rows; i32 is an int32 row (holds the 2^m bits).
__aicore__ inline void ExpHighPrec(LocalTensor<float>& dst, const LocalTensor<float>& x, LocalTensor<float>& z,
                                   LocalTensor<float>& m, LocalTensor<float>& g, LocalTensor<int32_t>& i32,
                                   uint32_t count)
{
    constexpr float LOG2E = 1.4426950408889634f;      // 1/ln2
    constexpr float LN2_HI = 0.693145751953125f;      // ln2 head, 14 significant bits
    constexpr float LN2_LO = 1.4286067653301868e-06f; // ln2 - LN2_HI

    AscendC::Muls(z, x, LOG2E, count);                                        // z ~ x/ln2
    AscendC::Cast<float, float>(m, z, AscendC::RoundMode::CAST_FLOOR, count); // m = floor(z)
    AscendC::Muls(g, m, LN2_HI, count);                                       // m*LN2_HI (exact)
    AscendC::Sub(z, x, g, count);                                             // x - m*LN2_HI
    AscendC::Muls(g, m, LN2_LO, count);                                       // m*LN2_LO
    AscendC::Sub(g, z, g, count);                                             // c in [0, ln2)

    // expm1(c) = c*(1/1! + c*(1/2! + c*(... + c/9!)))
    AscendC::Duplicate(dst, 2.7557319e-06f, count); // 1/9!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 2.4801587e-05f, count); // 1/8!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.9841270e-04f, count); // 1/7!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.3888889e-03f, count); // 1/6!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 8.3333333e-03f, count); // 1/5!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 4.1666667e-02f, count); // 1/4!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.6666667e-01f, count); // 1/3!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 5.0e-01f, count); // 1/2!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.0f, count); // 1/1!
    AscendC::Mul(dst, dst, g, count);     // expm1(c)
    AscendC::Adds(dst, dst, 1.0f, count); // e^c

    // 2^m by exponent-bit construction. Clamp m first so an out-of-range exponent saturates
    // instead of aliasing into the sign bit: m <= -127 -> bits 0 -> +0 (underflow to zero),
    // m >= 128 -> 0x7F800000 -> +inf (overflow). Without the clamp, a strongly negative
    // q*ln(r) produced a garbage float rather than 0.
    AscendC::Maxs(m, m, -127.0f, count);
    AscendC::Mins(m, m, 128.0f, count);
    AscendC::Cast<int32_t, float>(i32, m, AscendC::RoundMode::CAST_FLOOR, count);
    AscendC::Adds<int32_t>(i32, i32, static_cast<int32_t>(127), count);     // m + 127
    AscendC::ShiftLeft<int32_t>(i32, i32, static_cast<int32_t>(23), count); // (m+127)<<23
    LocalTensor<float> pw2 = i32.template ReinterpretCast<float>();         // 2^m exact
    AscendC::Mul(dst, dst, pw2, count);                                     // e^c * 2^m
}

// res = ln(x) via two Newton refinements using an accurate exp:  ln' = ln + (x*e^{-ln} - 1).
// The fixed point of that iteration is limited by the RELATIVE error of ExpHighPrec, which it
// turns into the ABSOLUTE error of res -- hence the care taken over the range reduction there.
// res / neg / aux are distinct fp32 scratch rows; z/m/g/i32 are the ExpHighPrec scratch.
__aicore__ inline void LnHighPrec(LocalTensor<float>& res, const LocalTensor<float>& x, LocalTensor<float>& neg,
                                  LocalTensor<float>& aux, LocalTensor<float>& z, LocalTensor<float>& m,
                                  LocalTensor<float>& g, LocalTensor<int32_t>& i32, uint32_t count)
{
    AscendC::Log(res, x, count);                // res = ln(x) seed
    AscendC::Muls(neg, res, -1.0f, count);      // neg = -ln0
    ExpHighPrec(aux, neg, z, m, g, i32, count); // e^{-ln0}
    AscendC::Mul(aux, x, aux, count);           // x * e^{-ln0}
    AscendC::Adds(aux, aux, -1.0f, count);      // x*e^{-ln0} - 1
    AscendC::Add(res, res, aux, count);         // ln1
    AscendC::Muls(neg, res, -1.0f, count);      // -ln1
    ExpHighPrec(aux, neg, z, m, g, i32, count); // e^{-ln1}
    AscendC::Mul(aux, x, aux, count);           // x * e^{-ln1}
    AscendC::Adds(aux, aux, -1.0f, count);      // x*e^{-ln1} - 1
    AscendC::Add(res, res, aux, count);         // ln2 (high precision)
}

// dst = base^exp for exp >= 0. Integer fast-path (exact); otherwise exp(exp * ln(base)).
// lnBuf / negBuf / auxBuf / baseBuf / zBuf / mBuf / gBuf are distinct fp32 scratch rows;
// i32Buf is an int32 scratch row.
__aicore__ inline void PowGeneral(LocalTensor<float>& dst, const LocalTensor<float>& base, float exp,
                                  LocalTensor<float>& lnBuf, LocalTensor<float>& negBuf, LocalTensor<float>& auxBuf,
                                  LocalTensor<float>& baseBuf, LocalTensor<float>& zBuf, LocalTensor<float>& mBuf,
                                  LocalTensor<float>& gBuf, LocalTensor<int32_t>& i32Buf, uint32_t count)
{
    int64_t eInt = static_cast<int64_t>(exp);
    if (exp == static_cast<float>(eInt) && eInt >= 1 && eInt <= POW_INT_FASTPATH_MAX_EXP) {
        PowIntExp(dst, base, eInt, count);
        return;
    }
    AscendC::Adds(baseBuf, base, 1e-30f, count);                                 // clamp to avoid ln(0)
    LnHighPrec(lnBuf, baseBuf, negBuf, auxBuf, zBuf, mBuf, gBuf, i32Buf, count); // ln(base)
    AscendC::Muls(lnBuf, lnBuf, exp, count);                                     // exp * ln(base)
    ExpHighPrec(dst, lnBuf, zBuf, mBuf, gBuf, i32Buf, count);                    // base^exp
}

template <typename T>
class CdistGradPGeneral : public CdistGradBase<T, CdistGradPGeneral<T>> {
public:
    using Base = CdistGradBase<T, CdistGradPGeneral<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeForJ(int64_t j);
    __aicore__ inline void ResetAccumCompensation();
    __aicore__ inline void FoldAccumCompensation();
};

// Compensated accumulation, tmp rows [9e, 13e).

// The reduce over j cancels catastrophically on these shapes -- measured up to 2.6e6:1 -- so
// once J gets past a few hundred terms it is the fp32 accumulator's OWN rounding, not the term
// math, that sets the error floor. Summing the exact terms in fp32 sequentially already
// reproduces essentially all of the CPU benchmark's error on such a case, which is why simply
// making the terms more accurate cannot get us below it.

// The other paths (p = 0/1/2/inf) compute a term that is bit-identical to the benchmark's, so
// their plain accumulate reproduces the benchmark error exactly and there is nothing to win.
// Only here do our terms differ from the reference's (it evaluates pow(|diff|,q)/pow(cdist,q)
// for 1 < p < 2 while we evaluate exp(q*ln(|diff|/cdist))), so our summation error compounds
// with a term difference instead of cancelling against it.

// Knuth's two-sum is used rather than Kahan's: it is branch-free, needs no mask register, and
// its residual is exact even when the running sum is smaller than the addend -- which is
// exactly what heavy cancellation produces.
//   s2 = s + x;  bv = s2 - s;  residual = (s - (s2 - bv)) + (x - bv)
// The residual is carried in a separate row and folded back once per M-segment. Row 9e must
// therefore stay live across every ComputeForJ of the segment, which is why the compensation
// sits above the [0, 9e) block that DivHighPrec and PowGeneral recycle.

// ONLY for p > 1. The two regimes are genuinely opposed, and which error dominates flips with
// the sign of p-1:
//   p > 1: r = |diff|/cdist <= 1, so every term is bounded by |grad| and the terms sit within
//          an ulp or two of each other. The fp32 accumulator's rounding is then the dominant
//          error and compensating it drops us onto the term floor -- measured 3.4e-6 -> 5.8e-7
//          on a 255-term reduce.
//   p < 1: r = cdist/|diff| >= 1 and unbounded, so the terms are far larger than their sum
//          (median sum|term|/|out| ~ 30) and it is the fp32 REPRESENTATION of each term, not
//          the accumulation, that dominates -- measured term floor 1.0e-2 against a plain-sum
//          error of 2.5e-3. Plain summation beats its own term floor because adding a term to
//          a larger running sum re-rounds it and discards part of that term's own rounding
//          error; the two are anti-correlated. Compensating faithfully preserves the term
//          error instead, and lands on the 4x worse floor. So p < 1 keeps the plain add.
template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::ResetAccumCompensation()
{
    if (this->pValueF_ < 1.0f) {
        return;
    }
    uint32_t count = static_cast<uint32_t>(this->mAligned_);
    LocalTensor<float> tmpF32 = this->tmpBuf.template Get<float>();
    AscendC::Duplicate(tmpF32[9 * count], 0.0f, count);
}

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::FoldAccumCompensation()
{
    if (this->pValueF_ < 1.0f) {
        return;
    }
    uint32_t count = static_cast<uint32_t>(this->mAligned_);
    LocalTensor<float> tmpF32 = this->tmpBuf.template Get<float>();
    AscendC::Add(this->accum_, this->accum_, tmpF32[9 * count], count);
}

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
    LocalTensor<float> powDst = this->powDstBuf.template Get<float>();
    LocalTensor<uint8_t> maskDiffZero = this->maskBuf2.template Get<uint8_t>();
    LocalTensor<uint8_t> maskDistZero = this->maskBuf.template Get<uint8_t>();
    float pMinus1 = this->pValueF_ - 1.0f;
    float q = pMinus1 >= 0.0f ? pMinus1 : -pMinus1; // |p-1|

    // diff = x1 - x2[j]
    AscendC::Sub(diff, this->x1Row_, this->x2Chunk_[rowOff], count);
    // sign(diff): hard decision
    AscendC::Compare(maskDiffZero, diff, this->zero_, AscendC::CMPMODE::GT, count);
    AscendC::Select(sign, maskDiffZero, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    AscendC::Compare(maskDiffZero, diff, this->zero_, AscendC::CMPMODE::LT, count);
    AscendC::Select(sign, maskDiffZero, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

    // |diff|, remember |diff|==0
    AscendC::Abs(diff, diff, count);
    AscendC::Compare(maskDiffZero, diff, this->zero_, AscendC::CMPMODE::EQ, count);

    // High-precision scratch carved from tmpBuf (e = mAligned_ fp32 element count).
    LocalTensor<uint8_t> tmpU8 = this->tmpBuf.template Get<uint8_t>();
    LocalTensor<float> tmpF32 = this->tmpBuf.template Get<float>();
    // Rows [0, 9e) are used first by DivHighPrec and then, once it has returned, by
    // PowGeneral -- the two never hold a live value at the same time.
    uint32_t e = count;
    LocalTensor<float> baseBuf = tmpF32[e];                                             // [e, 2e)
    LocalTensor<float> lnBuf = tmpF32[2 * e];                                           // [2e, 3e)
    LocalTensor<float> negBuf = tmpF32[3 * e];                                          // [3e, 4e)
    LocalTensor<float> auxBuf = tmpF32[4 * e];                                          // [4e, 5e)
    LocalTensor<float> zBuf = tmpF32[5 * e];                                            // [5e, 6e)
    LocalTensor<float> mBuf = tmpF32[6 * e];                                            // [6e, 7e)
    LocalTensor<float> gBuf = tmpF32[7 * e];                                            // [7e, 8e)
    LocalTensor<int32_t> i32Buf = tmpU8[8 * e * 4].template ReinterpretCast<int32_t>(); // [8e, 9e)

    // r = |diff|/cdist (p>1, r<=1)  or  cdist/|diff| (p<1, r>=1) — high-precision division.
    if (pMinus1 >= 0.0f) {
        DivHighPrec(diff, diff, this->distChunk_[rowOff], tmpF32, count);
    } else {
        // p < 1 puts cdist in the NUMERATOR, so a +inf cdist makes r = +inf and the power
        // evaluates to NaN (ln(+inf) drives floor(-inf) - (-inf) through the range reduction).
        // That is deliberate and must not be "fixed" here: the true gradient does diverge in
        // this regime, and the CPU reference mirrors it exactly -- see the p < 1.0 branch of
        // executor.py, which forces ln(inf) to NaN specifically to stay aligned with us.
        // Saturating r instead would return a huge finite value that the fp16 output cast
        // then turns into +-inf, which no longer matches the reference.
        DivHighPrec(diff, this->distChunk_[rowOff], diff, tmpF32, count);
    }
    // powDst = r^q
    PowGeneral(powDst, diff, q, lnBuf, negBuf, auxBuf, baseBuf, zBuf, mBuf, gBuf, i32Buf, count);
    // sign * r^q * grad
    AscendC::Mul(diff, sign, powDst, count);
    AscendC::Mul(diff, this->gradChunk_[rowOff], diff, count);
    // SelectZero(cdist==0)
    AscendC::Compare(maskDistZero, this->distChunk_[rowOff], this->zero_, AscendC::CMPMODE::EQ, count);
    AscendC::Select(diff, maskDistZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    if (pMinus1 >= 0.0f) {
        // SelectZero(cdist==+inf). |diff| is always finite (both operands come from a dtype
        // no wider than fp32), so for p > 1 the exact term is (|diff|/inf)^(p-1) == 0, which
        // is what the reference produces. The arithmetic cannot get there on its own:
        // Reciprocal(+inf) is 0 and DivHighPrec's Newton step then evaluates inf*0 = NaN,
        // which propagates through ln/exp and poisons every element of the output row.
        // Note that clamping cdist to MAX_FINITE_F32 instead is NOT equivalent: it leaves
        // r ~ 1e-34, and for a small exponent (p = 1.09 -> q = 0.09) r^q is still ~1e-3.
        AscendC::Compares(maskDistZero, this->distChunk_[rowOff], MAX_FINITE_F32, AscendC::CMPMODE::GE, count);
        AscendC::Select(diff, maskDistZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }
    // SelectZero(|diff|==0)
    AscendC::Select(diff, maskDiffZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    // accum += result. For p < 1 a plain add is deliberately more accurate than compensating
    // it -- see ResetAccumCompensation.
    if (pMinus1 < 0.0f) {
        AscendC::Add(this->accum_, this->accum_, diff, count);
        return;
    }
    // Otherwise accumulate carrying the exact rounding residual.
    LocalTensor<float> comp = tmpF32[9 * e];
    LocalTensor<float> s2 = tmpF32[10 * e];
    LocalTensor<float> bv = tmpF32[11 * e];
    LocalTensor<float> av = tmpF32[12 * e];
    AscendC::Add(s2, this->accum_, diff, count); // s2 = s + x
    AscendC::Sub(bv, s2, this->accum_, count);   // bv = s2 - s
    AscendC::Sub(av, s2, bv, count);             // av = s2 - bv
    AscendC::Sub(av, this->accum_, av, count);   // s - av   (part of s lost in s2)
    AscendC::Sub(bv, diff, bv, count);           // x - bv   (part of x lost in s2)
    AscendC::Add(av, av, bv, count);             // residual
    AscendC::Add(comp, comp, av, count);
    AscendC::Adds(this->accum_, s2, 0.0f, count); // s = s2
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_PGENERAL_H
