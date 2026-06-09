/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_grad_dag.h
 * \brief cdist_grad dag — six DAGs, all Compare+Select replaced by arithmetic
 *
 * Arithmetic replacements:
 *   sign:   diff / (|diff| + eps)              replaces Compare(NE) + Select
 *   mask:   (d + |d| + eps) / (2|d| + eps)     replaces Compare(GE/EQ) + Select
 *   nz_x:   x + eps                             replaces Compare(NE) + Select for safe divisor
 *   mask_nz: x / (x + eps)                      replaces Compare(NE) + Select for zero-out
 *
 * CdistGradP0Dag:      p == 0   → output zeros
 * CdistGradP1Dag:      p == 1   → grad * sign
 * CdistGradP2Dag:      p == 2   → grad * diff / cdist
 * CdistGradDag:        0<p<2    → sign * |diff|^(p-1) * grad / |cdist|^(p-1)
 * CdistGradLargePDag:  p>2      → diff * |diff|^(p-2) * grad / |cdist|^(p-1)
 * CdistGradInfDag:     p==inf   → grad * sign * mask(|diff| >= cdist)
 */

#ifndef CDIST_GRAD_DAG_H
#define CDIST_GRAD_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace CdistGrad {
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;

constexpr int32_t NORM_MODE_GENERAL  = 0;  // 0 < p < 2, p != 1
constexpr int32_t NORM_MODE_INF      = 1;  // p == inf
constexpr int32_t NORM_MODE_LARGE_P  = 2;  // p > 2
constexpr int32_t NORM_MODE_P0       = 3;  // p == 0
constexpr int32_t NORM_MODE_P1       = 4;  // p == 1
constexpr int32_t NORM_MODE_P2       = 5;  // p == 2

// ---------------------------------------------------------------------------
// CdistGradP0Dag — p == 0 → output zeros
// Var<0>: zero_scalar = 0.0
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradP0Dag {
    using OpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastGrad     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInGrad>;

    using OpZero    = Bind<Vec::Muls<PromoteT>, CastGrad, Placeholder::Var<PromoteT, 0>>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, OpZero>;
    using CastOut   = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradP1Dag — p == 1
// sign = diff / (|diff| + eps)   replaces Compare(NE)+Select for nz_diff
// result = grad * sign
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradP1Dag {
    using Eps = MAKE_CONST(PromoteT, 1e-30);

    using OpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastGrad     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInGrad>;

    using OpCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CastX1     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX1>;

    using OpCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using CastX2     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX2>;

    using OpDiff     = Bind<Vec::Sub<PromoteT>, CastX1, CastX2>;
    using OpDiffAbs  = Bind<Vec::Abs<PromoteT>, OpDiff>;
    using SafeAbsDiff = Bind<Vec::Adds<PromoteT>, OpDiffAbs, Eps>;   // |diff| + eps
    using OpSign     = Bind<Vec::Div<PromoteT>, OpDiff, SafeAbsDiff>; // diff / (|diff| + eps)

    using OpRes      = Bind<Vec::Mul<PromoteT>, CastGrad, OpSign>;

    using ReduceOp0  = Bind<Vec::ReduceSumOp<PromoteT>, OpRes>;
    using CastOut    = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut  = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradP2Dag — p == 2
// result = grad * diff / (cdist + eps)
// When cdist = 0: diff = 0, numerator = 0, result = 0 automatically
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradP2Dag {
    using Eps = MAKE_CONST(PromoteT, 1e-30);

    using OpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastGrad     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInGrad>;

    using OpCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CastX1     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX1>;

    using OpCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using CastX2     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX2>;

    using OpCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using CastCdist     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInCdist>;

    using OpDiff      = Bind<Vec::Sub<PromoteT>, CastX1, CastX2>;
    using SafeCdist   = Bind<Vec::Adds<PromoteT>, CastCdist, Eps>;      // cdist + eps
    using OpNumerator = Bind<Vec::Mul<PromoteT>, CastGrad, OpDiff>;     // grad * diff
    using OpResult    = Bind<Vec::Div<PromoteT>, OpNumerator, SafeCdist>; // grad * diff / (cdist+eps)

    using ReduceOp0  = Bind<Vec::ReduceSumOp<PromoteT>, OpResult>;
    using CastOut    = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut  = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradDag — 0 < p < 2, p != 1
// sign = diff / (|diff| + eps)
// safe_diff = |diff| + eps  (prevents log(0))
// safe_cdist = cdist + eps  (prevents log(0) and div-by-0)
// mask_diff = |diff| / (|diff| + eps)  (zero-out when |diff|=0)
// mask_cdist = cdist / (cdist + eps)   (zero-out when cdist=0)
// num = sign * exp(log(safe_diff) * (p-1))
// numerator = num * grad
// denominator = exp(log(safe_cdist) * (p-1))
// result = numerator / denominator * mask_cdist * mask_diff
// Var<0>: power = p - 1
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradDag {
    using DagEps = MAKE_CONST(PromoteT, 1e-30);

    using DagCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using DagCastGrad     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInGrad>;

    using DagCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using DagCastX1     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInX1>;

    using DagCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using DagCastX2     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInX2>;

    using DagCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using DagCastCdist     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInCdist>;

    // diff & abs
    using DagDiff      = Bind<Vec::Sub<PromoteT>, DagCastX1, DagCastX2>;
    using DagDiffAbs   = Bind<Vec::Abs<PromoteT>, DagDiff>;
    using DagSafeAbsDiff = Bind<Vec::Adds<PromoteT>, DagDiffAbs, DagEps>;       // |diff| + eps
    using DagSafeCdist   = Bind<Vec::Adds<PromoteT>, DagCastCdist, DagEps>;       // cdist + eps

    // sign = diff / (|diff| + eps)
    using DagSign      = Bind<Vec::Div<PromoteT>, DagDiff, DagSafeAbsDiff>;

    // masks: 0 when input=0, ~1 otherwise
    using DagMaskDiff    = Bind<Vec::Div<PromoteT>, DagDiffAbs, DagSafeAbsDiff>;  // |diff|/(|diff|+eps)
    using DagMaskCdist   = Bind<Vec::Div<PromoteT>, DagCastCdist, DagSafeCdist>;    // cdist/(cdist+eps)

    // power: (safe_x)^(p-1) via log/exp, Var<0> = p-1
    using DagPowDiff   = Bind<Vec::Exp<PromoteT>,
                          Bind<Vec::Muls<PromoteT>, Bind<Vec::Log<PromoteT>, DagSafeAbsDiff>,
                           Placeholder::Var<PromoteT, 0>>>;
    using DagPowCdist  = Bind<Vec::Exp<PromoteT>,
                          Bind<Vec::Muls<PromoteT>, Bind<Vec::Log<PromoteT>, DagSafeCdist>,
                           Placeholder::Var<PromoteT, 0>>>;

    // num = sign * |diff|^(p-1), numerator = num * grad
    using DagNum       = Bind<Vec::Mul<PromoteT>, DagSign, DagPowDiff>;
    using DagNumerator = Bind<Vec::Mul<PromoteT>, DagNum, DagCastGrad>;

    // res = numerator / denominator
    using DagDivResult = Bind<Vec::Div<PromoteT>, DagNumerator, DagPowCdist>;

    // apply masks: zero-out when cdist=0 or |diff|=0
    using DagMaskedCdist = Bind<Vec::Mul<PromoteT>, DagDivResult, DagMaskCdist>;
    using DagResult      = Bind<Vec::Mul<PromoteT>, DagMaskedCdist, DagMaskDiff>;

    using DagReduceOp0  = Bind<Vec::ReduceSumOp<PromoteT>, DagResult>;
    using DagCastOut    = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, DagReduceOp0>;
    using DagCopyOut  = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, DagCastOut>;

    using Outputs = Elems<DagCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradLargePDag — p > 2
// safe_cdist = cdist + eps
// mask_cdist = cdist / (cdist + eps)
// num = diff * |diff|^(p-2)        (0 when |diff|=0 for p>2)
// numerator = num * grad
// denominator = |cdist|^(p-1)
// result = numerator / denominator * mask_cdist
// Var<0>: power_diff = p - 2
// Var<1>: power_cdist = p - 1
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradLargePDag {
    using LpEps = MAKE_CONST(PromoteT, 1e-30);

    using LpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using LpCastGrad     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInGrad>;

    using LpCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using LpCastX1     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInX1>;

    using LpCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using LpCastX2     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInX2>;

    using LpCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using LpCastCdist     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInCdist>;

    using LpDiff     = Bind<Vec::Sub<PromoteT>, LpCastX1, LpCastX2>;
    using LpDiffAbs  = Bind<Vec::Abs<PromoteT>, LpDiff>;
    using LpSafeAbsDiff = Bind<Vec::Adds<PromoteT>, LpDiffAbs, LpEps>;       // |diff| + eps (prevents log(0))
    using LpSafeCdist  = Bind<Vec::Adds<PromoteT>, LpCastCdist, LpEps>;       // cdist + eps
    using LpMaskCdist  = Bind<Vec::Div<PromoteT>, LpCastCdist, LpSafeCdist>;   // cdist/(cdist+eps)

    // |diff|^(p-2), Var<0> = p-2
    using LpPowDiff  = Bind<Vec::Exp<PromoteT>,
                         Bind<Vec::Muls<PromoteT>, Bind<Vec::Log<PromoteT>, LpSafeAbsDiff>,
                          Placeholder::Var<PromoteT, 0>>>;

    // |cdist|^(p-1), Var<1> = p-1
    using LpPowCdist = Bind<Vec::Exp<PromoteT>,
                         Bind<Vec::Muls<PromoteT>, Bind<Vec::Log<PromoteT>, LpSafeCdist>,
                          Placeholder::Var<PromoteT, 1>>>;

    // num = diff * |diff|^(p-2)
    using LpNum       = Bind<Vec::Mul<PromoteT>, LpDiff, LpPowDiff>;
    // numerator = num * grad
    using LpNumerator = Bind<Vec::Mul<PromoteT>, LpNum, LpCastGrad>;
    // res = numerator / |cdist|^(p-1)
    using LpRawResult = Bind<Vec::Div<PromoteT>, LpNumerator, LpPowCdist>;
    // zero-out when cdist = 0
    using LpResult    = Bind<Vec::Mul<PromoteT>, LpRawResult, LpMaskCdist>;

    using LpReduceOp0  = Bind<Vec::ReduceSumOp<PromoteT>, LpResult>;
    using LpCastOut    = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, LpReduceOp0>;
    using LpCopyOut  = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, LpCastOut>;

    using Outputs = Elems<LpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradInfDag — p == inf
// sign = diff / (|diff| + eps)
// mask = (d + |d| + eps) / (2|d| + eps)  where d = |diff| - cdist
// result = grad * sign * mask
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradInfDag {
    using InfEps = MAKE_CONST(PromoteT, 1e-30);

    using InfCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using InfCastGrad     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInGrad>;

    using InfCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using InfCastX1     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInX1>;

    using InfCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using InfCastX2     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInX2>;

    using InfCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using InfCastCdist     = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInCdist>;

    using InfDiff      = Bind<Vec::Sub<PromoteT>, InfCastX1, InfCastX2>;
    using InfDiffAbs   = Bind<Vec::Abs<PromoteT>, InfDiff>;

    // sign = diff / (|diff| + eps)
    using InfSafeAbsDiff = Bind<Vec::Adds<PromoteT>, InfDiffAbs, InfEps>;
    using InfSign      = Bind<Vec::Div<PromoteT>, InfDiff, InfSafeAbsDiff>;

    // mask: d = |diff| - cdist, mask = (d + |d| + eps) / (2|d| + eps)
    using InfD         = Bind<Vec::Sub<PromoteT>, InfDiffAbs, InfCastCdist>;
    using InfDAbs      = Bind<Vec::Abs<PromoteT>, InfD>;
    using InfTwoAbsD     = Bind<Vec::Add<PromoteT>, InfDAbs, InfDAbs>;
    using InfNumer       = Bind<Vec::Add<PromoteT>, InfD, InfDAbs>;
    using InfNumerEps    = Bind<Vec::Adds<PromoteT>, InfNumer, InfEps>;
    using InfDenom       = Bind<Vec::Adds<PromoteT>, InfTwoAbsD, InfEps>;
    using InfMask      = Bind<Vec::Div<PromoteT>, InfNumerEps, InfDenom>;

    // result = grad * sign * mask
    using InfGradSign  = Bind<Vec::Mul<PromoteT>, InfCastGrad, InfSign>;
    using InfResult    = Bind<Vec::Mul<PromoteT>, InfGradSign, InfMask>;

    using InfReduceOp0   = Bind<Vec::ReduceSumOp<PromoteT>, InfResult>;
    using InfCastOut     = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, InfReduceOp0>;
    using InfCopyOut   = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, InfCastOut>;

    using Outputs = Elems<InfCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace CdistGrad

#endif  // CDIST_GRAD_DAG_H
