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
 * \file cdist_grad_p2.h
 * \brief CdistGrad p=2 (Euclidean) — replicates 950 CdistGradP2Dag exactly.
 *
 *   result = grad * diff / (cdist + 1e-38) * (cdist != 0 ? 1 : 0)
 *   (950: Mul(CastGrad, OpDiff) -> Div(numer, SafeCdist=Eps 1e-38) -> Mul(MaskNEZero))
 *
 * Fully vectorized on the broadcast [B,P,Q,M] layout. Select masks are always written at
 * the aligned BASE of a mask buffer (VSEL requires an aligned mask address).
 */

#ifndef CDIST_GRAD_P2_H
#define CDIST_GRAD_P2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradP2 : public CdistGradBase<T, CdistGradP2<T>> {
public:
    using Base = CdistGradBase<T, CdistGradP2<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeForJ(int64_t j);
};

template <typename T>
__aicore__ inline void CdistGradP2<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // per-row masks computed in ComputeForJ
}

template <typename T>
__aicore__ inline void CdistGradP2<T>::ComputeForJ(int64_t j)
{
    uint32_t count = static_cast<uint32_t>(this->mAligned_);
    int64_t rowOff = j * this->mAligned_;
    LocalTensor<float> diff = this->diffBuf.template Get<float>();
    LocalTensor<uint8_t> maskDistZero = this->maskBuf.template Get<uint8_t>();

    // diff = x1 - x2[j]
    AscendC::Sub(diff, this->x1Row_, this->x2Chunk_[rowOff], count);
    // mask = (cdist == 0) on the RAW queue value (matches 950 MaskNEZero)
    AscendC::Compare(maskDistZero, this->distChunk_[rowOff], this->zero_, AscendC::CMPMODE::EQ, count);
    // safe cdist = cdist + 1e-38 into a SCRATCH buffer (matches 950 Eps). Never write the
    // queue tensor in place: double-buffer slot reuse races the late V-pipe write with the
    // next chunk's MTE2 fill (write-write hazard, corrupts cdist of chunk c+2).
    LocalTensor<float> distSafe = this->powDstBuf.template Get<float>();
    AscendC::Adds(distSafe, this->distChunk_[rowOff], 1e-38f, count);
    // numer = grad * diff   (950: Mul(CastGrad, OpDiff))
    AscendC::Mul(diff, this->gradChunk_[rowOff], diff, count);
    // / (cdist + 1e-38)
    AscendC::Div(diff, diff, distSafe, count);
    // *(cdist != 0): where cdist==0 take 0
    AscendC::Select(diff, maskDistZero, this->zero_, diff, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    // accum += diff
    AscendC::Add(this->accum_, this->accum_, diff, count);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_P2_H
