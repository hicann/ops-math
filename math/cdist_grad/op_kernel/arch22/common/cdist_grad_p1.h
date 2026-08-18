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
 * \file cdist_grad_p1.h
 * \brief CdistGrad p=1 — replicates 950 CdistGradP1Dag exactly.
 *
 *   result = grad * sign(diff)
 *   sign(x) = 1.0 if x>0, -1.0 if x<0, 0.0 if x==0  (CompareScalar GT/LT + Select, hard decision)
 *
 * Hard sign (NOT the eps-division approximation) to match 950 precision.
 */

#ifndef CDIST_GRAD_P1_H
#define CDIST_GRAD_P1_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradP1 : public CdistGradBase<T, CdistGradP1<T>> {
public:
    using Base = CdistGradBase<T, CdistGradP1<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeForJ(int64_t j);
};

template <typename T>
__aicore__ inline void CdistGradP1<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // no chunk-level preprocessing needed
}

template <typename T>
__aicore__ inline void CdistGradP1<T>::ComputeForJ(int64_t j)
{
    uint32_t count = static_cast<uint32_t>(this->mAligned_);
    int64_t rowOff = j * this->mAligned_;
    LocalTensor<float> diff = this->diffBuf.template Get<float>();
    LocalTensor<float> sign = this->signBuf.template Get<float>();
    LocalTensor<uint8_t> mask = this->maskBuf2.template Get<uint8_t>();

    // diff = x1 - x2[j]
    AscendC::Sub(diff, this->x1Row_, this->x2Chunk_[rowOff], count);
    // sign(diff): x>0 -> 1, x<0 -> -1, x==0 -> 0 (hard decision, matches 950 CdistGradSignOp)
    AscendC::Compare(mask, diff, this->zero_, AscendC::CMPMODE::GT, count);
    AscendC::Select(sign, mask, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    AscendC::Compare(mask, diff, this->zero_, AscendC::CMPMODE::LT, count);
    AscendC::Select(sign, mask, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    // result = grad * sign   (950: Mul(CastGrad, OpSign))
    AscendC::Mul(sign, this->gradChunk_[rowOff], sign, count);
    // accum += result
    AscendC::Add(this->accum_, this->accum_, sign, count);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_P1_H
