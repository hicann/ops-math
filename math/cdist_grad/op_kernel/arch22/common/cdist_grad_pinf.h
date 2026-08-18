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
 * \file cdist_grad_pinf.h
 * \brief CdistGrad p=inf — replicates 950 CdistGradInfDag exactly.
 *
 *   result = grad * sign(diff) * (|diff| == cdist ? 1 : 0)
 *   sign = hard sign(diff) (Compare GT/LT + Select)
 *   mask = (|diff| - cdist) == 0  (Sub + CompareScalar EQ, exact equality)
 */

#ifndef CDIST_GRAD_PINF_H
#define CDIST_GRAD_PINF_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradPInf : public CdistGradBase<T, CdistGradPInf<T>> {
public:
    using Base = CdistGradBase<T, CdistGradPInf<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeForJ(int64_t j);
};

template <typename T>
__aicore__ inline void CdistGradPInf<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // no chunk-level preprocessing needed
}

template <typename T>
__aicore__ inline void CdistGradPInf<T>::ComputeForJ(int64_t j)
{
    uint32_t count = static_cast<uint32_t>(this->mAligned_);
    int64_t rowOff = j * this->mAligned_;
    LocalTensor<float> diff = this->diffBuf.template Get<float>();
    LocalTensor<float> sign = this->signBuf.template Get<float>();
    LocalTensor<uint8_t> mask = this->maskBuf2.template Get<uint8_t>();

    // diff = x1 - x2[j]
    AscendC::Sub(diff, this->x1Row_, this->x2Chunk_[rowOff], count);
    // sign(diff): hard decision (matches 950 CdistGradSignOp)
    AscendC::Compare(mask, diff, this->zero_, AscendC::CMPMODE::GT, count);
    AscendC::Select(sign, mask, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    AscendC::Compare(mask, diff, this->zero_, AscendC::CMPMODE::LT, count);
    AscendC::Select(sign, mask, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

    // |diff| (overwrite diff)
    AscendC::Abs(diff, diff, count);
    // mask = (|diff| - cdist) == 0  (950 CdistGradMaskEQOp: exact equality)
    AscendC::Sub(diff, diff, this->distChunk_[rowOff], count);
    AscendC::Compare(mask, diff, this->zero_, AscendC::CMPMODE::EQ, count);
    AscendC::Select(diff, mask, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

    // result = grad * sign * mask  (950: Mul(Mul(CastGrad, InfSign), InfMask))
    AscendC::Mul(sign, this->gradChunk_[rowOff], sign, count);
    AscendC::Mul(sign, sign, diff, count);
    AscendC::Add(this->accum_, this->accum_, sign, count);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_PINF_H
