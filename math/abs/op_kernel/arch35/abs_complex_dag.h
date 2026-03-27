/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file abs_dag_complex.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_ABS_DAG_COMPLEX_H
#define CANN_CUSTOM_OPS_ABS_DAG_COMPLEX_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

namespace AbsOp {
using namespace AscendC;
using namespace Ops::Base;

namespace AbsComplexVf {

template <class T, class U>
struct AbscomplexCustom : public Vec::ElemwiseUnaryOP<T, U> {
    __aicore__ inline AbscomplexCustom(LocalTensor<T>& dst, LocalTensor<U>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo> vSrcReg0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vDstReg0;
        uint32_t sreg = (uint32_t)count;
        MicroAPI::MaskReg preg;
        static constexpr uint32_t repeatStride =
            static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U) * MicroAPI::RegTraitNumTwo.REG_NUM);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, repeatStride));
        __ubuf__ U* srcAddr = (__ubuf__ U*)(src.GetPhyAddr());
        __ubuf__ T* dstAddr = (__ubuf__ T*)(dst.GetPhyAddr());
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<U, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(vSrcReg0, srcAddr + i * repeatStride);
                MicroAPI::Abs<
                    T, U, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne>,
                    MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>>(vDstReg0, vSrcReg0, preg);
                MicroAPI::StoreAlign(dstAddr + i * repeatStride, vDstReg0, preg);
            }
        }
#endif
    }
};
} // namespace AbsComplexVf

template <typename U, typename T = float>
struct AbscomplexDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpResult = Bind<AbsComplexVf::AbscomplexCustom<T, U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace AbsOp
#endif // CANN_CUSTOM_OPS_ABS_COMPLEX_DAG_H