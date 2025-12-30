/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file round_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_ROUND_DAG_H
#define CANN_CUSTOM_OPS_ROUND_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace RoundDag {
const uint32_t UINT32_SIGN = 0x80000000;
const uint16_t UINT16_SIGN = 0x8000;
template<class T>
struct RoundCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline RoundCustom(LocalTensor<T> &dst, LocalTensor<T> &src, uint32_t count) {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        if constexpr(std::is_same_v<T, float>) {
            MicroAPI::RegTensor<uint32_t, MicroAPI::RegTraitNumOne> vregOutInt;
            __VEC_SCOPE__ {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                    MicroAPI::Truncate<T, RoundMode::CAST_RINT, MicroAPI::MaskMergeMode::ZEROING>(vregOutput, vregInput, mask);
                    MicroAPI::Duplicate(vregOutInt, UINT32_SIGN, mask);
                    MicroAPI::And(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint32_t> &)vregInput, mask);
                    MicroAPI::Or(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint32_t> &)vregOutput, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), (MicroAPI::RegTensor<T> &)vregOutInt, mask);
                }
            }
        } else {
            MicroAPI::RegTensor<uint16_t, MicroAPI::RegTraitNumOne> vregOutInt;
            __VEC_SCOPE__ {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                    MicroAPI::Truncate<T, RoundMode::CAST_RINT, MicroAPI::MaskMergeMode::ZEROING>(vregOutput, vregInput, mask);
                    MicroAPI::Duplicate(vregOutInt, UINT16_SIGN, mask);
                    MicroAPI::And(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint16_t> &)vregInput, mask);
                    MicroAPI::Or(vregOutInt, vregOutInt, (MicroAPI::RegTensor<uint16_t> &)vregOutput, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), (MicroAPI::RegTensor<T> &)vregOutInt, mask);
                }
            }
        }
#endif
    }
};

template <typename U, typename T = float>
struct RoundInt {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyIn0>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundZero {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpTruncate = Bind<RoundCustom<U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpTruncate>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundNan {
    constexpr static std::int32_t CAST_MODE_RINT = 1;
    using ConstValueZero = MAKE_CONST(T, 0.0);
    using OpDup = Bind<Vec::Duplicate<T>, ConstValueZero>;
    using OpDiv = Bind<Vec::Div<T>, OpDup, OpDup>;
    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpDiv>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundPositiveDecimals {
    constexpr static std::int32_t CAST_MODE_NONE = 0;
    constexpr static std::int32_t CAST_MODE_RINT = 1;

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    using OpDup = Bind<Vec::Duplicate<T>, Placeholder::Var<T, 0>>;
    using OpMul = Bind<Vec::Mul<T>, OpCopyIn0Cast, OpDup>;
    using OpTruncate = Bind<RoundCustom<T>, OpMul>;
    using OpDiv = Bind<Vec::DivHighPrecision<T>, OpTruncate, OpDup>;

    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpDiv>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundNegativeDecimals {
    constexpr static std::int32_t CAST_MODE_NONE = 0;
    constexpr static std::int32_t CAST_MODE_RINT = 1;

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    using OpDup = Bind<Vec::Duplicate<T>, Placeholder::Var<T, 0>>;
    using OpDiv = Bind<Vec::DivHighPrecision<T>, OpCopyIn0Cast, OpDup>;
    using OpTruncate = Bind<RoundCustom<T>, OpDiv>;
    using OpMul = Bind<Vec::Mul<T>, OpTruncate, OpDup>;

    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpMul>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace RoundDag

#endif  // CANN_CUSTOM_OPS_ROUND_DAG_H