/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_all_dag.h
 * \brief reduce all dag
 */

#ifndef REDUCE_ALL_DAG_H
#define REDUCE_ALL_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#ifdef __CCE_AICORE__
#include "op_kernel/math_util.h"
#endif

namespace ReduceAll
{
using namespace Ops::Base;
using OutDtype = uint8_t;
using OutDtype2 = int32_t;

template <typename T>
struct CastZeroOne : public Vec::ElemwiseUnaryOP<T,T> {
    __aicore__ inline CastZeroOne(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_SIZE = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* srcAddr = (__local_mem__ T*)src.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_SIZE);
        uint32_t InSize = count;
        uint32_t OutSize = count;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> srcReg;
            MicroAPI::RegTensor<T> dstReg;
            MicroAPI::MaskReg cmpMask;
            MicroAPI::MaskReg InMask;
            MicroAPI::MaskReg OutMask;
            for (uint16_t j = 0; j < loopTimes; j++) {
                InMask = MicroAPI::UpdateMask<T>(InSize);
                uint32_t tmp = OutSize - InSize;
                OutSize = OutSize - tmp;
                OutMask = MicroAPI::UpdateMask<T>(tmp);
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcReg, srcAddr, VL_SIZE);
                MicroAPI::Compares<T, CMPMODE::NE>(cmpMask, srcReg, 0.0f, InMask);
                MicroAPI::Duplicate(dstReg, static_cast<T>(0.0f));
                MicroAPI::Duplicate(dstReg, static_cast<T>(1.0f), cmpMask);
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, dstReg, VL_SIZE, OutMask);
            }
        }
#endif
    }
};

template <typename T, typename PromteT>
struct ReduceAllDagFloat {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<CastZeroOne<T>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<PromteT, T, 0>, Cast0>;
    using ReduceOp0 = Bind<Vec::ReduceAllOp<PromteT>, Cast1>;
    using Cast2 = Bind<Vec::Cast<OutDtype2, PromteT, 1>, ReduceOp0>;
    using Cast3 = Bind<Vec::Cast<OutDtype, OutDtype2, 0>, Cast2>;
    using OpCopyOut = Bind<Vec::CopyOut<OutDtype>, Placeholder::Out0<OutDtype>, Cast3>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct ReduceAllDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceAllOp<PromteT>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ReduceAll

#endif