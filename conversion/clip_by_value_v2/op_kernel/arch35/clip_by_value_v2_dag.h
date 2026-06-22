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
 * \file clip_by_value_v2_dag.h
 * \brief clip_by_value_v2 dag
 */

#ifndef CLIP_BY_VALUE_V2_DAG_H
#define CLIP_BY_VALUE_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/vec.h"
#include <type_traits>

using namespace Ops::Base;
using namespace AscendC;

namespace ClipByValueV2Op {
template <typename T>
struct ClipByValueV2Fused : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline ClipByValueV2Fused(
        LocalTensor<T>& dst, LocalTensor<T>& x, LocalTensor<T>& clipValueMin, LocalTensor<T>& clipValueMax,
        const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        __local_mem__ T* xAddr = (__local_mem__ T*)x.GetPhyAddr();
        __local_mem__ T* minAddr = (__local_mem__ T*)clipValueMin.GetPhyAddr();
        __local_mem__ T* maxAddr = (__local_mem__ T*)clipValueMax.GetPhyAddr();

        if constexpr (std::is_same_v<T, int64_t>) {
            constexpr uint32_t vl = AscendC::VECTOR_REG_WIDTH_2XVL / sizeof(int64_t);
            const uint16_t loopNum = (count + vl - 1) / vl;

            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> xReg;
                MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> minReg;
                MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> maxReg;
                MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> resReg;
                MicroAPI::MaskReg mask;
                uint32_t remain = count;

                for (uint16_t idx = 0; idx < loopNum; idx++) {
                    const uint32_t offset = idx * vl;
                    mask = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(remain);
                    MicroAPI::DataCopy<int64_t, MicroAPI::LoadDist::DIST_NORM>(xReg, xAddr + offset);
                    MicroAPI::DataCopy<int64_t, MicroAPI::LoadDist::DIST_NORM>(minReg, minAddr + offset);
                    MicroAPI::Max<int64_t, MicroAPI::MaskMergeMode::ZEROING>(resReg, xReg, minReg, mask);
                    MicroAPI::DataCopy<int64_t, MicroAPI::LoadDist::DIST_NORM>(maxReg, maxAddr + offset);
                    MicroAPI::Min<int64_t, MicroAPI::MaskMergeMode::ZEROING>(resReg, resReg, maxReg, mask);
                    MicroAPI::DataCopy<int64_t, MicroAPI::StoreDist::DIST_NORM>(dstAddr + offset, resReg, mask);
                }
            }
        } else {
            constexpr uint32_t vl = AscendC::VECTOR_REG_WIDTH / sizeof(T);
            const uint16_t loopNum = (count + vl - 1) / vl;

            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<T> xReg;
                MicroAPI::RegTensor<T> minReg;
                MicroAPI::RegTensor<T> maxReg;
                MicroAPI::RegTensor<T> resReg;
                MicroAPI::MaskReg mask;
                uint32_t remain = count;

                for (uint16_t idx = 0; idx < loopNum; idx++) {
                    const uint32_t offset = idx * vl;
                    mask = MicroAPI::UpdateMask<T>(remain);
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(xReg, xAddr + offset);
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(minReg, minAddr + offset);
                    MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING>(resReg, xReg, minReg, mask);
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(maxReg, maxAddr + offset);
                    MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING>(resReg, resReg, maxReg, mask);
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + offset, resReg, mask);
                }
            }
        }
#endif
    }
};

template <typename T>
struct ClipByValueV2Compute {
    using OpInputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputMin = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpInputMax = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using OpClipRes = Bind<ClipByValueV2Fused<T>, OpInputX, OpInputMin, OpInputMax>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpClipRes>;
    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};
} // namespace ClipByValueV2Op

#endif // CLIP_BY_VALUE_V2_DAG_H
