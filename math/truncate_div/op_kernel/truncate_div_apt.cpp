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
 * \file truncate_div_apt.cpp
 * \brief truncate_div kernel
 */

#include "kernel_operator.h"
#include "arch35/truncate_div_dag.h"
#include "arch35/truncate_div_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode, bool canUseMul>
__global__ __aicore__ void truncate_div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, half>::value && std::is_same<DTYPE_X2, float>::value) {
        if constexpr (canUseMul) {
            using OpDag = TruncateDivOp::TruncateDivFloatWithCastScalar<half, float, float>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, y);
        } else {
            using OpDag = TruncateDivOp::TruncateDivFloatWithCast<half, float, float>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        }
    } else if constexpr (std::is_same<DTYPE_X1, half>::value || std::is_same<DTYPE_X1, bfloat16_t>::value) {
        if constexpr (std::is_same<DTYPE_X2, DTYPE_X1>::value) {
            if constexpr (canUseMul) {
                using OpDag = TruncateDivOp::TruncateDivFloat16Scalar<DTYPE_X1, float>::OpDag;
                BroadcastSch<schMode, OpDag> sch(tiling);
                sch.Process(x1, y);
            } else {
                using OpDag = TruncateDivOp::TruncateDivFloat16<DTYPE_X1, float>::OpDag;
                BroadcastSch<schMode, OpDag> sch(tiling);
                sch.Process(x1, x2, y);
            }
        }
    } else if constexpr (std::is_same<DTYPE_X1, float>::value) {
        if constexpr (std::is_same<DTYPE_X2, float>::value) {
            if constexpr (canUseMul) {
                using OpDag = TruncateDivOp::TruncateDivFloatScalar<float>::OpDag;
                BroadcastSch<schMode, OpDag> sch(tiling);
                sch.Process(x1, y);
            } else {
                using OpDag = TruncateDivOp::TruncateDivFloat<float>::OpDag;
                BroadcastSch<schMode, OpDag> sch(tiling);
                sch.Process(x1, x2, y);
            }
        } else if constexpr (std::is_same<DTYPE_X2, int32_t>::value) {
            using OpDag = TruncateDivOp::TruncateDivFloatToLowBit<float, int32_t, float>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X2, half>::value) {
            if constexpr (canUseMul) {
                using OpDag = TruncateDivOp::TruncateDivFloatScalar<float>::OpDag;
                BroadcastSch<schMode, OpDag> sch(tiling);
                sch.Process(x1, y);
            } else {
                using OpDag = TruncateDivOp::TruncateDivFloatToLowBit<float, half, float>::OpDag;
                BroadcastSch<schMode, OpDag> sch(tiling);
                sch.Process(x1, x2, y);
            }
        }
    } else if constexpr (std::is_same<DTYPE_X1, int8_t>::value || std::is_same<DTYPE_X1, uint8_t>::value) {
        if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
            using OpDag = TruncateDivOp::TruncateDivIntS8<int8_t, half>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else {
            using OpDag = TruncateDivOp::TruncateDivIntU8<uint8_t, uint16_t>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        }
    } else if constexpr (
        std::is_same<DTYPE_X2, DTYPE_X1>::value &&
        (std::is_same<DTYPE_X1, int16_t>::value || std::is_same<DTYPE_X1, int32_t>::value)) {
        using OpDag = TruncateDivOp::TruncateDivInt<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int64_t>::value) {
        using OpDag = TruncateDivOp::TruncateDivInt64<DTYPE_X1>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, int32_t>::value && std::is_same<DTYPE_X2, float>::value) {
        using OpDag = TruncateDivOp::TruncateDivIntToFloat<int32_t, float, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}