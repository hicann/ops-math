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
 * \file stateless_drop_out_gen_mask_base.h
 * \brief
 */
#ifndef STATELESS_DROP_OUT_GEN_MASK_BASE_H
#define STATELESS_DROP_OUT_GEN_MASK_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace StatelessDropOutGenMask {
using namespace AscendC;

template <typename T>
class StatelessDropOutGenMaskBase {
public:
    __aicore__ inline StatelessDropOutGenMaskBase(){};

public:
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t RESULT_ELEMENT_CNT = 4;
    constexpr static float CURAND_2POW32_INV = 2.3283064365386963e-10;
    constexpr static float COEFF_2POW32_INV = 2.0f;

    constexpr static uint32_t RIGHT_SHIFT_BIT = 32;
    constexpr static uint32_t COUNTER_IDX2 = 2;
    constexpr static uint32_t COUNTER_IDX3 = 3;
    constexpr static uint32_t SET_CTRL_VAL = 60;

    constexpr static uint32_t gainCoeff = 2;
    constexpr static uint32_t byteBitRatio = 8;
    constexpr static uint32_t RoundUpByte256 = 256;
    constexpr static uint32_t OnceAlignNumSize32 = 8;

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };
    template <typename TK>
    __aicore__ inline TK AlignUp256(TK param)
    {
        return (param + RoundUpByte256 - 1) / RoundUpByte256 * RoundUpByte256;
    };
    template <typename TK>
    __aicore__ inline TK PadProcessInt32(TK param)
    {
        return (OnceAlignNumSize32 - param % OnceAlignNumSize32);
    };
};
} // namespace StatelessDropOutGenMask
#endif // STATELESS_DROP_OUT_GEN_MASK_BASE_H