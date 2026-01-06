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
 * \file util_type_simd.h
 * \brief util simd impl
 */
#ifndef SORT_WITH_INDEX_UTIL_TYPE_SIMD_H
#define SORT_WITH_INDEX_UTIL_TYPE_SIMD_H
#include "kernel_operator.h"
#include "constant_var_simd.h"
template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline uint32_t ROUND_UP_AGLIN(uint32_t x) {
    return (x + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE * UB_AGLIN_VALUE;
}

template <typename T>
__aicore__ inline T SortGetMin(T left, T right)
{
    return (left > right ? right : left);
}
template <typename T>
struct DoubleBufferSimd
{
    AscendC::GlobalTensor<T> doubleBuffer_[2];
    int selector_ = 0;
    __aicore__ inline DoubleBufferSimd() {}
    __aicore__ inline void SetDoubleBuffer(AscendC::GlobalTensor<T> currentBuffer, AscendC::GlobalTensor<T> alternateBuffer) {
        selector_ = 0;
        doubleBuffer_[0] = currentBuffer;
        doubleBuffer_[1] = alternateBuffer;
    }
    __aicore__ inline AscendC::GlobalTensor<T> Current() const {
        return doubleBuffer_[selector_];
    }
    __aicore__ inline AscendC::GlobalTensor<T> Alternate() const {
        return doubleBuffer_[selector_ ^ 1];
    }
    __aicore__ inline void UpdateSelect() {
        selector_ = selector_ ^ 1;
    }
};
#endif