/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUMULATIVE_LOGSUMEXP_SIMT_H
#define CUMULATIVE_LOGSUMEXP_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "cumulative_logsumexp_tiling_data.h"

#include <type_traits>

namespace NsCumulativeLogsumexp {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <typename T>
__simt_callee__ __aicore__ inline float LoadAsFloat(__gm__ T* input, int64_t offset)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(input[offset]);
    } else {
        return static_cast<float>(input[offset]);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline void StoreFromFloat(__gm__ T* output, int64_t offset, float value)
{
    if constexpr (std::is_same_v<T, half>) {
        output[offset] = __float2half_rn(value);
    } else {
        output[offset] = value;
    }
}

__simt_callee__ __aicore__ inline float LogAddExp(float lhs, float rhs)
{
    if ((lhs != lhs) || (rhs != rhs)) {
        return lhs + rhs;
    }
    float maxVal = fmaxf(lhs, rhs);
    float minVal = fminf(lhs, rhs);
    if (isinf(maxVal)) {
        return maxVal;
    }
    return maxVal + log1pf(expf(minVal - maxVal));
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void CumulativeLogsumexpKernel(
    __gm__ T* x, __gm__ T* y, int64_t totalNum, int64_t outerNum, int64_t axisNum, int64_t innerNum, int64_t exclusive,
    int64_t reverse)
{
    (void)totalNum;
    uint64_t threadNum = static_cast<uint64_t>(Simt::GetThreadNum());
    uint64_t blockNum = static_cast<uint64_t>(Simt::GetBlockNum());
    uint64_t stride = threadNum * blockNum;
    float negInf = -__builtin_inff();
    int64_t axisVectorNum = outerNum * innerNum;

    // O(N) single-pass scan: each thread owns whole axis vectors and keeps a
    // running accumulator along the axis, so every element is touched once.
    for (int64_t vecId = static_cast<int64_t>(Simt::GetBlockIdx()) * static_cast<int64_t>(threadNum) +
                         static_cast<int64_t>(Simt::GetThreadIdx());
         vecId < axisVectorNum; vecId += static_cast<int64_t>(stride)) {
        int64_t outerIdx = vecId / innerNum;
        int64_t innerIdx = vecId % innerNum;
        int64_t base = outerIdx * axisNum * innerNum + innerIdx;
        float acc = negInf;
        if (reverse != 0) {
            for (int64_t i = axisNum - 1; i >= 0; --i) {
                float prev = acc;
                acc = LogAddExp(acc, LoadAsFloat<T>(x, base + i * innerNum));
                StoreFromFloat<T>(y, base + i * innerNum, exclusive != 0 ? prev : acc);
            }
        } else {
            for (int64_t i = 0; i < axisNum; ++i) {
                float prev = acc;
                acc = LogAddExp(acc, LoadAsFloat<T>(x, base + i * innerNum));
                StoreFromFloat<T>(y, base + i * innerNum, exclusive != 0 ? prev : acc);
            }
        }
    }
}

class CumulativeLogsumexpSimt {
public:
    template <typename T>
    __aicore__ static inline void Process(GM_ADDR x, GM_ADDR y, const CumulativeLogsumexpTilingData* tiling)
    {
        __gm__ T* xGm = (__gm__ T*)x;
        __gm__ T* yGm = (__gm__ T*)y;
        Simt::VF_CALL<CumulativeLogsumexpKernel<T>>(Simt::Dim3(THREAD_NUM), xGm, yGm, tiling->totalNum,
                                                    tiling->outerNum, tiling->axisNum, tiling->innerNum,
                                                    tiling->exclusive, tiling->reverse);
    }
};

} // namespace NsCumulativeLogsumexp

#endif // CUMULATIVE_LOGSUMEXP_SIMT_H
