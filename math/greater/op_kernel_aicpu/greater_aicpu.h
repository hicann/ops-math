/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_NORMALIZED_GREATER_H_
#define AICPU_KERNELS_NORMALIZED_GREATER_H_

#include <cstdint>

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
// Upper bound on the number of loop levels the stride iterator carries. It matches the
// rank ceiling the previous implementation enforced, so no shape that used to compute
// starts failing; shapes above it are collapsed by Bcast before building the same strip plan.
constexpr int32_t kMaxBcastDims = 8;

/**
 * @brief Pre-computed iteration plan for a broadcast comparison.
 *
 * Broadcast dimensions carry a stride of 0, so the operand is re-read instead of being
 * materialised -- the same device numpy and PyTorch's TensorIterator use. Output
 * dimensions of size 1 are dropped and adjacent dimensions whose strides are already
 * contiguous are merged, which shortens the loop nest and lengthens the innermost run.
 * Everything lives in fixed-size arrays, so building this plan allocates nothing.
 */
struct BcastPlan {
    int32_t ndims;
    int64_t out_shape[kMaxBcastDims];
    int64_t x_strides[kMaxBcastDims];
    int64_t y_strides[kMaxBcastDims];
    int64_t total_elements;
};

class GreaterCpuKernel : public CpuKernel {
public:
    GreaterCpuKernel() = default;
    ~GreaterCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    uint32_t GreaterParamCheck(CpuKernelContext& ctx) const;

    template <typename T>
    uint32_t GreaterCompute(const CpuKernelContext& ctx, int64_t out_num) const;

    /**
     * @brief Same-shape and scalar-operand cases, written straight into the output.
     */
    template <typename T>
    uint32_t NoBcastCompute(const CpuKernelContext& ctx, int64_t x_num, int64_t y_num, int64_t out_num) const;

    /**
     * @brief General broadcast, driven by a BcastPlan.
     */
    template <typename T>
    uint32_t BcastCompute(const CpuKernelContext& ctx, const BcastPlan& plan, int64_t out_num) const;
};
} // namespace aicpu
#endif
