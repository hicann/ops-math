/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_MATH_MUL_OP_KERNEL_AICPU_MUL_AICPU_H_
#define OPS_MATH_MATH_MUL_OP_KERNEL_AICPU_MUL_AICPU_H_

#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include <cstdint>

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
// Upper bound on the loop levels the stride iterator carries. It matches the rank ceiling
// the previous implementation enforced, so no shape that used to compute starts failing.
constexpr int32_t kMulMaxBcastDims = 8;

/**
 * @brief Pre-computed iteration plan for a broadcast multiply.
 *
 * Broadcast dimensions carry a stride of 0, so the operand is re-read instead of being
 * materialised. Output dimensions of size 1 are dropped and adjacent dimensions that are
 * already contiguous for both operands are merged. Everything lives in fixed-size arrays,
 * so building the plan allocates nothing.
 */
struct MulBcastPlan {
    int32_t ndims;
    int64_t out_shape[kMulMaxBcastDims];
    int64_t x_strides[kMulMaxBcastDims];
    int64_t y_strides[kMulMaxBcastDims];
    int64_t total_elements;
};

class MulCpuKernel : public CpuKernel {
public:
    MulCpuKernel() = default;
    ~MulCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    uint32_t MulCompute(const CpuKernelContext& ctx) const;

    /**
     * @brief Same-shape and scalar-operand cases, written straight into the output.
     */
    template <typename T>
    uint32_t MulNoBcast(const CpuKernelContext& ctx, int64_t x_num, int64_t y_num, int64_t out_num) const;

    /**
     * @brief General broadcast driven by MulBcastPlan, rank-agnostic.
     */
    template <typename T>
    uint32_t MulBcastByStride(const CpuKernelContext& ctx, const MulBcastPlan& plan) const;

    uint32_t MulSameTypeCompute(const CpuKernelContext& ctx) const;
};

} // namespace aicpu

#endif // OPS_MATH_MATH_MUL_OP_KERNEL_AICPU_MUL_AICPU_H_
