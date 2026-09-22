/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "greater_aicpu.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "Eigen/Core"

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "log.h"
#include "status.h"

namespace {
const char* const kGreater = "Greater";
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
} // namespace

namespace aicpu {
namespace {

bool PadAndValidate(const std::vector<int64_t>& x, const std::vector<int64_t>& y, int32_t rank,
                    int64_t expected_elements, int64_t* xp, int64_t* yp, int64_t* out)
{
    const int32_t x_rank = static_cast<int32_t>(x.size());
    const int32_t y_rank = static_cast<int32_t>(y.size());
    int64_t total = 1;
    for (int32_t i = 0; i < rank; ++i) {
        xp[i] = (i >= rank - x_rank) ? x[static_cast<size_t>(i - (rank - x_rank))] : 1;
        yp[i] = (i >= rank - y_rank) ? y[static_cast<size_t>(i - (rank - y_rank))] : 1;
        if (xp[i] == yp[i]) {
            out[i] = xp[i];
        } else if (xp[i] == 1) {
            out[i] = yp[i];
        } else if (yp[i] == 1) {
            out[i] = xp[i];
        } else {
            return false;
        }
        int64_t next_total = 0;
        if ((out[i] <= 0) || __builtin_mul_overflow(total, out[i], &next_total) || (next_total > expected_elements)) {
            return false;
        }
        total = next_total;
    }
    return total == expected_elements;
}

void EffectiveStrides(const int64_t* padded, const int64_t* out, int32_t rank, int64_t* eff)
{
    int64_t natural[kMaxBcastDims];
    natural[rank - 1] = 1;
    for (int32_t d = rank - 2; d >= 0; --d) {
        natural[d] = natural[d + 1] * padded[d + 1];
    }
    for (int32_t d = 0; d < rank; ++d) {
        eff[d] = (padded[d] == out[d]) ? natural[d] : 0;
    }
}

void DropAndCollapse(const int64_t* out, const int64_t* xe, const int64_t* ye, int32_t rank, int64_t total_elements,
                     BcastPlan& plan)
{
    int64_t to[kMaxBcastDims];
    int64_t tx[kMaxBcastDims];
    int64_t ty[kMaxBcastDims];
    int32_t kept = 0;
    for (int32_t d = 0; d < rank; ++d) {
        if (out[d] != 1) {
            to[kept] = out[d];
            tx[kept] = xe[d];
            ty[kept] = ye[d];
            ++kept;
        }
    }
    if (kept == 0) {
        plan.ndims = 1;
        plan.out_shape[0] = 1;
        plan.x_strides[0] = 0;
        plan.y_strides[0] = 0;
        plan.total_elements = total_elements;
        return;
    }

    plan.out_shape[0] = to[0];
    plan.x_strides[0] = tx[0];
    plan.y_strides[0] = ty[0];
    int32_t n = 1;
    for (int32_t d = 1; d < kept; ++d) {
        // A broadcast dim carries stride 0, and 0 == 0 * to[d] already holds, so no extra case is needed.
        const bool x_ok = (plan.x_strides[n - 1] == tx[d] * to[d]);
        const bool y_ok = (plan.y_strides[n - 1] == ty[d] * to[d]);
        if (x_ok && y_ok) {
            plan.out_shape[n - 1] *= to[d];
            plan.x_strides[n - 1] = tx[d];
            plan.y_strides[n - 1] = ty[d];
        } else {
            plan.out_shape[n] = to[d];
            plan.x_strides[n] = tx[d];
            plan.y_strides[n] = ty[d];
            ++n;
        }
    }
    plan.ndims = n;
    plan.total_elements = total_elements;
}

bool BuildBcastPlan(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, int64_t out_num,
                    BcastPlan& plan)
{
    const int32_t rank = std::max(static_cast<int32_t>(x_shape.size()), static_cast<int32_t>(y_shape.size()));
    if ((rank <= 0) || (rank > kMaxBcastDims)) {
        return false;
    }
    int64_t xp[kMaxBcastDims];
    int64_t yp[kMaxBcastDims];
    int64_t out[kMaxBcastDims];
    if (!PadAndValidate(x_shape, y_shape, rank, out_num, xp, yp, out)) {
        return false;
    }
    int64_t xe[kMaxBcastDims];
    int64_t ye[kMaxBcastDims];
    EffectiveStrides(xp, out, rank, xe);
    EffectiveStrides(yp, out, rank, ye);
    DropAndCollapse(out, xe, ye, rank, out_num, plan);
    return true;
}

// Tensor lengths and effective strides are non-negative. Unsigned indexing keeps
// -ftrapv from adding signed increment and offset checks to every comparison.
// These helpers run once per inner strip, so they must not remain out-of-line calls.
template <typename T>
__attribute__((always_inline)) inline void RunContiguous(const T* x, const T* y, bool* out, int64_t n)
{
    if (n <= 0) {
        return;
    }
    for (uint64_t i = 0; i < static_cast<uint64_t>(n); ++i) {
        out[i] = x[i] > y[i];
    }
}

template <typename T>
__attribute__((always_inline)) inline void RunYScalar(const T* x, T y_val, bool* out, int64_t n, int64_t x_step)
{
    if (n <= 0) {
        return;
    }
    for (uint64_t i = 0; i < static_cast<uint64_t>(n); ++i) {
        out[i] = x[i * static_cast<uint64_t>(x_step)] > y_val;
    }
}

template <typename T>
__attribute__((always_inline)) inline void RunXScalar(T x_val, const T* y, bool* out, int64_t n, int64_t y_step)
{
    if (n <= 0) {
        return;
    }
    for (uint64_t i = 0; i < static_cast<uint64_t>(n); ++i) {
        out[i] = x_val > y[i * static_cast<uint64_t>(y_step)];
    }
}

template <typename T>
__attribute__((always_inline)) inline void RunStrided(const T* x, const T* y, bool* out, int64_t n, int64_t x_step,
                                                      int64_t y_step)
{
    if (n <= 0) {
        return;
    }
    for (uint64_t i = 0; i < static_cast<uint64_t>(n); ++i) {
        out[i] = x[i * static_cast<uint64_t>(x_step)] > y[i * static_cast<uint64_t>(y_step)];
    }
}

template <typename T>
__attribute__((always_inline)) inline void RunStrip(const T* x, const T* y, bool* out, int64_t n, int64_t x_step,
                                                    int64_t y_step)
{
    if ((x_step == 1) && (y_step == 1)) {
        RunContiguous<T>(x, y, out, n);
    } else if (y_step == 0) {
        RunYScalar<T>(x, *y, out, n, x_step);
    } else if (x_step == 0) {
        RunXScalar<T>(*x, y, out, n, y_step);
    } else {
        RunStrided<T>(x, y, out, n, x_step, y_step);
    }
}

uint32_t ValidateBcast(const Bcast& bcast, size_t x_rank, size_t y_rank)
{
    KERNEL_CHECK_FALSE(bcast.IsValid(), KERNEL_STATUS_PARAM_INVALID,
                       "%s cannot broadcast input[x1] of rank[%zu] against input[x2] of rank[%zu].", kGreater, x_rank,
                       y_rank);
    const size_t collapsed_rank = bcast.XReshape().size();
    KERNEL_CHECK_FALSE(collapsed_rank <= static_cast<size_t>(kMaxBcastDims), KERNEL_STATUS_PARAM_INVALID,
                       "%s does not support broadcast rank[%zu]; the maximum supported rank is [%d].", kGreater,
                       collapsed_rank, kMaxBcastDims);
    return KERNEL_STATUS_OK;
}

uint32_t ValidateEmptyBcastResult(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape)
{
    const size_t rank = std::max(x_shape.size(), y_shape.size());
    const size_t x_offset = rank - x_shape.size();
    const size_t y_offset = rank - y_shape.size();
    bool is_empty = false;
    for (size_t dim = 0; dim < rank; ++dim) {
        const int64_t x_dim = (dim < x_offset) ? 1 : x_shape[dim - x_offset];
        const int64_t y_dim = (dim < y_offset) ? 1 : y_shape[dim - y_offset];
        const bool compatible = (x_dim == y_dim) || (x_dim == 1) || (y_dim == 1);
        KERNEL_CHECK_FALSE(compatible, KERNEL_STATUS_PARAM_INVALID,
                           "%s cannot broadcast input[x1] of rank[%zu] against input[x2] of rank[%zu].", kGreater,
                           x_shape.size(), y_shape.size());
        const int64_t result_dim = (x_dim == 1) ? y_dim : x_dim;
        is_empty = is_empty || (result_dim == 0);
    }
    KERNEL_CHECK_FALSE(is_empty, KERNEL_STATUS_PARAM_INVALID,
                       "%s output is empty but the broadcast result is non-empty.", kGreater);
    return KERNEL_STATUS_OK;
}

uint32_t HandleEmptyOutput(std::vector<int64_t>& x_shape, std::vector<int64_t>& y_shape, bool use_direct_compute)
{
    if (!use_direct_compute) {
        const uint32_t status = ValidateEmptyBcastResult(x_shape, y_shape);
        if (status != KERNEL_STATUS_OK) {
            return status;
        }
    }
    const Bcast bcast(x_shape, y_shape);
    const uint32_t status = ValidateBcast(bcast, x_shape.size(), y_shape.size());
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    KERNEL_LOG_INFO("The %s output tensor is empty; computation is skipped.", kGreater);
    return KERNEL_STATUS_OK;
}

uint32_t ValidateBcastSize(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, int64_t out_num)
{
    KERNEL_CHECK_FALSE((out_num > 0), KERNEL_STATUS_PARAM_INVALID,
                       "%s output elements[%ld] must be positive for a non-empty broadcast.", kGreater, out_num);
    const size_t rank = std::max(x_shape.size(), y_shape.size());
    const size_t x_offset = rank - x_shape.size();
    const size_t y_offset = rank - y_shape.size();
    int64_t total = 1;
    for (size_t dim = 0; dim < rank; ++dim) {
        const int64_t x_dim = (dim < x_offset) ? 1 : x_shape[dim - x_offset];
        const int64_t y_dim = (dim < y_offset) ? 1 : y_shape[dim - y_offset];
        const bool compatible = (x_dim == y_dim) || (x_dim == 1) || (y_dim == 1);
        KERNEL_CHECK_FALSE(compatible, KERNEL_STATUS_PARAM_INVALID,
                           "%s cannot broadcast input[x1] of rank[%zu] against input[x2] of rank[%zu].", kGreater,
                           x_shape.size(), y_shape.size());
        const int64_t result_dim = (x_dim == 1) ? y_dim : x_dim;
        int64_t next_total = 0;
        if ((result_dim <= 0) || __builtin_mul_overflow(total, result_dim, &next_total) || (next_total > out_num)) {
            KERNEL_LOG_ERROR("%s output elements[%ld] do not match the broadcast result.", kGreater, out_num);
            return KERNEL_STATUS_PARAM_INVALID;
        }
        total = next_total;
    }
    KERNEL_CHECK_FALSE((total == out_num), KERNEL_STATUS_PARAM_INVALID,
                       "%s output elements[%ld] do not match broadcast result[%ld].", kGreater, out_num, total);
    return KERNEL_STATUS_OK;
}

void BuildValidatedBcastPlan(const Bcast& bcast, int64_t out_num, BcastPlan& plan)
{
    const auto& x = bcast.XReshape();
    const auto& y = bcast.YReshape();
    const int32_t rank = static_cast<int32_t>(x.size());
    int64_t out[kMaxBcastDims];
    int64_t xe[kMaxBcastDims];
    int64_t ye[kMaxBcastDims];
    for (int32_t d = 0; d < rank; ++d) {
        out[d] = std::max(x[d], y[d]);
    }
    EffectiveStrides(x.data(), out, rank, xe);
    EffectiveStrides(y.data(), out, rank, ye);
    DropAndCollapse(out, xe, ye, rank, out_num, plan);
}

} // namespace

template <typename T>
uint32_t GreaterCpuKernel::NoBcastCompute(const CpuKernelContext& ctx, int64_t x_num, int64_t y_num,
                                          int64_t out_num) const
{
    const T* x = PtrToPtr<const void, const T>(ctx.Input(kFirstInputIndex)->GetData());
    const T* y = PtrToPtr<const void, const T>(ctx.Input(kSecondInputIndex)->GetData());
    bool* out = PtrToPtr<void, bool>(ctx.Output(kFirstOutputIndex)->GetData());

    if (x_num == y_num) {
        RunContiguous<T>(x, y, out, out_num);
    } else if (x_num == 1) {
        RunXScalar<T>(*x, y, out, out_num, 1);
    } else {
        RunYScalar<T>(x, *y, out, out_num, 1);
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GreaterCpuKernel::BcastCompute(const CpuKernelContext& ctx, const BcastPlan& plan, int64_t out_num) const
{
    KERNEL_CHECK_FALSE((plan.total_elements == out_num), KERNEL_STATUS_PARAM_INVALID,
                       "%s output elements[%ld] do not match broadcast result[%ld].", kGreater, out_num,
                       plan.total_elements);
    const T* x = PtrToPtr<const void, const T>(ctx.Input(kFirstInputIndex)->GetData());
    const T* y = PtrToPtr<const void, const T>(ctx.Input(kSecondInputIndex)->GetData());
    bool* out = PtrToPtr<void, bool>(ctx.Output(kFirstOutputIndex)->GetData());

    const int32_t nd = plan.ndims;
    const int64_t inner = plan.out_shape[nd - 1];
    const int64_t x_step = plan.x_strides[nd - 1];
    const int64_t y_step = plan.y_strides[nd - 1];

    if (nd == 1) {
        RunStrip<T>(x, y, out, inner, x_step, y_step);
        return KERNEL_STATUS_OK;
    }

    // Carry propagation avoids a division for every output element.
    int64_t coords[kMaxBcastDims] = {0};
    int64_t x_off = 0;
    int64_t y_off = 0;
    for (int64_t written = 0; written < plan.total_elements; written += inner) {
        RunStrip<T>(x + x_off, y + y_off, out + written, inner, x_step, y_step);
        for (int32_t d = nd - 2; d >= 0; --d) {
            coords[d] += 1;
            x_off += plan.x_strides[d];
            y_off += plan.y_strides[d];
            if (coords[d] < plan.out_shape[d]) {
                break;
            }
            x_off -= plan.out_shape[d] * plan.x_strides[d];
            y_off -= plan.out_shape[d] * plan.y_strides[d];
            coords[d] = 0;
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GreaterCpuKernel::GreaterCompute(const CpuKernelContext& ctx, int64_t out_num) const
{
    Tensor* x_tensor = ctx.Input(kFirstInputIndex);
    Tensor* y_tensor = ctx.Input(kSecondInputIndex);
    std::vector<int64_t> x_shape = x_tensor->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> y_shape = y_tensor->GetTensorShape()->GetDimSizes();
    const int64_t x_num = x_tensor->NumElements();
    const int64_t y_num = y_tensor->NumElements();
    KERNEL_CHECK_FALSE((x_num >= 0) && (y_num >= 0) && (out_num >= 0), KERNEL_STATUS_PARAM_INVALID,
                       "%s input/output element counts must be non-negative, but got x1[%ld], x2[%ld], y[%ld].",
                       kGreater, x_num, y_num, out_num);
    const bool is_same_shape = x_shape == y_shape;
    const bool use_direct_compute = is_same_shape || (x_num == 1) || (y_num == 1);

    if (use_direct_compute) {
        const int64_t broadcast_num = is_same_shape ? x_num : ((x_num == 1) ? y_num : x_num);
        KERNEL_CHECK_FALSE((broadcast_num == out_num), KERNEL_STATUS_PARAM_INVALID,
                           "%s output elements[%ld] do not match broadcast result[%ld].", kGreater, out_num,
                           broadcast_num);
    }

    if (out_num == 0) {
        return HandleEmptyOutput(x_shape, y_shape, use_direct_compute);
    }

    if (use_direct_compute) {
        return NoBcastCompute<T>(ctx, x_num, y_num, out_num);
    }

    BcastPlan plan;
    const size_t rank = std::max(x_shape.size(), y_shape.size());
    if (rank <= static_cast<size_t>(kMaxBcastDims)) {
        KERNEL_CHECK_FALSE(BuildBcastPlan(x_shape, y_shape, out_num, plan), KERNEL_STATUS_PARAM_INVALID,
                           "%s output elements[%ld] do not match a valid broadcast result.", kGreater, out_num);
        return BcastCompute<T>(ctx, plan, out_num);
    }

    const uint32_t result_status = ValidateBcastSize(x_shape, y_shape, out_num);
    if (result_status != KERNEL_STATUS_OK) {
        return result_status;
    }
    const Bcast bcast(x_shape, y_shape);
    const uint32_t ret = ValidateBcast(bcast, x_shape.size(), y_shape.size());
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    // Bcast has already collapsed valid high-rank shapes to at most eight dimensions.
    // Reuse the strip iterator instead of rebuilding two indices for every element.
    BuildValidatedBcastPlan(bcast, out_num, plan);
    return BcastCompute<T>(ctx, plan, out_num);
}

uint32_t GreaterCpuKernel::GreaterParamCheck(CpuKernelContext& ctx) const
{
    const uint32_t status = NormalCheck(ctx, kInputNum, kOutputNum);
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    const DataType x_type = ctx.Input(kFirstInputIndex)->GetDataType();
    const DataType y_type = ctx.Input(kSecondInputIndex)->GetDataType();
    KERNEL_CHECK_FALSE((x_type == y_type), KERNEL_STATUS_PARAM_INVALID,
                       "Input[x1] data type[%s] and input[x2] data type[%s] must be the same.",
                       DTypeStr(x_type).c_str(), DTypeStr(y_type).c_str());
    return KERNEL_STATUS_OK;
}

#define GREATER_COMPUTE_CASE(DTYPE, TYPE, CTX, N) \
    case (DTYPE):                                 \
        return GreaterCompute<TYPE>(CTX, N)

uint32_t GreaterCpuKernel::Compute(CpuKernelContext& ctx)
{
    const uint32_t status = GreaterParamCheck(ctx);
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    const int64_t out_elements = ctx.Output(kFirstOutputIndex)->NumElements();
    const DataType data_type = ctx.Input(kFirstInputIndex)->GetDataType();
    KERNEL_LOG_INFO("%s Compute begin, dtype[%s], out_elements[%ld].", kGreater, DTypeStr(data_type).c_str(),
                    out_elements);
    switch (data_type) {
        GREATER_COMPUTE_CASE(DT_FLOAT, float, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_DOUBLE, double, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_INT8, int8_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_INT16, int16_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_INT32, int32_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_INT64, int64_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_UINT8, uint8_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_UINT16, uint16_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_UINT32, uint32_t, ctx, out_elements);
        GREATER_COMPUTE_CASE(DT_UINT64, uint64_t, ctx, out_elements);
        default:
            KERNEL_LOG_ERROR("%s kernel does not support input data type[%s].", kGreater, DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

REGISTER_CPU_KERNEL(kGreater, GreaterCpuKernel);
} // namespace aicpu
