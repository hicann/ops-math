/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expand_aicpu.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "securec.h"
#include "aicpu/math_aicpu_register.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
const char* const kExpand = "Expand";
constexpr uint64_t kHalfBulkCopyThreshold = 16U;

#define EXPAND_EMPTY_TENSOR_CASE(DTYPE, TYPE, CTX) \
    case (DTYPE): {                                \
        EmptyTensorCompute<TYPE>(CTX);             \
        break;                                     \
    }
} // namespace

namespace expand {
template <typename IndexT>
uint32_t NormalizeExpandShape(std::vector<IndexT>& input_shape, std::vector<IndexT>& target_shape)
{
    if (target_shape.size() < input_shape.size()) {
        KERNEL_LOG_ERROR("Param error, target rank [%zu] cannot be less than input rank [%zu].", target_shape.size(),
                         input_shape.size());
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }

    const size_t diff = target_shape.size() - input_shape.size();
    if (diff > 0) {
        input_shape.insert(input_shape.begin(), diff, static_cast<IndexT>(1));
    }

    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] < static_cast<IndexT>(-1)) {
            KERNEL_LOG_ERROR("Param error, target_shape[%zu] [%ld] is invalid.", i,
                             static_cast<int64_t>(target_shape[i]));
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }

        if (aicpu::IsValueEqual<IndexT>(target_shape[i], static_cast<IndexT>(-1))) {
            target_shape[i] = input_shape[i];
            continue;
        }

        if (aicpu::IsValueEqual<IndexT>(target_shape[i], static_cast<IndexT>(1)) &&
            !aicpu::IsValueEqual<IndexT>(input_shape[i], static_cast<IndexT>(1))) {
            target_shape[i] = input_shape[i];
            continue;
        }

        if (!aicpu::IsValueEqual<IndexT>(input_shape[i], static_cast<IndexT>(1)) &&
            !aicpu::IsValueEqual<IndexT>(input_shape[i], target_shape[i])) {
            KERNEL_LOG_ERROR("Param error, input_shape[%zu] [%ld] cannot broadcast to target_shape[%zu] [%ld].", i,
                             static_cast<int64_t>(input_shape[i]), i, static_cast<int64_t>(target_shape[i]));
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
    }

    return aicpu::KERNEL_STATUS_OK;
}

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t& product)
{
    if (right != 0U && left > std::numeric_limits<uint64_t>::max() / right) {
        return false;
    }
    product = left * right;
    return true;
}

template <typename IndexT>
struct ExpandPlan {
    std::vector<IndexT> input_shape;
    std::vector<IndexT> target_shape;
    std::vector<uint64_t> input_strides;
    std::vector<uint64_t> coordinates;
    size_t outer_rank = 0U;
    uint64_t copy_elements = 1U;
    uint64_t output_blocks = 1U;
    uint64_t input_elements = 1U;
};

template <typename IndexT>
uint32_t BuildExpandPlan(ExpandPlan<IndexT>& plan)
{
    for (IndexT dim : plan.target_shape) {
        if (aicpu::IsValueEqual<IndexT>(dim, static_cast<IndexT>(0))) {
            plan.copy_elements = 0U;
            plan.output_blocks = 0U;
            return aicpu::KERNEL_STATUS_OK;
        }
    }

    plan.outer_rank = plan.target_shape.size();
    while (plan.outer_rank > 0U && aicpu::IsValueEqual<IndexT>(plan.input_shape[plan.outer_rank - 1U],
                                                               plan.target_shape[plan.outer_rank - 1U])) {
        if (!CheckedMultiply(plan.copy_elements, static_cast<uint64_t>(plan.target_shape[plan.outer_rank - 1U]),
                             plan.copy_elements)) {
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
        --plan.outer_rank;
    }

    for (size_t i = 0; i < plan.outer_rank; ++i) {
        if (!CheckedMultiply(plan.output_blocks, static_cast<uint64_t>(plan.target_shape[i]), plan.output_blocks)) {
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
    }
    plan.input_strides.resize(plan.outer_rank);
    plan.coordinates.assign(plan.outer_rank, 0U);
    uint64_t stride = plan.copy_elements;
    for (size_t i = plan.outer_rank; i > 0U; --i) {
        plan.input_strides[i - 1U] = stride;
        if (!CheckedMultiply(stride, static_cast<uint64_t>(plan.input_shape[i - 1U]), stride)) {
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
    }
    plan.input_elements = stride;
    return aicpu::KERNEL_STATUS_OK;
}

template <typename IndexT>
void AdvanceInputOffset(ExpandPlan<IndexT>& plan, uint64_t& input_offset)
{
    for (size_t i = plan.outer_rank; i > 0U; --i) {
        const size_t axis = i - 1U;
        ++plan.coordinates[axis];
        if (plan.coordinates[axis] < static_cast<uint64_t>(plan.target_shape[axis])) {
            if (!aicpu::IsValueEqual<IndexT>(plan.input_shape[axis], static_cast<IndexT>(1))) {
                input_offset += plan.input_strides[axis];
            }
            return;
        }
        plan.coordinates[axis] = 0U;
        if (!aicpu::IsValueEqual<IndexT>(plan.input_shape[axis], static_cast<IndexT>(1))) {
            input_offset -= (static_cast<uint64_t>(plan.target_shape[axis]) - 1U) * plan.input_strides[axis];
        }
    }
}

template <typename T, typename IndexT>
uint32_t ValidateCopyBuffers(const aicpu::CpuKernelContext& ctx, const ExpandPlan<IndexT>& plan, uint64_t& copy_bytes)
{
    uint64_t input_bytes = 0U;
    uint64_t output_elements = 0U;
    uint64_t output_bytes = 0U;
    if (!CheckedMultiply(plan.copy_elements, sizeof(T), copy_bytes) ||
        !CheckedMultiply(plan.input_elements, sizeof(T), input_bytes) ||
        !CheckedMultiply(plan.output_blocks, plan.copy_elements, output_elements) ||
        !CheckedMultiply(output_elements, sizeof(T), output_bytes)) {
        KERNEL_LOG_ERROR("Expand tensor size calculation overflowed.");
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }

    const uint64_t input_data_size = ctx.Input(0)->GetDataSize();
    const uint64_t output_data_size = ctx.Output(0)->GetDataSize();
    const uint64_t max_size = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (input_bytes > input_data_size || output_bytes > output_data_size || input_bytes > max_size ||
        output_data_size > max_size) {
        KERNEL_LOG_ERROR("Expand buffer is invalid, input [%lu/%lu], output [%lu/%lu], copy [%lu].", input_bytes,
                         input_data_size, output_bytes, output_data_size, copy_bytes);
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T>
bool ShouldUseBiggerMemCpy(uint64_t copy_bytes)
{
    return copy_bytes > static_cast<uint64_t>(SECUREC_MEM_MAX_LEN);
}

template <>
bool ShouldUseBiggerMemCpy<Eigen::half>(uint64_t copy_bytes)
{
    return copy_bytes >= kHalfBulkCopyThreshold;
}

template <>
bool ShouldUseBiggerMemCpy<Eigen::bfloat16>(uint64_t copy_bytes)
{
    return copy_bytes >= kHalfBulkCopyThreshold;
}

template <bool UseBiggerMemCpy, typename T, typename IndexT>
uint32_t CopyExpandedBlocks(const aicpu::CpuKernelContext& ctx, ExpandPlan<IndexT>& plan, uint64_t copy_bytes)
{
    const uint64_t output_data_size = ctx.Output(0)->GetDataSize();
    const auto* input_data = static_cast<const T*>(ctx.Input(0)->GetData());
    auto* output_data = static_cast<T*>(ctx.Output(0)->GetData());
    uint64_t input_offset = 0U;
    uint64_t output_offset_bytes = 0U;
    for (uint64_t block = 0U; block < plan.output_blocks; ++block) {
        if (UseBiggerMemCpy) {
            if (!aicpu::BiggerMemCpy(output_data + block * plan.copy_elements,
                                     static_cast<size_t>(output_data_size - output_offset_bytes),
                                     input_data + input_offset, static_cast<size_t>(copy_bytes))) {
                KERNEL_LOG_ERROR("Expand copy block [%lu] failed, copy bytes [%lu].", block, copy_bytes);
                return aicpu::KERNEL_STATUS_INNER_ERROR;
            }
        } else {
            std::copy_n(input_data + input_offset, plan.copy_elements, output_data + block * plan.copy_elements);
        }
        output_offset_bytes += copy_bytes;
        if (block + 1U < plan.output_blocks) {
            AdvanceInputOffset(plan, input_offset);
        }
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t CopyExpandedData(const aicpu::CpuKernelContext& ctx, ExpandPlan<IndexT>& plan)
{
    if (plan.output_blocks == 0U) {
        return aicpu::KERNEL_STATUS_OK;
    }
    uint64_t copy_bytes = 0U;
    KERNEL_HANDLE_ERROR(ValidateCopyBuffers<T>(ctx, plan, copy_bytes), "Expand buffer validation failed.");
    if (ShouldUseBiggerMemCpy<T>(copy_bytes)) {
        return CopyExpandedBlocks<true, T, IndexT>(ctx, plan, copy_bytes);
    }
    return CopyExpandedBlocks<false, T, IndexT>(ctx, plan, copy_bytes);
}

template <typename T, typename IndexT>
uint32_t DoExpandCompute(const aicpu::CpuKernelContext& ctx)
{
    const auto* shape_data = static_cast<const IndexT*>(ctx.Input(1)->GetData());
    const auto* input_tensor = ctx.Input(0);
    const auto* shape_tensor = ctx.Input(1);

    ExpandPlan<IndexT> plan;
    std::vector<int64_t> origin_shape = input_tensor->GetTensorShape()->GetDimSizes();
    const int64_t target_rank = shape_tensor->NumElements();
    if (target_rank < 0) {
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
    if (origin_shape.empty()) {
        origin_shape.push_back(static_cast<int64_t>(1));
    }

    plan.input_shape.reserve(static_cast<size_t>(target_rank));
    plan.target_shape.reserve(static_cast<size_t>(target_rank));
    for (int64_t dim : origin_shape) {
        plan.input_shape.push_back(static_cast<IndexT>(dim));
    }
    for (int64_t i = 0; i < target_rank; ++i) {
        plan.target_shape.push_back(shape_data[i]);
    }

    uint32_t ret = NormalizeExpandShape(plan.input_shape, plan.target_shape);
    if (ret != aicpu::KERNEL_STATUS_OK) {
        return ret;
    }
    ret = BuildExpandPlan(plan);
    if (ret != aicpu::KERNEL_STATUS_OK) {
        return ret;
    }
    return CopyExpandedData<T, IndexT>(ctx, plan);
}

template <typename IndexT>
uint32_t IndicesExpandCompute(aicpu::CpuKernelContext& ctx)
{
    auto input_type = static_cast<aicpu::DataType>(ctx.Input(0)->GetDataType());
    switch (input_type) {
        case aicpu::DT_FLOAT16:
            return DoExpandCompute<Eigen::half, IndexT>(ctx);
        case aicpu::DT_BFLOAT16:
            return DoExpandCompute<Eigen::bfloat16, IndexT>(ctx);
        case aicpu::DT_FLOAT:
            return DoExpandCompute<float, IndexT>(ctx);
        case aicpu::DT_INT8:
            return DoExpandCompute<int8_t, IndexT>(ctx);
        case aicpu::DT_INT32:
            return DoExpandCompute<int32_t, IndexT>(ctx);
        case aicpu::DT_INT64:
            return DoExpandCompute<int64_t, IndexT>(ctx);
        case aicpu::DT_UINT8:
            return DoExpandCompute<uint8_t, IndexT>(ctx);
        case aicpu::DT_BOOL:
            return DoExpandCompute<bool, IndexT>(ctx);
        default:
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
}
} // namespace expand

namespace aicpu {
template <typename T>
void ExpandCpuKernel::EmptyTensorCompute(const CpuKernelContext& ctx)
{
    const int64_t shape_num = ctx.Input(kSecondInputIndex)->NumElements();
    KERNEL_LOG_INFO("shape num elements [%ld]", shape_num);
    if (shape_num == 0) {
        auto* output_data = aicpu::PtrToPtr<void, T>(ctx.Output(kFirstOutputIndex)->GetData());
        const auto* input_data = aicpu::PtrToPtr<void, const T>(ctx.Input(kFirstInputIndex)->GetData());
        *output_data = *input_data;
        is_empty_tensor_ = true;
    }
}

void ExpandCpuKernel::HandleEmptyTensor(const CpuKernelContext& ctx)
{
    auto input_type = ctx.Input(kFirstInputIndex)->GetDataType();
    switch (input_type) {
        EXPAND_EMPTY_TENSOR_CASE(DT_FLOAT16, Eigen::half, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_BFLOAT16, Eigen::bfloat16, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_FLOAT, float, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_INT32, int32_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_INT64, int64_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_INT8, int8_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_UINT8, uint8_t, ctx)
        EXPAND_EMPTY_TENSOR_CASE(DT_BOOL, bool, ctx)
        default:
            KERNEL_LOG_WARN("Expand empty tensor data type [%u] not support.", input_type);
    }
}

uint32_t ExpandCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("ExpandCpuKernel start.");
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check Expand params failed.");

    is_empty_tensor_ = false;
    HandleEmptyTensor(ctx);
    if (is_empty_tensor_) {
        KERNEL_LOG_INFO("shape of expand empty tensor scenario.");
        return KERNEL_STATUS_OK;
    }

    auto shape_type = static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
    switch (shape_type) {
        case DT_INT32:
            return expand::IndicesExpandCompute<int32_t>(ctx);
        case DT_INT64:
            return expand::IndicesExpandCompute<int64_t>(ctx);
        default:
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

OPS_MATH_REGISTER_CPU_KERNELV2(kExpand, ExpandCpuKernel);
} // namespace aicpu
