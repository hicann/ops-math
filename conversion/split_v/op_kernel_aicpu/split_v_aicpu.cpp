/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file split_v_aicpu.cpp
 * \brief SplitV AICPU kernel implementation
 */
#include "split_v_aicpu.h"

#include <algorithm>

#include "aicpu/math_aicpu_register.h"
#include "utils/kernel_util.h"

namespace {
const char* const kSplitV = "SplitV";
constexpr int64_t kSmallSplitThreshold = 4;
constexpr int64_t kGatherChunk = 64;
constexpr uint32_t kSizeSplitsInputIndex = 1;
constexpr uint32_t kSplitDimInputIndex = 2;
} // namespace

namespace aicpu {
uint32_t SplitVCpuKernel::ValidateAndGetNumSplit(const CpuKernelContext& ctx)
{
    AttrValue* num_split_ptr = ctx.GetAttr("num_split");
    KERNEL_CHECK_NULLPTR(num_split_ptr, KERNEL_STATUS_PARAM_INVALID, "Failed to get attribute[num_split].");
    num_split_ = num_split_ptr->GetInt();
    KERNEL_CHECK_FALSE((num_split_ >= 1), KERNEL_STATUS_PARAM_INVALID,
                       "Attribute[num_split] must be greater than or equal to 1, but got [%ld].", num_split_);
    // Widen the output count instead of narrowing num_split, which could wrap values above UINT32_MAX.
    KERNEL_CHECK_FALSE((num_split_ == static_cast<int64_t>(ctx.GetOutputsSize())), KERNEL_STATUS_PARAM_INVALID,
                       "Attribute[num_split] must equal the number of outputs, but got num_split[%ld] and output "
                       "count[%u].",
                       num_split_, ctx.GetOutputsSize());
    return KERNEL_STATUS_OK;
}

uint32_t SplitVCpuKernel::ValidateAndGetSplitDim(const CpuKernelContext& ctx)
{
    Tensor* split_dim_ptr = ctx.Input(kSplitDimInputIndex);
    KERNEL_CHECK_NULLPTR(split_dim_ptr, KERNEL_STATUS_PARAM_INVALID, "Failed to get input[split_dim].");
    auto split_dim_shape_ptr = split_dim_ptr->GetTensorShape();
    KERNEL_CHECK_NULLPTR(split_dim_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Failed to get the shape of input[split_dim].");
    KERNEL_CHECK_FALSE((split_dim_shape_ptr->GetDims() == 0), KERNEL_STATUS_PARAM_INVALID,
                       "Input[split_dim] must be a scalar, but got rank[%d].", split_dim_shape_ptr->GetDims());
    KERNEL_CHECK_FALSE((split_dim_ptr->GetDataType() == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "Input[split_dim] data type must be DT_INT32, but got [%s].",
                       DTypeStr(split_dim_ptr->GetDataType()).c_str());
    auto split_dim_data_ptr = split_dim_ptr->GetData();
    KERNEL_CHECK_NULLPTR(split_dim_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Failed to get the data of input[split_dim].");
    split_dim_ = *(PtrToPtr<void, int32_t>(split_dim_data_ptr));
    KERNEL_CHECK_FALSE((split_dim_ >= 0), KERNEL_STATUS_PARAM_INVALID,
                       "Input[split_dim] must be greater than or equal to 0, but got [%d].", split_dim_);
    return KERNEL_STATUS_OK;
}

uint32_t SplitVCpuKernel::ValidateAndGetValue(const CpuKernelContext& ctx, int64_t& real_dim)
{
    Tensor* value_ptr = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID, "Failed to get input[x].");
    value_data_ptr_ = value_ptr->GetData();
    KERNEL_CHECK_NULLPTR(value_data_ptr_, KERNEL_STATUS_PARAM_INVALID, "Failed to get the data of input[x].");
    auto value_shape_ptr = value_ptr->GetTensorShape();
    KERNEL_CHECK_NULLPTR(value_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Failed to get the shape of input[x].");
    int64_t value_dim = value_shape_ptr->GetDims();
    KERNEL_CHECK_FALSE(value_dim > split_dim_, KERNEL_STATUS_PARAM_INVALID,
                       "The rank of input[x] must be greater than "
                       "input[split_dim], but got rank[%ld] and split_dim[%d].",
                       value_dim, split_dim_);
    real_dim = value_shape_ptr->GetDimSize(split_dim_);
    // Derive the three products the copy paths need instead of copying the shape out.
    // GetDimSizes() returns std::vector by value, so it always costs one heap allocation
    // whose size grows with rank; GetDimSize(i) allocates nothing.
    prefix_ = 1;
    for (int32_t i = 0; i < split_dim_; i++) {
        prefix_ *= value_shape_ptr->GetDimSize(i);
    }
    midfix_ = real_dim;
    subfix_ = 1;
    for (int32_t i = split_dim_ + 1; i < static_cast<int32_t>(value_dim); i++) {
        subfix_ *= value_shape_ptr->GetDimSize(i);
    }
    data_type_ = value_ptr->GetDataType();
    value_num_ = value_ptr->NumElements();
    return KERNEL_STATUS_OK;
}

uint32_t SplitVCpuKernel::ValidateAndGetSizeSplits(const CpuKernelContext& ctx, int64_t real_dim)
{
    Tensor* size_splits_ptr = ctx.Input(kSizeSplitsInputIndex);
    KERNEL_CHECK_NULLPTR(size_splits_ptr, KERNEL_STATUS_PARAM_INVALID, "Failed to get input[size_splits].");
    auto size_splits_shape_ptr = size_splits_ptr->GetTensorShape();
    KERNEL_CHECK_NULLPTR(size_splits_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Failed to get the shape of input[size_splits].");
    int64_t size_splits_dim = size_splits_shape_ptr->GetDims();
    KERNEL_CHECK_FALSE((size_splits_dim == 1), KERNEL_STATUS_PARAM_INVALID,
                       "Input[size_splits] must be a 1-D tensor, but got rank[%ld].", size_splits_dim);
    int64_t size_split_num = size_splits_shape_ptr->GetDimSize(0);
    KERNEL_CHECK_FALSE((size_split_num == num_split_), KERNEL_STATUS_PARAM_INVALID,
                       "The number of elements in input[size_splits] must equal "
                       "attribute[num_split], but got element count[%ld] and "
                       "num_split[%ld].",
                       size_split_num, num_split_);
    size_splits_type_ = size_splits_ptr->GetDataType();
    KERNEL_CHECK_FALSE(((size_splits_type_ == DT_INT32) || (size_splits_type_ == DT_INT64)),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Input[size_splits] data type must be DT_INT32 or DT_INT64, but got [%s].",
                       DTypeStr(size_splits_type_).c_str());
    size_splits_data_ptr_ = size_splits_ptr->GetData();
    KERNEL_CHECK_NULLPTR(size_splits_data_ptr_, KERNEL_STATUS_PARAM_INVALID,
                         "Failed to get the data of input[size_splits].");
    if (size_splits_type_ == DT_INT32) {
        return ValidateSizeSplits<int32_t>(real_dim);
    }
    return ValidateSizeSplits<int64_t>(real_dim);
}

template <typename T>
uint32_t SplitVCpuKernel::ValidateSizeSplits(int64_t real_dim)
{
    const T* size_splits_data = PtrToPtr<const void, const T>(size_splits_data_ptr_);
    unique_one_index_ = -1;
    unique_one_size_ = 0;
    // Accumulate in int64_t to avoid overflow when size_splits uses int32_t.
    int64_t total_dim = 0;
    for (int64_t i = 0; i < num_split_; i++) {
        const int64_t cur_dim = static_cast<int64_t>(size_splits_data[i]);
        if (cur_dim == -1) {
            KERNEL_CHECK_FALSE(unique_one_index_ == -1, KERNEL_STATUS_PARAM_INVALID,
                               "Only one element in size_splits may be -1, but a second one was found at index[%ld].",
                               i);
            unique_one_index_ = i;
        } else {
            KERNEL_CHECK_FALSE(cur_dim >= 0, KERNEL_STATUS_PARAM_INVALID,
                               "Each element in size_splits must be -1 or non-negative, but got [%ld] at index[%ld].",
                               cur_dim, i);
            int64_t next_total_dim = 0;
            const bool is_overflow = __builtin_add_overflow(total_dim, cur_dim, &next_total_dim);
            KERNEL_CHECK_FALSE(!is_overflow, KERNEL_STATUS_PARAM_INVALID,
                               "The sum of size_splits exceeds the int64 range at index[%ld].", i);
            total_dim = next_total_dim;
        }
    }
    KERNEL_CHECK_FALSE(
        ((unique_one_index_ == -1) && (total_dim == real_dim)) || ((unique_one_index_ >= 0) && (total_dim <= real_dim)),
        KERNEL_STATUS_PARAM_INVALID,
        "The sum of size_splits must equal input dimension size[%ld] when fully specified, or not exceed it when "
        "one entry is -1, but got sum[%ld].",
        real_dim, total_dim);
    if (unique_one_index_ >= 0) {
        unique_one_size_ = real_dim - total_dim;
    }
    return KERNEL_STATUS_OK;
}

int64_t SplitVCpuKernel::GetSizeSplit(int64_t index) const
{
    if (index == unique_one_index_) {
        return unique_one_size_;
    }
    if (size_splits_type_ == DT_INT32) {
        return static_cast<int64_t>(PtrToPtr<const void, const int32_t>(size_splits_data_ptr_)[index]);
    }
    return PtrToPtr<const void, const int64_t>(size_splits_data_ptr_)[index];
}

uint32_t SplitVCpuKernel::CheckAndInitParams(const CpuKernelContext& ctx)
{
    uint32_t status = ValidateAndGetNumSplit(ctx);
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    status = ValidateAndGetSplitDim(ctx);
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    int64_t real_dim = 0;
    status = ValidateAndGetValue(ctx, real_dim);
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    return ValidateAndGetSizeSplits(ctx, real_dim);
}

namespace {
template <typename T>
uint32_t ResolveOutput(Tensor* out, int64_t out_index, bool require_data, T*& dst)
{
    KERNEL_CHECK_NULLPTR(out, KERNEL_STATUS_PARAM_INVALID, "Failed to get output[%ld].", out_index);
    dst = PtrToPtr<void, T>(out->GetData());
    // A split of size 0 writes nothing, so its data pointer is allowed to be null.
    // Every split that is actually copied is validated here, before any byte of the
    // current batch is written, rather than surfacing later as a memcpy failure.
    if (require_data) {
        KERNEL_CHECK_NULLPTR(dst, KERNEL_STATUS_PARAM_INVALID, "Failed to get output data[%ld].", out_index);
    }
    return KERNEL_STATUS_OK;
}

// Keep this helper inlined because it runs once per copied segment.
template <typename T>
__attribute__((always_inline)) inline uint32_t CopyRun(T* dst, const T* src, size_t copy_bytes, int64_t out_index)
{
    auto mem_ret = BiggerMemCpy(dst, copy_bytes, src, copy_bytes);
    KERNEL_CHECK_FALSE(mem_ret, KERNEL_STATUS_PARAM_INVALID, "Failed to copy [%zu] bytes from input[x] to output[%ld].",
                       copy_bytes, out_index);
    return KERNEL_STATUS_OK;
}
} // namespace

template <typename T>
uint32_t SplitVCpuKernel::SplitVWithOneOutput(const CpuKernelContext& ctx, const T* input_data_ptr) const
{
    T* dst = nullptr;
    const uint32_t ret = ResolveOutput<T>(ctx.Output(0), 0, GetSizeSplit(0) != 0L, dst);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return CopyRun<T>(dst, input_data_ptr, static_cast<size_t>(value_num_) * sizeof(T), 0);
}

template <typename T>
uint32_t SplitVCpuKernel::SplitVWithDimZero(const CpuKernelContext& ctx, const T* input_data_ptr) const
{
    // split_dim_ is 0 here, so the per-slice element count is exactly the product of the
    // trailing dims, i.e. subfix_. Using it avoids a division whose divisor would need a
    // zero guard of its own.
    const int64_t copy_num = subfix_;
    const T* src = input_data_ptr;
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_split_); i++) {
        const int64_t split_size = GetSizeSplit(i);
        // Resolve before the zero-length skip: the output tensor itself must exist for
        // every slot, only the data pointer of a zero-length split may be null.
        T* dst = nullptr;
        uint32_t ret = ResolveOutput<T>(ctx.Output(static_cast<uint32_t>(i)), i, split_size != 0L, dst);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
        if (split_size == 0L) {
            continue;
        }
        const int64_t elems = split_size * copy_num;
        ret = CopyRun<T>(dst, src, static_cast<size_t>(elems) * sizeof(T), i);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
        src += elems;
    }
    return KERNEL_STATUS_OK;
}

uint32_t SplitVCpuKernel::SplitVComputeSmall(const CpuKernelContext& ctx, const void* input_data_ptr,
                                             size_t element_size) const
{
    const uint64_t prefix = static_cast<uint64_t>(prefix_);
    const int64_t subfix = subfix_;
    const size_t src_stride_bytes = static_cast<size_t>(subfix_ * midfix_) * element_size;

    uint8_t* dst_chunk[kSmallSplitThreshold];
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_split_); i++) {
        const uint32_t ret = ResolveOutput<uint8_t>(ctx.Output(static_cast<uint32_t>(i)), i, GetSizeSplit(i) != 0L,
                                                    dst_chunk[i]);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
    }

    const uint8_t* input_bytes = PtrToPtr<const void, const uint8_t>(input_data_ptr);
    size_t offset_bytes = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(num_split_); i++) {
        const int64_t split_size = GetSizeSplit(i);
        if (split_size == 0L) {
            continue;
        }
        const int64_t copy_num = subfix * split_size;
        const size_t copy_bytes = static_cast<size_t>(copy_num) * element_size;
        for (uint64_t j = 0; j < prefix; j++) {
            const uint32_t ret = CopyRun<uint8_t>(dst_chunk[i] + j * copy_bytes,
                                                  input_bytes + offset_bytes + j * src_stride_bytes, copy_bytes, i);
            if (ret != KERNEL_STATUS_OK) {
                return ret;
            }
        }
        offset_bytes += copy_bytes;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::SplitVCompute(const CpuKernelContext& ctx, const T* input_data_ptr) const
{
    const uint64_t prefix = static_cast<uint64_t>(prefix_);
    const uint64_t subfix = static_cast<uint64_t>(subfix_);
    const uint64_t src_stride = subfix * static_cast<uint64_t>(midfix_);

    // Gather a bounded group, then read its input slices in source order.
    // This avoids revisiting distant input pages for every output tensor.
    T* dst_chunk[kGatherChunk];
    uint64_t offset = 0;
    for (int64_t base = 0; base < num_split_; base += kGatherChunk) {
        const int64_t chunk = std::min<int64_t>(kGatherChunk, num_split_ - base);
        uint64_t chunk_end = offset;
        for (uint64_t c = 0; c < static_cast<uint64_t>(chunk); c++) {
            const int64_t split_size = GetSizeSplit(base + c);
            const uint32_t ret = ResolveOutput<T>(ctx.Output(static_cast<uint32_t>(base + c)), base + c,
                                                  split_size != 0L, dst_chunk[c]);
            if (ret != KERNEL_STATUS_OK) {
                return ret;
            }
            chunk_end += subfix * static_cast<uint64_t>(split_size);
        }
        for (uint64_t j = 0; j < prefix; j++) {
            uint64_t src_offset = offset + j * src_stride;
            for (uint64_t c = 0; c < static_cast<uint64_t>(chunk); c++) {
                const int64_t i = base + c;
                const uint64_t copy_num = subfix * static_cast<uint64_t>(GetSizeSplit(i));
                if (copy_num == 0) {
                    continue;
                }
                const uint32_t ret = CopyRun<T>(dst_chunk[c] + j * copy_num, input_data_ptr + src_offset,
                                                copy_num * sizeof(T), i);
                if (ret != KERNEL_STATUS_OK) {
                    return ret;
                }
                src_offset += copy_num;
            }
        }
        offset = chunk_end;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::DoCompute(const CpuKernelContext& ctx) const
{
    const T* input_data_ptr = PtrToPtr<const void, const T>(value_data_ptr_);
    if (num_split_ == 1) {
        return SplitVWithOneOutput<T>(ctx, input_data_ptr);
    }
    // Gather larger output sets before copying to avoid interleaving output lookups with copies.
    if ((split_dim_ == 0) && (num_split_ <= kSmallSplitThreshold)) {
        return SplitVWithDimZero<T>(ctx, input_data_ptr);
    }
    if (num_split_ <= kSmallSplitThreshold) {
        return SplitVComputeSmall(ctx, input_data_ptr, sizeof(T));
    }
    return SplitVCompute<T>(ctx, input_data_ptr);
}

uint32_t SplitVCpuKernel::Compute(CpuKernelContext& ctx)
{
    Tensor* value_ptr = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID, "Failed to get input[x].");
    auto value_size = value_ptr->GetDataSize();
    if (value_size == 0UL) {
        KERNEL_LOG_INFO("Input[x] is empty; computation is skipped.");
        return KERNEL_STATUS_OK;
    }
    const uint32_t status = CheckAndInitParams(ctx);
    if (status != KERNEL_STATUS_OK) {
        return status;
    }
    KERNEL_LOG_INFO("%s Compute begin, dtype[%s], num_split[%ld], split_dim[%d], elements[%ld].", kSplitV,
                    DTypeStr(data_type_).c_str(), num_split_, split_dim_, value_num_);
    switch (data_type_) {
        case DT_FLOAT16:
            return DoCompute<uint16_t>(ctx);
        case DT_FLOAT:
            return DoCompute<float>(ctx);
        case DT_DOUBLE:
            return DoCompute<double>(ctx);
        case DT_BOOL:
            return DoCompute<bool>(ctx);
        case DT_INT8:
            return DoCompute<int8_t>(ctx);
        case DT_INT16:
            return DoCompute<int16_t>(ctx);
        case DT_INT32:
            return DoCompute<int32_t>(ctx);
        case DT_INT64:
            return DoCompute<int64_t>(ctx);
        case DT_UINT8:
            return DoCompute<uint8_t>(ctx);
        case DT_UINT16:
            return DoCompute<uint16_t>(ctx);
        case DT_UINT32:
            return DoCompute<uint32_t>(ctx);
        case DT_UINT64:
            return DoCompute<uint64_t>(ctx);
        default:
            KERNEL_LOG_ERROR("%s kernel does not support input data type[%s].", kSplitV, DTypeStr(data_type_).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

OPS_MATH_REGISTER_CPU_KERNELV2(kSplitV, SplitVCpuKernel);
} // namespace aicpu
