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
 * \file split_v_aicpu.h
 * \brief SplitV AICPU kernel implementation
 */
#ifndef AICPU_KERNELS_NORMALIZED_SPLIT_V_H_
#define AICPU_KERNELS_NORMALIZED_SPLIT_V_H_

#include <cstddef>
#include <cstdint>

#include "securec.h"

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class SplitVCpuKernel : public CpuKernel {
public:
    SplitVCpuKernel()
        : data_type_(DT_DOUBLE),
          size_splits_type_(DT_INT64),
          split_dim_(0),
          num_split_(0),
          value_num_(0),
          unique_one_index_(-1),
          unique_one_size_(0),
          value_data_ptr_(nullptr),
          size_splits_data_ptr_(nullptr),
          prefix_(1),
          midfix_(1),
          subfix_(1)
    {}

    ~SplitVCpuKernel() = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    /**
     * @brief Init params
     * @param ctx cpu kernel context
     * @return status if success
     */
    uint32_t CheckAndInitParams(const CpuKernelContext& ctx);

    /**
     * @brief Validate and get num_split attribute
     * @param ctx cpu kernel context
     * @return status if success
     */
    uint32_t ValidateAndGetNumSplit(const CpuKernelContext& ctx);

    /**
     * @brief Validate and get split_dim input
     * @param ctx cpu kernel context
     * @return status if success
     */
    uint32_t ValidateAndGetSplitDim(const CpuKernelContext& ctx);

    /**
     * @brief Validate and get value input
     * @param ctx cpu kernel context
     * @param real_dim output the size of split dimension
     * @return status if success
     */
    uint32_t ValidateAndGetValue(const CpuKernelContext& ctx, int64_t& real_dim);

    /**
     * @brief Validate size_splits input and remember only the resolved "-1" entry
     * @param ctx cpu kernel context
     * @param real_dim total size of dim which be split
     * @return status if success
     */
    uint32_t ValidateAndGetSizeSplits(const CpuKernelContext& ctx, int64_t real_dim);

    /**
     * @brief Single pass over size_splits: checks every entry and records the index and
     *        resolved size of the one entry allowed to be -1. The array itself is not
     *        copied; GetSizeSplit reads it back from the input tensor on demand.
     * @param real_dim total size of dim which be split
     * @return status if success
     */
    template <typename T>
    uint32_t ValidateSizeSplits(int64_t real_dim);

    /**
     * @brief Size of split[index], read from the size_splits input tensor on demand
     * @param index split index, must be in [0, num_split_)
     * @return size of that split
     */
    int64_t GetSizeSplit(int64_t index) const;

    /**
     * @brief split data when split num is 1
     * @param ctx cpu kernel context
     * @param input_data_ptr ptr which store input data
     * @return status if success
     */
    template <typename T>
    uint32_t SplitVWithOneOutput(const CpuKernelContext& ctx, const T* input_data_ptr) const;

    /**
     * @brief split data when split dim is 0
     * @param ctx cpu kernel context
     * @param input_data_ptr ptr which store input data
     * @return status if success
     */
    template <typename T>
    uint32_t SplitVWithDimZero(const CpuKernelContext& ctx, const T* input_data_ptr) const;

    /**
     * @brief split data
     * @param ctx cpu kernel context
     * @param input_data_ptr ptr which store input data
     * @return status if success
     */
    uint32_t SplitVComputeSmall(const CpuKernelContext& ctx, const void* input_data_ptr, size_t element_size) const;

    template <typename T>
    uint32_t SplitVCompute(const CpuKernelContext& ctx, const T* input_data_ptr) const;

    template <typename T>
    uint32_t DoCompute(const CpuKernelContext& ctx) const;

    DataType data_type_;
    DataType size_splits_type_;
    int32_t split_dim_;
    int64_t num_split_;
    int64_t value_num_;
    // Index of the single entry of size_splits allowed to be -1, or -1 when there is none.
    int64_t unique_one_index_;
    // Size that the -1 entry resolves to. Meaningless when unique_one_index_ is -1.
    int64_t unique_one_size_;
    const void* value_data_ptr_;
    const void* size_splits_data_ptr_;
    // The three products the copy paths need, derived from the input shape in
    // ValidateAndGetValue. Keeping them instead of the shape itself is what removes the
    // last O(rank) heap allocation: TensorShape::GetDimSizes() returns std::vector by
    // value, so the caller cannot avoid the copy, while GetDimSize(i) allocates nothing.
    int64_t prefix_; // product of the dims before split_dim
    int64_t midfix_; // the dim at split_dim
    int64_t subfix_; // product of the dims after split_dim
};
} // namespace aicpu
#endif
