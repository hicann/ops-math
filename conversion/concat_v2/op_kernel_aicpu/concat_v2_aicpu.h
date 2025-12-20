/**
* This program is free software, you can redistribute it and/or modify it.
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
* BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
* the software repository for the full text of the License.
*/

#ifndef AICPU_KERNELS_CONCATV2_H_
#define AICPU_KERNELS_CONCATV2_H_

#include <memory>
#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
template <typename T>
struct TTypes {
  constexpr static int kTensorRank = 2;
  // Rank-2 tensor (matrix) of scalar type T.
  using Matrix = Eigen::TensorMap <
  Eigen::Tensor<T, kTensorRank, Eigen::RowMajor, Eigen::DenseIndex>,
  Eigen::Aligned>;

  using ConstMatrix = Eigen::TensorMap<
      Eigen::Tensor<const T, kTensorRank, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>;
};

template <typename T>
struct CopyRowsContext {
  int64_t skipped_rows = 0;
  int64_t row_size = 0;
  std::vector<ptrdiff_t> sizes;
  std::vector <std::shared_ptr<typename TTypes<T>::ConstMatrix>> inputs;
  std::shared_ptr<typename TTypes<T>::Matrix> output;
  T *out = nullptr;
  T *out_start = nullptr;
  T *out_end = nullptr;
 };

class ConcatV2CpuKernel : public CpuKernel {
public:
  ConcatV2CpuKernel() = default;

  ~ConcatV2CpuKernel() override = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

private:

  uint32_t CheckAndInitParams(const CpuKernelContext &ctx);

  uint32_t CheckConcatV2Params(const CpuKernelContext &ctx);

  uint32_t InitConcatV2Params(const CpuKernelContext &ctx);

  template<typename T>
  uint32_t PrepareInput(
      const CpuKernelContext &ctx,
      std::vector <std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs) {
    inputs.reserve(static_cast<size_t>(n_));
    output_concat_dim_ = 0;
    auto input0_shape_ptr = ctx.Input(0)->GetTensorShape();
    for (int64_t i = 0; i < n_; ++i) {
      Tensor *input_i_ptr = ctx.Input(static_cast<uint32_t>(i));
      KERNEL_CHECK_NULLPTR(input_i_ptr, KERNEL_STATUS_PARAM_INVALID,
                           "Get input x%u failed.", i);
      int64_t input_i_num = input_i_ptr->NumElements();
      if (input_i_num == 0) {
        continue;
      }
      auto input_i_shape_ptr = input_i_ptr->GetTensorShape();
      KERNEL_CHECK_NULLPTR(input_i_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                           "Get input x%u shape failed.", i);
      int32_t input_i_dims = input_i_shape_ptr->GetDims();
      KERNEL_CHECK_FALSE(
          (input_i_dims == input_dims_), KERNEL_STATUS_PARAM_INVALID,
          "Ranks of inputs should match: shape[0]=%d vs. shape[%u]=%d",
          input_dims_, i, input_i_dims);
      for (int32_t j = 0; j < input_dims_; ++j) {
        int64_t dim_ij = input_i_shape_ptr->GetDimSize(j);
        if (j == axis_) {
          int64_t dim_to_add = input_i_dims > 0 ? dim_ij : 1;
          output_concat_dim_ += dim_to_add;
          continue;
        }
        int64_t dim_0j = input0_shape_ptr->GetDimSize(j);
        KERNEL_CHECK_FALSE(
            (dim_0j == dim_ij), KERNEL_STATUS_PARAM_INVALID,
            "Dimensions of inputs should match: shape[0][%d]=%ld vs. shape[%u][%d]=%ld",
            j, dim_0j, i, j, dim_ij);
      }

      int64_t inputs_flat_dim1 = input_i_num / inputs_flat_dim0_;
      auto input_i_data_ptr = input_i_ptr->GetData();
      KERNEL_CHECK_NULLPTR(input_i_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                           "Get input x%u data failed.", i);
      auto input_i = std::make_shared<typename TTypes<T>::ConstMatrix>(
          static_cast<T *>(input_i_data_ptr), inputs_flat_dim0_,
          inputs_flat_dim1);
      KERNEL_CHECK_NULLPTR(input_i, KERNEL_STATUS_PARAM_INVALID,
                           "Create input x%u failed!", i);
      inputs.emplace_back(std::move(input_i));
    }

    if (input_dims_ == 0) {
      output_concat_dim_ = n_;
    }
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t PrepareOutput(CpuKernelContext &ctx,
                         std::shared_ptr<typename TTypes<T>::Matrix> &output) {
    Tensor *output_ptr = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output failed.");
    auto output_data_ptr = output_ptr->GetData();
    KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output data failed.");
    int64_t output_num = output_ptr->NumElements();
    int64_t output_dim1 = output_num / inputs_flat_dim0_;
    output = std::make_shared<typename TTypes<T>::Matrix>(
        static_cast<T *>(output_data_ptr), inputs_flat_dim0_, output_dim1);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                         "Create output matrix failed.");
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t DoCompute(CpuKernelContext &ctx) {
    std::vector <std::shared_ptr<typename TTypes<T>::ConstMatrix>> inputs;
    KERNEL_CHECK_FALSE((PrepareInput<T>(ctx, inputs) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "PrepareInput failed.");
    std::shared_ptr<typename TTypes<T>::Matrix> output = nullptr;
    KERNEL_CHECK_FALSE((PrepareOutput<T>(ctx, output) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "PrepareOutput failed.");
    if (inputs.size() > 0) {
      return ConcatV2Compute<T>(ctx, inputs, output);
    }
    KERNEL_LOG_INFO("ConcatV2CpuKernel success.");
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t PrepareConcatV2Sizes(
      const std::vector <std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs,
      std::vector <ptrdiff_t> &sizes, int64_t &row_size) {
    size_t num_inputs = inputs.size();
    sizes.reserve(num_inputs);
    row_size = 0;

    for (const auto &input: inputs) {
      sizes.push_back(input->dimension(1));
      row_size += sizes.back();
    }
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t CopyPartialRows(CopyRowsContext<T> &rows_ctx, uint32_t &ret) {
    if (rows_ctx.out >= rows_ctx.out_start) {
      return KERNEL_STATUS_OK;
    }
    for (size_t j = 0; j < rows_ctx.sizes.size(); ++j) {
      ptrdiff_t size = rows_ctx.sizes[j];
      ptrdiff_t offset = rows_ctx.out_start - rows_ctx.out;
      if (size <= offset) {
        rows_ctx.out += size;
        continue;
      }
      const T *inp = &(*rows_ctx.inputs[j])(rows_ctx.skipped_rows, 0);
      if (offset > 0) {
        rows_ctx.out += offset;
        inp += offset;
        size -= offset;
      }
      size = std::min(size, rows_ctx.out_end - rows_ctx.out);
      KERNEL_CHECK_FALSE_EXEC((size > 0),
      break)
      size_t copy_size = static_cast<size_t>(size) * sizeof(T);
      auto mem_ret = memcpy_s(rows_ctx.out, copy_size, inp, copy_size);
      if (mem_ret != EOK) {
        KERNEL_LOG_ERROR("Memcpy size[%zu] from inp[%p] to out[%p] failed.",
                         copy_size, rows_ctx.out, inp);
        ret = KERNEL_STATUS_INNER_ERROR;
        return KERNEL_STATUS_INNER_ERROR;
      }
      rows_ctx.out += size;
    }
    ++rows_ctx.skipped_rows;
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t CopyRemainRows(
      CopyRowsContext<T> &rows_ctx, uint32_t &ret) {
    std::vector<const T *> inp;
    inp.reserve(rows_ctx.inputs.size());
    for (const auto &input: rows_ctx.inputs) {
      inp.push_back(&(*input)(rows_ctx.skipped_rows, 0));
    }
    const int64_t dim0 = rows_ctx.output->dimension(0);
    for (int64_t i = rows_ctx.skipped_rows; i < dim0; ++i) {
      for (size_t j = 0; j < rows_ctx.inputs.size(); ++j) {
        ptrdiff_t size = std::min(rows_ctx.sizes[j], rows_ctx.out_end - rows_ctx.out);
        size_t copy_size = static_cast<size_t>(size) * sizeof(T);
        auto mem_ret = memcpy_s(rows_ctx.out, copy_size, inp[j], copy_size);
        if (mem_ret != EOK) {
          KERNEL_LOG_ERROR("Memcpy size[%zu] from inp[%p] to out[%p] failed.",
                           copy_size, inp[j], rows_ctx.out);
          ret = KERNEL_STATUS_INNER_ERROR;
          return KERNEL_STATUS_INNER_ERROR;
        }
        rows_ctx.out += size;
        inp[j] += size;
        KERNEL_CHECK_FALSE_EXEC((rows_ctx.out != rows_ctx.out_end),
        return KERNEL_STATUS_OK);
      }
    }
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t ExecConcatV2Tasks(
    const CpuKernelContext &ctx,
      const std::vector <std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs,
      std::shared_ptr<typename TTypes<T>::Matrix> &output,
      const std::vector <ptrdiff_t> &sizes, int64_t row_size) {
    uint32_t ret = KERNEL_STATUS_OK;
    auto work = [this, &row_size, &sizes, &inputs, &output, &ret](
        int64_t start, int64_t end) -> uint32_t {
      CopyRowsContext<T> rows_ctx;
      rows_ctx.skipped_rows = (row_size == 0) ? 0 : start / row_size;
      rows_ctx.row_size = row_size;
      rows_ctx.sizes = sizes;
      rows_ctx.inputs = inputs;
      rows_ctx.output = output;
      rows_ctx.out_start = output->data() + start;
      rows_ctx.out_end = output->data() + end;
      rows_ctx.out = output->data() + rows_ctx.skipped_rows * row_size;
      uint32_t result = this->CopyPartialRows<T>(rows_ctx, ret);
      if (ret != KERNEL_STATUS_OK || rows_ctx.out == rows_ctx.out_end) {
        return ret;
      }
      result = this->CopyRemainRows<T>(rows_ctx, ret);
      return result;
    };
    uint32_t result = CpuKernelUtils::ParallelFor(ctx, static_cast<int64_t>(output->size()),
                                                  static_cast<int64_t>(sizeof(T)), work);
    KERNEL_CHECK_FALSE((result == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR,
                       "ParallelFor exec failed, result[%u].", result);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR,
                       "ConcatV2CpuKernel failed.");
    KERNEL_LOG_INFO("ConcatV2CpuKernel success.");
    return KERNEL_STATUS_OK;
  }

  template<typename T>
  uint32_t ConcatV2Compute(
      const CpuKernelContext &ctx,
      const std::vector <std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs,
      std::shared_ptr<typename TTypes<T>::Matrix> &output) {
    std::vector <ptrdiff_t> sizes;
    int64_t row_size = 0;
    uint32_t ret = PrepareConcatV2Sizes<T>(inputs, sizes, row_size);
    KERNEL_CHECK_FALSE_EXEC((ret == KERNEL_STATUS_OK),
    return ret)
    ret = ExecConcatV2Tasks<T>(ctx, inputs, output, sizes, row_size);
    return ret;
  }

  DataType data_type_ = DT_DOUBLE;
  int32_t input_dims_ = 0;
  int64_t n_ = 0;
  int64_t output_concat_dim_ = 0;
  int64_t axis_ = 0;
  int64_t inputs_flat_dim0_ = 0;
};
}  // namespace aicpu
#endif //AICPU_KERNELS_CONCATV2_H_