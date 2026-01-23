/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_ADD_H
#define AICPU_KERNELS_ADD_H
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class AddCpuKernel : public CpuKernel {
 public:
  AddCpuKernel() = default;
  ~AddCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief compute for all types
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T>
  uint32_t AddCompute(const CpuKernelContext &ctx) const;

  /**
   * @brief Check if input&output addr is aligned
   * @param calc_info data used to calculate
   * @return true: aligned, false: not aligned
   */
  bool AlignedCheck(const BCalcInfo &calc_info) const;
  
  template <typename T>
  uint32_t AddCalculateWithRankCheck(const CpuKernelContext &ctx, BCalcInfo &calc_info) const;

  template <int32_t RANK, typename T>
  uint32_t AddCalculateWithAlignedCheck(const CpuKernelContext &ctx, BCalcInfo &calc_info) const;

  /**
   * @brief Eigen calculate for all types
   * @param calc_info data used to calculate
   */
  template <int32_t RANK, typename T, int32_t OPTION>
  uint32_t AddCalculate(BCalcInfo &calc_info) const;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_ADD_H_
