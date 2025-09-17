/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_SEARCH_SORTED_H
#define AICPU_KERNELS_NORMALIZED_SEARCH_SORTED_H

#include <type_traits>
#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {

class SearchSortedKernel : public CpuKernel {
 public:
  ~SearchSortedKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  KernelStatus GetInputAndCheck(const CpuKernelContext &ctx);

  bool right_ = false;
  DataType sequence_dtype_ = DT_INT32;
  DataType values_dtype_ = DT_INT32;
  DataType output_dtype_ = DT_INT32;

  Tensor *sequence_t_ = nullptr;
  Tensor *values_t_ = nullptr;
  Tensor *sorter_t_ = nullptr;
  Tensor *output_t_ = nullptr;
};
}  // namespace aicpu
#endif
