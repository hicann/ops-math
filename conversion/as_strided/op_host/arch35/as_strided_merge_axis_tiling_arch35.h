/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file as_strided_prepare_tiling.h
 * \brief as_strided_prepare_tiling
 */
#include "as_strided_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "op_host/util/const_util.h"
#include "op_host/tiling_util.h"

namespace optiling {
constexpr size_t IN_X = 0;
constexpr size_t IN_SIZE = 1;
constexpr size_t IN_STRIDE = 2;
constexpr size_t IN_OFFSET = 3;

static bool GetSizeAndStride(gert::TilingContext* context, AsStridedRunInfo& run_info) {
  OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context, IN_SIZE, run_info.out_size),
                  OP_LOGE(context->GetNodeName(), "get const of size failed"), return false);
  OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context, IN_STRIDE, run_info.out_stride),
                  OP_LOGE(context->GetNodeName(), "get const of stride failed"), return false);

  OP_CHECK_IF(run_info.out_size.GetDimNum() != run_info.out_stride.GetDimNum(),
                  OP_LOGE(
                      context->GetNodeName(), "the dimension count of size and stride should be same! but %zu Vs %zu",
                      run_info.out_size.GetDimNum(), run_info.out_stride.GetDimNum()),
                  return false);

  const int64_t ori_size_len = run_info.out_size.GetDimNum();
  OP_CHECK_IF(
      ori_size_len == 0,
      OP_LOGE(context->GetNodeName(), "the dimension count should be bigger than 0, but is 0!"),
      return false);

  return true;
}

static void MergeAxisRule1(AsStridedRunInfo& run_info) {
  // merge rule 1: delete axis which size is 1
  const int64_t ori_size_len = run_info.out_size.GetDimNum();
  size_t size_idx = 0;
  for (int64_t i = 0; i < ori_size_len; i++) {
    if (run_info.out_size.GetDim(i) != 1) {
      run_info.out_size.SetDim(size_idx, run_info.out_size.GetDim(i));
      run_info.out_stride.SetDim(size_idx, run_info.out_stride.GetDim(i));
      size_idx += 1UL;
    }
  }

  run_info.out_size.SetDimNum(size_idx);
  run_info.out_stride.SetDimNum(size_idx);
}

static void MergeAxisRule2(AsStridedRunInfo& run_info) {
  // merge rule 2: merge the axes with continuous stride by 0
  const size_t size_idx = run_info.out_size.GetDimNum();
  size_t cu_merge_size_idx = 0;
  for (size_t i = 1; i < size_idx; i++) {
    const int64_t stride_value = run_info.out_stride.GetDim(i);
    const int64_t previous_stride_value = run_info.out_stride.GetDim(cu_merge_size_idx);
    if (stride_value == 0 && previous_stride_value == 0) {
      run_info.out_size.SetDim(cu_merge_size_idx,
                               run_info.out_size.GetDim(cu_merge_size_idx) * run_info.out_size.GetDim(i));
      continue;
    }

    cu_merge_size_idx += 1UL;
    run_info.out_size.SetDim(cu_merge_size_idx, run_info.out_size.GetDim(i));
    run_info.out_stride.SetDim(cu_merge_size_idx, run_info.out_stride.GetDim(i));
  }
  run_info.out_size.SetDimNum(cu_merge_size_idx + 1UL);
  run_info.out_stride.SetDimNum(cu_merge_size_idx + 1UL);
}

static void MergeAxisRule3(AsStridedRunInfo& run_info) {
  // merge rule 4: merge dims that is continuous stride except last dim
  const size_t out_dim_num = run_info.out_size.GetDimNum();

  const size_t cu_merge_size_idx = out_dim_num - 1;
  int64_t last_dim_merge_value =
      (run_info.out_stride.GetDim(cu_merge_size_idx) * run_info.out_size.GetDim(cu_merge_size_idx));
  int64_t last_dim_merge_size = run_info.out_size.GetDim(cu_merge_size_idx);
  int64_t last_dim_merge_stride = run_info.out_stride.GetDim(cu_merge_size_idx);
  size_t last_dim_merge_dim = cu_merge_size_idx;
  for (size_t i = 0; i < cu_merge_size_idx; i++) {
    if (last_dim_merge_value != run_info.out_stride.GetDim(cu_merge_size_idx - 1UL - i)) {
      run_info.out_size.SetDim(last_dim_merge_dim, last_dim_merge_size);
      run_info.out_stride.SetDim(last_dim_merge_dim, last_dim_merge_stride);
      last_dim_merge_dim--;
      last_dim_merge_value =
          (run_info.out_stride.GetDim(cu_merge_size_idx - 1 - i) * run_info.out_size.GetDim(cu_merge_size_idx - 1 - i));
      last_dim_merge_size = run_info.out_size.GetDim(cu_merge_size_idx - 1UL - i);
      last_dim_merge_stride = run_info.out_stride.GetDim(cu_merge_size_idx - 1UL - i);
      continue;
    }
    last_dim_merge_value *= run_info.out_size.GetDim(cu_merge_size_idx - 1UL - i);
    last_dim_merge_size *= run_info.out_size.GetDim(cu_merge_size_idx - 1UL - i);
  }
  run_info.out_size.SetDim(last_dim_merge_dim, last_dim_merge_size);
  run_info.out_stride.SetDim(last_dim_merge_dim, last_dim_merge_stride);

  if (last_dim_merge_dim != 0UL) {
    for (size_t i = last_dim_merge_dim, j = 0; i < out_dim_num; i++, j++) {
      run_info.out_size[j] = run_info.out_size[i];
      run_info.out_stride[j] = run_info.out_stride[i];
    }
    run_info.out_size.SetDimNum(out_dim_num - last_dim_merge_dim);
    run_info.out_stride.SetDimNum(out_dim_num - last_dim_merge_dim);
  }
}

static void MergeAxis(AsStridedRunInfo& run_info) {
  // merge rule 1: delete axis which size is 1
  MergeAxisRule1(run_info);

  // 规则1会将全1的shape，如[1, 1, 1]合为空的[]，但规则2能够恢复一维[1]
  // merge rule 2: merge the axes with continuous stride by 0
  /***
   *** stride[0, 0, 1, 0, 0] -> stride[0, 1, 0]
   *** size[1, 2, 3, 4, 5] -> size[2, 3, 20]
   ***/
  MergeAxisRule2(run_info);

  // merge rule 3: merge dims that is continuous stride
  /***
   *** stride[2, 18, 6, 3] -> stride[2, 3]
   *** size[3, 4, 3, 2] -> size[3, 24]

   *** stride[2, 18, 6, 3, 1] -> stride[2, 3, 1]
   *** size[3, 4, 3, 2, 10] -> size[3, 24, 10]
   ***/
  MergeAxisRule3(run_info);
}
} // namespace