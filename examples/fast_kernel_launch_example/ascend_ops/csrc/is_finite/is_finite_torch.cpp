/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file is_finite_torch.cce
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"

// 直接调用math目录中已经实现的算子公共逻辑
#include "math/is_finite/op_kernel/is_finite.h"
#include "math/is_finite/op_host/is_finite_tiling_common.h"

namespace ascend_ops {

namespace IsFinite {

using namespace IsFiniteNs;

template <typename T>
__global__ __aicore__ void isfinite_kernel(__gm__ uint8_t* x, __gm__ uint8_t* y, const IsFiniteTilingData tilingData)
{
    if constexpr (std::is_same_v<T, c10::Half>) {
        IsFiniteKernelImpl<IS_FINITE_TPL_FP16, IS_FINITE_TPL_BOOL>(x, y, &tilingData);
        return;
    }
    if constexpr (std::is_same_v<T, c10::BFloat16>) {
        IsFiniteKernelImpl<IS_FINITE_TPL_BF16, IS_FINITE_TPL_BOOL>(x, y, &tilingData);
        return;
    }
    if constexpr (std::is_same_v<T, float>) {
        IsFiniteKernelImpl<IS_FINITE_TPL_FP32, IS_FINITE_TPL_BOOL>(x, y, &tilingData);
        return;
    }
}

template <typename T>
void isfinite_api(aclrtStream stream, const at::Tensor& x, const at::Tensor& y)
{
    int64_t num_element = x.numel();
    IsFiniteTilingData tilingData;
    IsFiniteTiling::IsFiniteCommonTiling<at::Tensor>(x, tilingData);
    uint32_t blockDim = tilingData.needCoreNum;
    auto x_ptr = x.data_ptr<T>();
    auto y_ptr = y.data_ptr<bool>();
    isfinite_kernel<T><<<blockDim, nullptr, stream>>>((__gm__ uint8_t*)x_ptr, (__gm__ uint8_t*)y_ptr, tilingData);
}

template <>
void isfinite_api<double>(aclrtStream stream, const at::Tensor& x, const at::Tensor& y)
{
    throw std::runtime_error("double is not supported on aicore!");
}

torch::Tensor isfinite_npu(const torch::Tensor& x)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "Input tensor must be on NPU device");
    TORCH_CHECK(x.scalar_type() != at::kDouble, "Double type is not supported by isfinite_npu");
    at::Tensor y = at::empty_like(x, at::dtype(at::kBool));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "isfinite_npu", [&] { isfinite_api<scalar_t>(stream, x, y); });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("IsFinite", acl_call);
    return y;
}

torch::Tensor isfinite_meta(const torch::Tensor& x)
{
    TORCH_CHECK(x.defined(), "Input tensor must be defined");
    return torch::empty(
        x.sizes(),
        torch::TensorOptions().dtype(torch::kBool).device(torch::kMeta).memory_format(x.suggest_memory_format()));
}

// Register Ascend implementations for isfinite
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("isfinite", isfinite_npu);
}

// Register Meta Function for isfinite
TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("isfinite", TORCH_FN(isfinite_meta));
}

} // namespace IsFinite
} // namespace ascend_ops
