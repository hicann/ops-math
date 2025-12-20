/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "clip_by_value_v2_aicpu.h"
#include <cstdio>
#include <complex>
#include <string>
#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char* const kClipByValueV2 = "ClipByValueV2";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const uint32_t kIndexTwo = 2;
constexpr int64_t kParallelDataNum = static_cast<int64_t>(256) * static_cast<int64_t>(1024);
constexpr int64_t kComplexParallelDataNum = static_cast<int64_t>(32) * static_cast<int64_t>(1024);
} // namespace
namespace aicpu {

template <typename T>
void CommonCompute(
    const CpuKernelContext& ctx, int64_t start, int64_t end, T (*y_index_compute_func)(T, T, T))
{
    auto input_x = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto input_min = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto input_max = reinterpret_cast<T*>(ctx.Input(kIndexTwo)->GetData());
    auto output_y = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    auto input_min_index = input_min;
    auto input_max_index = input_max;
    // NumElements() == 1兼容标量和shape为[1]的场景
    bool is_scalar = ctx.Input(1)->NumElements() == 1 ? true : false;

    for (int64_t i = start; i < end; i++) {
        auto x_index = input_x + i;
        auto y_index = output_y + i;
        if (!is_scalar) {
            input_min_index = input_min + i;
            input_max_index = input_max + i;
        }

        *y_index = y_index_compute_func(*x_index, *input_min_index, *input_max_index);
    }
}

template <typename T>
auto ComputeClamp(T x_index, T input_min_index, T input_max_index) -> T
{
    if (std::isnan(x_index) || std::isnan(input_min_index) || std::isnan(input_max_index)) {
        return static_cast<T>(NAN);
    }
    return std::min(std::max(x_index, input_min_index), input_max_index);
}

template <typename T>
auto ComputeComplexClamp(T x_index, T input_min_index, T input_max_index) -> T
{
    auto tmp = std::abs(x_index) <= std::abs(input_min_index) ? input_min_index : x_index;
    return std::abs(tmp) <= std::abs(input_max_index) ? tmp : input_max_index;
}

template <typename T>
void ComputeImpl(const CpuKernelContext& ctx, int64_t start, int64_t end)
{
    CommonCompute<T>(ctx, start, end, ComputeClamp<T>);
}

template <typename T>
uint32_t DoCompute(const CpuKernelContext& ctx)
{
    if (ctx.Input(0)->NumElements() == 1) {
        auto input_x = reinterpret_cast<T*>(ctx.Input(0)->GetData());
        auto input_min = reinterpret_cast<T*>(ctx.Input(1)->GetData());
        auto input_max = reinterpret_cast<T*>(ctx.Input(kIndexTwo)->GetData());
        auto output_y = reinterpret_cast<T*>(ctx.Output(0)->GetData());
        *output_y = std::min(std::max(*input_x, *input_min), *input_max);
        return KERNEL_STATUS_OK;
    }

    int64_t data_num = ctx.Input(0)->NumElements();
    if (data_num >= kParallelDataNum) {
        uint32_t min_core_num = 1;
        uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        auto sharde = [&ctx](size_t start, size_t end) { ComputeImpl<T>(ctx, start, end); };
        auto per_unit_size = CeilMultiple(data_num, max_core_num);
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, sharde), "Compute failed.");
    } else {
        ComputeImpl<T>(ctx, 0, data_num);
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
void ComplexComputeImpl(const CpuKernelContext& ctx, int64_t start, int64_t end)
{
    CommonCompute<T>(ctx, start, end, ComputeComplexClamp<T>);
}

template <typename T>
uint32_t DoComplexCompute(const CpuKernelContext& ctx)
{
    if (ctx.Input(0)->NumElements() == 1) {
        auto input_x = reinterpret_cast<T*>(ctx.Input(0)->GetData());
        auto input_min = reinterpret_cast<T*>(ctx.Input(1)->GetData());
        auto input_max = reinterpret_cast<T*>(ctx.Input(kIndexTwo)->GetData());
        auto output_y = reinterpret_cast<T*>(ctx.Output(0)->GetData());
        auto tmp = std::abs(*input_x) <= std::abs(*input_min) ? *input_min : *input_x;
        *output_y = std::abs(tmp) <= std::abs(*input_max) ? tmp : *input_max;
        return KERNEL_STATUS_OK;
    }

    int64_t data_num = ctx.Input(0)->NumElements();
    if (data_num >= kComplexParallelDataNum) {
        uint32_t min_core_num = 1;
        uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        auto sharde = [&ctx](size_t start, size_t end) { ComplexComputeImpl<T>(ctx, start, end); };
        auto per_unit_size = CeilMultiple(data_num, max_core_num);
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, sharde), "Compute failed.");
    } else {
        ComplexComputeImpl<T>(ctx, 0, data_num);
    }

    return KERNEL_STATUS_OK;
}

static std::map<DataType, std::function<uint32_t(const CpuKernelContext&)>> kg_clip_calls = {
    {DT_DOUBLE, DoCompute<double>},
    {DT_FLOAT, DoCompute<float>},
    {DT_FLOAT16, DoCompute<Eigen::half>},
    {DT_INT16, DoCompute<int16_t>},
    {DT_INT32, DoCompute<int32_t>},
    {DT_INT64, DoCompute<int64_t>},
    {DT_INT8, DoCompute<int8_t>},
    {DT_UINT8, DoCompute<uint8_t>},
    {DT_UINT16, DoCompute<uint16_t>},
    {DT_UINT32, DoCompute<uint32_t>},
    {DT_UINT64, DoCompute<uint64_t>},
    {DT_QUINT8, DoCompute<uint8_t>},
    {DT_QINT8, DoCompute<int8_t>},
    {DT_QINT32, DoCompute<int32_t>},
    {DT_BFLOAT16, DoCompute<Eigen::bfloat16>},
    {DT_COMPLEX64, DoComplexCompute<std::complex<float>>},
    {DT_COMPLEX128, DoComplexCompute<std::complex<double>>}};

uint32_t GetInputAndCheck(const CpuKernelContext& ctx)
{
    auto x_tensor = ctx.Input(0);
    auto min_tensor = ctx.Input(1);
    auto max_tensor = ctx.Input(kIndexTwo);
    auto y_tensor = ctx.Output(0);
    auto dtype = x_tensor->GetDataType();
    if ((dtype != min_tensor->GetDataType()) || (dtype != max_tensor->GetDataType()) ||
        (dtype != y_tensor->GetDataType())) {
        KERNEL_LOG_ERROR(
            "dtype is not same, x dtype is [%s], clip_value_min dtype is [%s], "
            "clip_value_max dtype is [%s], y dtype is [%s]",
            DTypeStr(dtype).c_str(), DTypeStr(min_tensor->GetDataType()).c_str(),
            DTypeStr(max_tensor->GetDataType()).c_str(), DTypeStr(y_tensor->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (x_tensor->NumElements() != y_tensor->NumElements()) {
        KERNEL_LOG_ERROR(
            "The numelements of the x [%ld] need be the same as the y [%ld].", x_tensor->NumElements(),
            y_tensor->NumElements());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if ((min_tensor->NumElements() == 1) && (max_tensor->NumElements() == 1)) {
        return KERNEL_STATUS_OK;
    }

    auto x_shape = x_tensor->GetTensorShape()->GetDimSizes();
    auto min_shape = min_tensor->GetTensorShape()->GetDimSizes();
    auto max_shape = max_tensor->GetTensorShape()->GetDimSizes();
    if (x_shape != min_shape || x_shape != max_shape) {
        KERNEL_LOG_ERROR(
            "input shape is not same, x_shape is [%s], min_shape is [%s], max_shape is [%s]",
            VectorToString(x_shape).c_str(), VectorToString(min_shape).c_str(), VectorToString(max_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ClipByValueV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check ClipByValueV2 params failed.");
    if (ctx.Input(0)->NumElements() == 0) {
        KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_OK;
    }

    KERNEL_HANDLE_ERROR(GetInputAndCheck(ctx), "Check ClipByValueV2 params failed.");

    auto input_dtype = ctx.Input(0)->GetDataType();
    const auto& func = kg_clip_calls.find(input_dtype);
    if (func != kg_clip_calls.end()) {
        return (func->second)(ctx);
    } else {
        KERNEL_LOG_ERROR("input type[%d] not support", input_dtype);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kClipByValueV2, ClipByValueV2CpuKernel);
} // namespace aicpu