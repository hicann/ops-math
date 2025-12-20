/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cumsum_aicpu.h"
#include <iostream>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kCumsumInputNum = 2;
const uint32_t kCumsumOutputNum = 1;
const int64_t kParalledDataSize = static_cast<int64_t>(512 * 1024);
const char* const kCumsum = "Cumsum";
#define CUMSUM_COMPUTE_CASE(DTYPE, TYPE, CTX)                  \
    case (DTYPE): {                                            \
        uint32_t result = CumsumCompute<TYPE>(CTX);            \
        if (result != KERNEL_STATUS_OK) {                      \
            KERNEL_LOG_ERROR("Cumsum kernel compute failed."); \
            return result;                                     \
        }                                                      \
        break;                                                 \
    }
    
#define CUMSUM_COMPUTE_CASE_COMPLEX(DTYPE, TYPE, IN_TYPE, CTX) \
    case (DTYPE): {                                            \
        uint32_t result = CumsumCompute2<TYPE, IN_TYPE>(CTX);  \
        if (result != KERNEL_STATUS_OK) {                      \
            KERNEL_LOG_ERROR("Cumsum kernel compute failed."); \
            return result;                                     \
        }                                                      \
        break;                                                 \
    }
} // namespace

namespace aicpu {
void CumsumCpuKernel::AxesCal(
    CpuKernelContext& ctx, int64_t& inner, int64_t& outer, int64_t& depth, const int32_t& axis)
{
    auto shape = ctx.Input(kFirstInputIndex)->GetTensorShape();
    const int64_t rank = shape->GetDims();
    for (int32_t i = 0; i < rank; ++i) {
        if (i < axis) {
            inner *= shape->GetDimSize(i);
        } else if (i > axis) {
            outer *= shape->GetDimSize(i);
        } else {
            depth = shape->GetDimSize(i);
        }
    }
}
uint32_t CumsumCpuKernel::Compute(CpuKernelContext& ctx)
{
    // check params
    KERNEL_HANDLE_ERROR(
        NormalCheck(ctx, kCumsumInputNum, kCumsumOutputNum), "[%s] check input and output failed.", kCumsum);
    // parse params
    KERNEL_HANDLE_ERROR(CumsumCheck(ctx), "[%s] check params failed.", kCumsum);
    auto input_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
    switch (input_data_type) {
        CUMSUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        CUMSUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
        CUMSUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
        CUMSUM_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
        CUMSUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX64, std::complex<float>, float, ctx)
        CUMSUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX128, std::complex<double>, double, ctx)
        default:
            KERNEL_LOG_ERROR("Cumsum kernel data type [%s] not support.", DTypeStr(input_data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}
uint32_t CumsumCpuKernel::CumsumCheck(const CpuKernelContext& ctx)
{
    KERNEL_CHECK_NULLPTR(ctx.Input(kFirstInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input failed.");
    KERNEL_CHECK_NULLPTR(ctx.Output(kFirstInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get output failed.");

    if (ctx.Input(1)->GetData() != nullptr) {
        KERNEL_CHECK_FALSE(
            (ctx.Input(1)->GetDataType() == DT_INT32 || ctx.Input(1)->GetDataType() == DT_INT64),
            KERNEL_STATUS_PARAM_INVALID, "Data type of axis is not support, axis data type is [%u].",
            ctx.Input(1)->GetDataType());
        KERNEL_CHECK_FALSE(ctx.Input(1)->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID, "axis is out of shape")
    }
    std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();

    KERNEL_CHECK_FALSE(
        (shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
        "The output shape size should be same as the output shape size")
    return KERNEL_STATUS_OK;
}

void CumsumCpuKernel::CumsumGetAttr(const CpuKernelContext& ctx, bool& exclusive, bool& reverse) const
{
    exclusive = false;
    AttrValue* exclusive_attr = ctx.GetAttr("exclusive");
    if (exclusive_attr != nullptr) {
        exclusive = exclusive_attr->GetBool();
    }

    reverse = false;
    AttrValue* reverse_attr = ctx.GetAttr("reverse");
    if (reverse_attr != nullptr) {
        reverse = reverse_attr->GetBool();
    }
}

template <typename T>
void computeOutPut(
    T* input_data, T* output_data, int64_t inner, int64_t outer, int64_t depth, int64_t start, int64_t end,
    bool reverse, bool exclusive)
{
    for (int64_t outer_index = start; outer_index < end; ++outer_index) {
        int64_t outer_index_adj = reverse ? (outer - 1) - outer_index : outer_index;

        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
            auto accumulator = static_cast<T>(0);
            int64_t inner_index_adj = reverse ? (inner - 1) - inner_index : inner_index;

            for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
                int64_t depth_index_adj = reverse ? (depth - 1) - depth_index : depth_index;

                int64_t index = outer_index_adj;
                index += inner_index_adj * depth * outer + depth_index_adj * outer;
                if (exclusive) {
                    output_data[index] = accumulator;
                    accumulator += input_data[index];
                } else {
                    accumulator += input_data[index];
                    output_data[index] = accumulator;
                }
            }
        }
    }
}

template <typename T, typename T2>
void computeOutPut4Complex(
    std::vector<T2>& input_data_real, std::vector<T2>& input_data_imag, T* output_data, int64_t inner, int64_t outer,
    int64_t depth, int64_t start, int64_t end, bool reverse, bool exclusive)
{
    for (int64_t outer_index = start; outer_index < end; ++outer_index) {
        int64_t outer_index_adj = reverse ? (outer - 1) - outer_index : outer_index;

        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
            auto accumulator_real = static_cast<T2>(0);
            auto accumulator_imag = static_cast<T2>(0);
            int64_t inner_index_adj = reverse ? (inner - 1) - inner_index : inner_index;

            for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
                int64_t depth_index_adj = reverse ? (depth - 1) - depth_index : depth_index;

                int64_t index = outer_index_adj;
                index += inner_index_adj * depth * outer;
                index += depth_index_adj * outer;
                if (exclusive) {
                    output_data[index] = std::complex<T2>(accumulator_real, accumulator_imag);
                    accumulator_real += input_data_real[index];
                    accumulator_imag += input_data_imag[index];
                } else {
                    accumulator_real += input_data_real[index];
                    accumulator_imag += input_data_imag[index];
                    output_data[index] = std::complex<T2>(accumulator_real, accumulator_imag);
                }
            }
        }
    }
}

template <typename T>
uint32_t CumsumCpuKernel::CumsumCompute(CpuKernelContext& ctx)
{
    auto input_data = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto output_data = reinterpret_cast<T*>(ctx.Output(0)->GetData());

    if ((ctx.Input(0)->GetTensorShape()->GetDims() == 0) && (ctx.Input(0)->NumElements() == 1)) {
        output_data[0] = input_data[0];
        return KERNEL_STATUS_OK;
    }
    auto shape = ctx.Input(kFirstInputIndex)->GetTensorShape();
    auto axis_data = reinterpret_cast<int32_t*>(ctx.Input(1)->GetData());
    bool exclusive;
    bool reverse;
    int32_t axis = 0;
    if (axis_data != nullptr) {
        axis = *axis_data;
    }
    CumsumGetAttr(ctx, exclusive, reverse);
    if (axis < 0) {
        axis += shape->GetDims();
    }
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t depth = 1;
    AxesCal(ctx, inner, outer, depth, axis);
    int64_t data_num = ctx.Input(kFirstInputIndex)->NumElements();
    int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
    if (data_size <= kParalledDataSize) {
        computeOutPut<T>(
            input_data, output_data, inner, outer, depth, static_cast<int64_t>(0), outer, reverse, exclusive);
    } else {
        auto shard_cumsum = [&](int64_t start, int64_t end) {
            computeOutPut<T>(input_data, output_data, inner, outer, depth, start, end, reverse, exclusive);
        };
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        if (max_core_num > outer) {
            max_core_num = outer;
        }
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, outer, outer / max_core_num, shard_cumsum), "CumSum Compute failed.")
    }
    return KERNEL_STATUS_OK;
}
template <typename T, typename T2>
uint32_t CumsumCpuKernel::CumsumCompute2(CpuKernelContext& ctx)
{
    auto input_data = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto output_data = reinterpret_cast<T*>(ctx.Output(0)->GetData());

    if ((ctx.Input(0)->GetTensorShape()->GetDims() == 0) && (ctx.Input(0)->NumElements() == 1)) {
        output_data[0].real(input_data[0].real());
        output_data[0].imag(input_data[0].imag());
        return KERNEL_STATUS_OK;
    }

    auto axis_data = reinterpret_cast<int32_t*>(ctx.Input(1)->GetData());
    bool exclusive;
    bool reverse;
    int32_t axis = 0;
    if (axis_data != nullptr) {
        axis = *axis_data;
    }

    CumsumGetAttr(ctx, exclusive, reverse);

    auto shape = ctx.Input(0)->GetTensorShape();
    if (axis < 0) {
        axis += shape->GetDims();
    }
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t depth = 1;

    AxesCal(ctx, inner, outer, depth, axis);
    int64_t data_num = ctx.Input(0)->NumElements();
    std::vector<T2> input_data_real(data_num);
    std::vector<T2> input_data_imag(data_num);
    for (int64_t i = 0; i < data_num; ++i) {
        input_data_real[i] = input_data[i].real();
        input_data_imag[i] = input_data[i].imag();
    }
    int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
    if (data_size <= kParalledDataSize) {
        computeOutPut4Complex<T, T2>(
            input_data_real, input_data_imag, output_data, inner, outer, depth, static_cast<int64_t>(0), outer, reverse, exclusive);
    } else {
        auto shard_cumsum = [&](int64_t start, int64_t end) {
            computeOutPut4Complex<T, T2>(
                input_data_real, input_data_imag, output_data, inner, outer, depth, start, end, reverse, exclusive);
        };
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        if (max_core_num > outer) {
            max_core_num = outer;
        }
        KERNEL_CHECK_FALSE(
            (max_core_num != 0), KERNEL_STATUS_PARAM_INVALID,
            "The max core num is zero, please check input 0 dim size.");
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, outer, outer / max_core_num, shard_cumsum), "CumSum Compute failed.")
    }
    return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCumsum, CumsumCpuKernel);
} // namespace aicpu
