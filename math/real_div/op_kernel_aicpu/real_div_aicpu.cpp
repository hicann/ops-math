/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "real_div_aicpu.h"

#include <cstdint>
#include <vector>

#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
const char *const kDIV = "Div";
const char *const kRealDiv = "RealDiv";
constexpr int32_t kDim0 = 0;
constexpr int32_t kDim1 = 1;
constexpr int32_t kDim2 = 2;
constexpr int32_t kDim3 = 3;
constexpr int32_t kDim4 = 4;
constexpr int32_t kDim5 = 5;
constexpr int32_t kDim6 = 6;
constexpr int32_t kDim7 = 7;
constexpr int32_t kDim8 = 8;
}  // namespace

namespace aicpu {
uint32_t RealDivKernel::RealDivSameTypeCompute(const CpuKernelContext &ctx, DataType data_type) {
  switch (data_type) {
    case DT_FLOAT16:
      return RealDivCompute<Eigen::half>(ctx, false);
    case DT_FLOAT:
      return RealDivCompute<float>(ctx, false);
    case DT_DOUBLE:
      return RealDivCompute<double>(ctx, false);
    case DT_INT8:
      return RealDivCompute<int8_t>(ctx);
    case DT_INT16:
      return RealDivCompute<int16_t>(ctx);
    case DT_INT32:
      return RealDivCompute<int32_t>(ctx);
    case DT_INT64:
      return RealDivCompute<int64_t>(ctx);
    case DT_UINT8:
      return RealDivCompute<uint8_t>(ctx);
    case DT_UINT16:
      return RealDivCompute<uint16_t>(ctx);
    case DT_UINT32:
      return RealDivCompute<uint32_t>(ctx);
    case DT_UINT64:
      return RealDivCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return RealDivCompute<std::complex<float>>(ctx, false);
    case DT_COMPLEX128:
      return RealDivCompute<std::complex<double>>(ctx, false);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
bool RealDivKernel::IsInputHasZero(T *input_data, const int64_t num_of_elems) {
  std::vector<T> check_data(input_data, input_data + num_of_elems);
  return std::any_of(check_data.begin(), check_data.end(), [](T &cur_data) {
    return IsValueEqual<T>(cur_data, T(0));
  });
}

template <typename T>
uint32_t RealDivKernel::DispatchByRank(int32_t rank, BCalcInfo &calc_info, const CpuKernelContext &ctx) {
    switch (rank) {
        case kDim0: {
            T v0 = *(reinterpret_cast<const T*>(calc_info.input_0->GetData()));
            T v1 = *(reinterpret_cast<const T*>(calc_info.input_1->GetData()));
            T* value_out = reinterpret_cast<T*>(calc_info.output->GetData());
            *(value_out) = v0 / v1;
            return KERNEL_STATUS_OK;
        }
        case kDim1:
            return RealDivCalculateWithAlignedCheck<kDim1, T>(calc_info);
        case kDim2:
            return RealDivCalculateWithAlignedCheck<kDim2, T>(calc_info);
        case kDim3:
            return RealDivCalculateWithAlignedCheck<kDim3, T>(calc_info);
        case kDim4:
            return RealDivCalculateWithAlignedCheck<kDim4, T>(calc_info);
        case kDim5:
            return RealDivCalculateWithAlignedCheck<kDim5, T>(calc_info);
        case kDim6:
            return RealDivCalculateWithAlignedCheck<kDim6, T>(calc_info);
        case kDim7:
            return RealDivCalculateWithAlignedCheck<kDim7, T>(calc_info);
        case kDim8:
            return RealDivCalculateWithAlignedCheck<kDim8, T>(calc_info);
        default:
            KERNEL_LOG_ERROR(
                "[%s] Rank of output should less than 8 but get [%zu].", ctx.GetOpType().c_str(),
                calc_info.shape_out.size());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

template <typename T>
uint32_t RealDivKernel::RealDivCompute(const CpuKernelContext &ctx, const bool verify_zero) {
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  if (verify_zero && IsInputHasZero<T>(static_cast<T *>(input1->GetData()), input1->NumElements())) {
    KERNEL_LOG_ERROR("Invalid argument, division by zero.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = input1;
  calc_info.output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 0 data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 1 data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get output data failed", ctx.GetOpType().c_str())
  KERNEL_LOG_INFO(
      "[%s] Input[0] data size is [%lu], input[1] data size is [%lu], "
      "output data size is [%lu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetDataSize());
  // broadcast input
  Bcast bcast;
  if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  bcast.GetBcastVec(calc_info);
  int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());
  return DispatchByRank<T>(rank, calc_info, ctx);
}

template <int32_t RANK, typename T>
uint32_t RealDivKernel::RealDivCalculateWithAlignedCheck(BCalcInfo &calc_info) {
  if (AlignedCheck(calc_info)) {
    return RealDivCalculate<RANK, T, Eigen::Aligned>(calc_info);
  }
  return RealDivCalculate<RANK, T, Eigen::Unaligned>(calc_info);
}

bool RealDivKernel::AlignedCheck(const BCalcInfo &calc_info) const {
  return AddrAlignedCheck(calc_info.input_0->GetData()) &&
         AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
uint32_t RealDivKernel::RealDivCalculate(BCalcInfo &calc_info) {
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(
      static_cast<T *>(calc_info.input_0->GetData()),
      calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(
      static_cast<T *>(calc_info.input_1->GetData()),
      calc_info.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(
      static_cast<T *>(calc_info.output->GetData()),
      calc_info.output->GetTensorShape()->NumElements());
  auto input_shape_0 = calc_info.input_0->GetTensorShape()->GetDimSizes();
  auto input_shape_1 = calc_info.input_1->GetTensorShape()->GetDimSizes();
  if (input_shape_0.empty()) {
    T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
    output = v0 / input1;
    return KERNEL_STATUS_OK;
  }

  if (input_shape_1.empty()) {
    T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
    output = input0 / v1;
    return KERNEL_STATUS_OK;
  }

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_1;

  for (int32_t i = 0; i < RANK; i++) {
    reshape_0[(RANK - i) - 1] = calc_info.reshape_0[i];
    reshape_1[(RANK - i) - 1] = calc_info.reshape_1[i];
    shape_out[(RANK - i) - 1] = calc_info.shape_out[i];
    bcast_0[(RANK - i) - 1] = calc_info.bcast_0[i];
    bcast_1[(RANK - i) - 1] = calc_info.bcast_1[i];
  }
  if (input_shape_0 == input_shape_1) {
    output.reshape(shape_out) =
        input0.reshape(reshape_0) / input1.reshape(reshape_1);
  }
  else {
    output.reshape(shape_out) =
        input0.reshape(reshape_0).broadcast(bcast_0) / input1.reshape(reshape_1).broadcast(bcast_1);
  }
  return KERNEL_STATUS_OK;
}

uint32_t RealDivKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, INPUT_NUM2, 1), "Div check input output params failed.");
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }
  
  DataType input0_type = input0->GetDataType();
  DataType input1_type = input1->GetDataType();
  KERNEL_CHECK_FALSE((
    input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID, "input0 type[%s] is not equal to input1 type[%s]",
                      DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str());
  return RealDivSameTypeCompute(ctx, input0_type);
}

REGISTER_CPU_KERNEL(kRealDiv, RealDivKernel);
REGISTER_CPU_KERNEL(kDIV, RealDivKernel);
}  // namespace aicpu