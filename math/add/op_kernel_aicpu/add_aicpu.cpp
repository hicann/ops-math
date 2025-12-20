/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_aicpu.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const char *const kAdd = "Add";
}
namespace aicpu {
uint32_t CheckPermissionType(const CpuKernelContext &ctx) {
  const DataType input0_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  const DataType input1_data_type = ctx.Input(kSecondInputIndex)->GetDataType();
  const DataType output_data_type = ctx.Output(kFirstOutputIndex)->GetDataType();
  if (input0_data_type != output_data_type) {
    KERNEL_LOG_ERROR("Output must have the same type as input, but got input0[%s] input1[%s] output[%s].",
                     DTypeStr(input0_data_type).c_str(), DTypeStr(input1_data_type).c_str(),
                     DTypeStr(output_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t AddCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalMathCheck(ctx), "Normal match check params failed.");
  KERNEL_HANDLE_ERROR(CheckPermissionType(ctx), "Type permission check failed.");
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }

  // choose compute function depend on dataType
  const DataType input0_data_type = input0->GetDataType();
  switch (input0_data_type) {
    case DT_FLOAT16:
      return AddCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return AddCompute<float>(ctx);
    case DT_DOUBLE:
      return AddCompute<double>(ctx);
    case DT_INT8:
      return AddCompute<int8_t>(ctx);
    case DT_INT16:
      return AddCompute<int16_t>(ctx);
    case DT_INT32:
      return AddCompute<int32_t>(ctx);
    case DT_INT64:
      return AddCompute<int64_t>(ctx);
    case DT_UINT8:
      return AddCompute<uint8_t>(ctx);
    case DT_UINT16:
      return AddCompute<uint16_t>(ctx);
    case DT_UINT32:
      return AddCompute<uint32_t>(ctx);
    case DT_UINT64:
      return AddCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return AddCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return AddCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(input0_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t AddCpuKernel::AddCompute(const CpuKernelContext &ctx) const{
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);

  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[0] data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[1] data failed",
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
  return AddCalculateWithRankCheck<T>(ctx, calc_info);
}

template <typename T>
uint32_t AddCpuKernel::AddCalculateWithRankCheck(const CpuKernelContext &ctx, BCalcInfo &calc_info) const {
  int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());
  switch (rank) {
    case 0:
    {
      T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
      T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
      T *value_out = reinterpret_cast<T *>(calc_info.output->GetData());
      *(value_out) = v0 + v1;
      return KERNEL_STATUS_OK;
    }
    case kRank1:
      return AddCalculateWithAlignedCheck<kRank1, T>(ctx, calc_info);
    case kRank2:
      return AddCalculateWithAlignedCheck<kRank2, T>(ctx, calc_info);
    case kRank3:
      return AddCalculateWithAlignedCheck<kRank3, T>(ctx, calc_info);
    case kRank4:
      return AddCalculateWithAlignedCheck<kRank4, T>(ctx, calc_info);
    case kRank5:
      return AddCalculateWithAlignedCheck<kRank5, T>(ctx, calc_info);
    case kRank6:
      return AddCalculateWithAlignedCheck<kRank6, T>(ctx, calc_info);
    case kRank7:
      return AddCalculateWithAlignedCheck<kRank7, T>(ctx, calc_info);
    case kRank8:
      return AddCalculateWithAlignedCheck<kRank8, T>(ctx, calc_info);
    default:
      KERNEL_LOG_ERROR("[%s] Rank of output should less than 8 but get [%zu].",
                       ctx.GetOpType().c_str(), calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <int32_t RANK, typename T>
uint32_t AddCpuKernel::AddCalculateWithAlignedCheck(const CpuKernelContext &ctx, BCalcInfo &calc_info) const {
  (void)ctx;
  if (AlignedCheck(calc_info)) {
    return AddCalculate<RANK, T, Eigen::Aligned>(calc_info);
  }
  return AddCalculate<RANK, T, Eigen::Unaligned>(calc_info);
}

bool AddCpuKernel::AlignedCheck(const BCalcInfo &calc_info) const {
  return AddrAlignedCheck(calc_info.input_0->GetData()) &&
         AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
uint32_t AddCpuKernel::AddCalculate(BCalcInfo &calc_info) const {
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
    output = v0 + input1;
    return KERNEL_STATUS_OK;
  }

  if (input_shape_1.empty()) {
    T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
    output = input0 + v1;
    return KERNEL_STATUS_OK;
  }

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast1;

  for (int32_t i = 0; i < RANK; i++) {
    reshape0[(RANK - i) - 1] = calc_info.reshape_0[i];
    reshape1[(RANK - i) - 1] = calc_info.reshape_1[i];
    shape_out[(RANK - i) - 1] = calc_info.shape_out[i];
    bcast0[(RANK - i) - 1] = calc_info.bcast_0[i];
    bcast1[(RANK - i) - 1] = calc_info.bcast_1[i];
  }
  output.reshape(shape_out) =
      input0.reshape(reshape0).broadcast(bcast0) + input1.reshape(reshape1).broadcast(bcast1);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAdd, AddCpuKernel);
}  // namespace aicpu
