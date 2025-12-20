/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

#undef private
#undef protected

#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_CONCATV2_UT : public testing::Test {
};

template<typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *) input0->GetData();
  int32_t inputs_size = node_def.InputsSize() - 1;
  int64_t input0_num = input0->NumElements();
  for (int32_t i = 0; i < inputs_size; ++i) {
    for (int64_t j = 0; j < input0_num; ++j) {
      int64_t index = input0_num * i + j;
      expect_out[index] = input0_data[j];
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas, builder)                 \
  do {                                                                     \
    (builder).Input({"x0", data_types[0], shapes[0], datas[0]})            \
            .Input({"x1", data_types[1], shapes[1], datas[1]})             \
            .Input({"concat_dim", data_types[2], shapes[2], datas[2]})     \
            .Output({"y", data_types[3], shapes[3], datas[3]})             \
            .Attr("N", 2);                                                 \
  } while (0)

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_CONCATV2_UT, TestConcatV2_##aicpu_type) {                        \
    vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32,           \
                                   aicpu_type};                                \
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {}, {4, 11}};          \
    base_type input[22];                                                       \
    SetRandomValue<base_type>(input, 22);                                      \
    base_type output[44] = {(base_type)0};                                     \
    int32_t concat_dim = 0;                                                    \
    vector<void *> datas = {(void *)input, (void *)input, (void *)&concat_dim, \
                            (void *)output};                                   \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();           \
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");            \
    CREATE_NODEDEF(shapes, data_types, datas, builder);                        \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[44] = {(base_type)0};                                 \
    CalcExpectFunc(*node_def.get(), expect_out);                               \
    CompareResult<base_type>(output, expect_out, 44);                          \
  }

#define ADD_CASE_WITH_SHAPE(case_name, base_type, aicpu_type, shapes,        \
                            data_num)                                         \
  TEST_F(TEST_CONCATV2_UT, TestConcatV2_##aicpu_type##_##case_name) {        \
    vector<DataType> data_types = {aicpu_type, aicpu_type, DT_INT32,         \
                                   aicpu_type};                              \
    base_type input[data_num];                                               \
    SetRandomValue<base_type>(input, data_num);                              \
    base_type output[data_num * 2] = {(base_type)0};                         \
    int32_t concat_dim = 0;                                                  \
    vector<void *> datas = {(void *)input, (void *)input,                    \
                            (void *)&concat_dim, (void *)output};             \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
    NodeDefBuilder builder(node_def.get(), "ConcatV2", "ConcatV2");          \
    CREATE_NODEDEF(shapes, data_types, datas, builder);                       \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                             \
    base_type expect_out[data_num * 2] = {(base_type)0};                      \
    CalcExpectFunc(*node_def.get(), expect_out);                              \
    CompareResult<base_type>(output, expect_out, data_num * 2);               \
  }


ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

vector <vector<int64_t>> shapes = {{2, 4},
                                   {2, 4},
                                   {},
                                   {4, 4}};
ADD_CASE_WITH_SHAPE(2_4__2_4__0__4_4, float, DT_FLOAT, shapes, 8)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(uint16_t, DT_UINT16)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(uint64_t, DT_UINT64)

ADD_CASE(bool, DT_BOOL)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(std::complex < float >, DT_COMPLEX64)

ADD_CASE(std::complex < double >, DT_COMPLEX128)

ADD_CASE(Eigen::bfloat16, DT_BFLOAT16)