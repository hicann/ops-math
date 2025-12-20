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
#include "utils/aicpu_read_file.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include <cmath>
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_ROUND_UT : public testing::Test {};

const std::string ktestcaseFilePath = 
    "../../../../math/round/tests/ut/op_kernel_aicpu/";

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  int64_t input0_num = input0->NumElements();
  for (int64_t i = 0; i < input0_num; ++i) {
    expect_out[i] = std::round(input0_data[i]);
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas, decimals)        \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Round", "Round")                 \
      .Input({"input", data_types[0], shapes[0], datas[0]})        \
      .Output({"output", data_types[1], shapes[1], datas[1]})      \
      .Attr("decimals", decimals);

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_ROUND_UT, TestRound_##aicpu_type) {                              \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                    \
    vector<vector<int64_t>> shapes = {{4}, {4}};                               \
    base_type input[4];                                                        \
    SetRandomValue<base_type>(input, 4);                                       \
    base_type output[4] = {(base_type)0};                                      \
    vector<void *> datas = {(void *)input, (void *)output};                    \
    int64_t decimals = 0;                                                      \
    CREATE_NODEDEF(shapes, data_types, datas, decimals);                       \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[4] = {(base_type)0};                                  \
    CalcExpectFunc(*node_def.get(), expect_out);                               \
    CompareResult<base_type>(output, expect_out, 4);                          \
  }

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

TEST_F(TEST_ROUND_UT, Input_DT_UINT64_Error) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{4}, {4}};
  uint64_t input[4];
  SetRandomValue<uint64_t>(input, 4);
  uint64_t output[4] = {(uint64_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  int64_t decimals = 0;
  CREATE_NODEDEF(shapes, data_types, datas, decimals);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// read input and output data from files which generate by your python file

template<typename T1, typename T2>
void RunRoundKernel(vector<string> data_files,
               vector<DataType> data_types,
               vector<vector<int64_t>> &shapes,
               int64_t decimals) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, decimals);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT32_WITH_DECIMALS_1) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
  vector<string> files{"round/data/round_data_input1_float32_1.txt",
                       "round/data/round_data_output1_float32_1.txt"};
  RunRoundKernel<float, float>(files, data_types, shapes, 1);
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT32_WITH_DECIMALS_2) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
  vector<string> files{"round/data/round_data_input1_float32_2.txt",
                       "round/data/round_data_output1_float32_2.txt"};
  RunRoundKernel<float, float>(files, data_types, shapes, 2);
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT32_WITH_DECIMALS_3) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
  vector<string> files{"round/data/round_data_input1_float32_-1.txt",
                       "round/data/round_data_output1_float32_-1.txt"};
  RunRoundKernel<float, float>(files, data_types, shapes, -1);
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT32_WITH_DECIMALS_4) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
  vector<string> files{"round/data/round_data_input1_float32_-2.txt",
                       "round/data/round_data_output1_float32_-2.txt"};
  RunRoundKernel<float, float>(files, data_types, shapes, -2);
}

TEST_F(TEST_ROUND_UT, Input_DT_FLOAT64_WITH_DECIMALS_1) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}};
  vector<string> files{"round/data/round_data_input1_float64_1.txt",
                       "round/data/round_data_output1_float64_1.txt"};
  RunRoundKernel<double, double>(files, data_types, shapes, 1);
}

TEST_F(TEST_ROUND_UT, DATA_IS_POINT_FIVE) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  float_t input[1] = {-79.2500};
  float_t output1[1] = {0.0f};
  vector<void *> datas = {(void *)input, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}


