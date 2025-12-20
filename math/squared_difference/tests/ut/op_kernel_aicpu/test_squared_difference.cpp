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
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

const std::string squaredDifferenceTestcaseFilePath = "../../../../math/squared_difference/tests/ut/op_kernel_aicpu/";

class TEST_SQUAREDDIFFERENCE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "SquaredDifference", "SquaredDifference")     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})                       \
      .Input({"x2", data_types[1], shapes[1], datas[1]})                       \
      .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 1024}, {1,1024}, {4, 1024}};

  // read data from file for input1
  string data_path =
      squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_input1_7.txt";
  constexpr uint64_t input1_size = 4 * 1024;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_input2_7.txt";
  constexpr uint64_t input2_size = 1024;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 4 * 1024;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_output1_7.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_X_NUM_ONE_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1}, {1,3}, {1,3}};

  // read data from file for input1
  string data_path =
      squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_input1_8.txt";
  constexpr uint64_t input1_size = 1;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_input2_8.txt";
  constexpr uint64_t input2_size = 3;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 3;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_output1_8.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_Y_NUM_ONESUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1,3}, {1}, {1,3}};

  // read data from file for input1
  string data_path =
      squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_input1_9.txt";
  constexpr uint64_t input1_size = 3;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_input2_9.txt";
  constexpr uint64_t input2_size = 1;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 3;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = squaredDifferenceTestcaseFilePath + "squared_difference/data/squareddifference_data_output1_9.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}