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
#include "utils/aicpu_read_file.h"
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_INVERT_PERMUTATION_UT : public testing::Test {};

const std::string ktestcaseFilePath =
    "../../../../conversion/invert_permutation/tests/ut/op_kernel_aicpu/";

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "InvertPermutation", "InvertPermutation") \
      .Input({"x", data_types[0], shapes[0], datas[0]})                    \
      .Output({"y", data_types[1], shapes[1], datas[1]})

template <typename T1, typename T2>
void RunInvertPermutationKernel(vector<string> data_files,
                                vector<DataType> data_types,
                                vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 input[input_size];
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 output[output_size];
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T2 output_exp[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1024}, {1024}};
  vector<string> files{"invert_permutation/data/input_int32.txt",
                       "invert_permutation/data/output_int32.txt"};
  RunInvertPermutationKernel<int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, DATA_TYPE_INT32_PARA_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{131072}, {131072}};
  vector<string> files{"invert_permutation/data/input_int32_para.txt",
                       "invert_permutation/data/output_int32_para.txt"};
  RunInvertPermutationKernel<int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1024}, {1024}};
  vector<string> files{"invert_permutation/data/input_int64.txt",
                       "invert_permutation/data/output_int64.txt"};
  RunInvertPermutationKernel<int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, DATA_TYPE_INT64_PARA_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{131072}, {131072}};
  vector<string> files{"invert_permutation/data/input_int64_para.txt",
                       "invert_permutation/data/output_int64_para.txt"};
  RunInvertPermutationKernel<int64_t, int64_t>(files, data_types, shapes);
}

// exception test instance
TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT16, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {4}};
  int32_t input[4] = {(int16_t)1};
  int32_t output[4] = {(int32_t)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_DIM_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4}};
  int32_t input[8] = {(int32_t)1};
  int32_t output[4] = {(int32_t)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_VALUE_RANGE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {4}};
  int32_t input[4];
  input[0] = 4;
  input[1] = 0;
  input[2] = 2;
  input[3] = 3;
  int32_t output[4] = {(int32_t)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVERT_PERMUTATION_UT, INPUT_ELEMENT_DUPLICATED_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {4}};
  int32_t input[4];
  input[0] = 1;
  input[1] = 0;
  input[2] = 1;
  input[3] = 3;
  int32_t output[4] = {(int32_t)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}