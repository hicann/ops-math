/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/status.h"
#include "utils/aicpu_read_file.h"
#include <string>
#include <vector>

using namespace aicpu;
using namespace std;

const std::string ktestcaseFilePath =
    "../../../../math/reduce_mean/tests/ut/op_kernel_aicpu/";
	
static void GenData() {
  system(("cp -r " + ktestcaseFilePath + "script ./").c_str());
  system("chmod -R 755 ./script");
  system("cd ./script && python3 reduce_mean_gen_data.py");
  char * path_ = get_current_dir_name();
  string path(path_);
  system(("mkdir -p " + ktestcaseFilePath + "data").c_str());
  system(("cp -r " + path + "/reduce_mean/data/* " + ktestcaseFilePath + "data").c_str());
}

struct ReduceMeanKernelTest : public testing::Test {
	  protected:
    static void SetUpTestCase() {
      cout << "reduce_mean_test SetUp\n" << endl;
      cout << "begin to gen test data\n" << endl;
      GenData();
    }
    static void TearDownTestCase() {
      cout << "reduce_mean_test TearDown\n" << endl;
    }
};


template <typename T1, typename T2>
void RunReduceMeanKernel(std::vector<std::string> data_files, DataType data_type1, DataType data_type2,
                         std::vector<std::vector<int64_t>>& shapes) {
  std::string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1* input1_tensor_buffer = new T1[input1_size];
  bool status = ReadFile(data_path, input1_tensor_buffer, input1_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2* input2_tensor_buffer = new T2[input2_size];
  status = ReadFile(data_path, input2_tensor_buffer, input2_size);
  EXPECT_EQ(status, true);
  uint64_t output_size = CalTotalElements(shapes, 2);
  T1* output_tensor_buffer = new T1[output_size];
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, shapes[0], (void*)input1_tensor_buffer})
      .Input({"axes", data_type2, shapes[1], (void*)input2_tensor_buffer})
      .Output({"y", data_type1, shapes[2], (void*)output_tensor_buffer})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T1* output_exp = new T1[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  bool compare = CompareResult(output_tensor_buffer, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1_tensor_buffer;
  delete[] input2_tensor_buffer;
  delete[] output_tensor_buffer;
  delete[] output_exp;
}

TEST_F(ReduceMeanKernelTest, INT32_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{3, 4, 2}, {3}, {1}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_int32_input1.txt",
                                      "reduce_mean/data/reduce_mean_int32_input2.txt",
                                      "reduce_mean/data/reduce_mean_int32_output.txt"};
  RunReduceMeanKernel<int32_t, int32_t>(data_files, DT_INT32, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, INT32_ALL_DIM1_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{1, 1, 1}, {1}, {1}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_int32_2_input1.txt",
                                      "reduce_mean/data/reduce_mean_int32_2_input2.txt",
                                      "reduce_mean/data/reduce_mean_int32_2_output.txt"};
  RunReduceMeanKernel<int32_t, int32_t>(data_files, DT_INT32, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, INT32_DIM1_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{3, 3, 1, 4}, {1}, {3, 4}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_int32_3_input1.txt",
                                      "reduce_mean/data/reduce_mean_int32_3_input2.txt",
                                      "reduce_mean/data/reduce_mean_int32_3_output.txt"};
  RunReduceMeanKernel<int32_t, int32_t>(data_files, DT_INT32, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, INT64_INT64_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 6}, {1}, {2, 6}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_int64_input1.txt",
                                      "reduce_mean/data/reduce_mean_int64_input2.txt",
                                      "reduce_mean/data/reduce_mean_int64_output.txt"};
  RunReduceMeanKernel<int64_t, int64_t>(data_files, DT_INT64, DT_INT64, shapes);
}

TEST_F(ReduceMeanKernelTest, FLOAT_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{1, 3, 6}, {1}, {3, 6}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_float_input1.txt",
                                      "reduce_mean/data/reduce_mean_float_input2.txt",
                                      "reduce_mean/data/reduce_mean_float_output.txt"};
  RunReduceMeanKernel<float, int32_t>(data_files, DT_FLOAT, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, FLOAT2_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{4, 3, 6}, {1}, {3, 6}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_float2_input1.txt",
                                      "reduce_mean/data/reduce_mean_float2_input2.txt",
                                      "reduce_mean/data/reduce_mean_float2_output.txt"};
  RunReduceMeanKernel<float, int32_t>(data_files, DT_FLOAT, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, FLOAT3_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{4, 3, 5, 6}, {1}, {4, 3, 6}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_float3_input1.txt",
                                      "reduce_mean/data/reduce_mean_float3_input2.txt",
                                      "reduce_mean/data/reduce_mean_float3_output.txt"};
  RunReduceMeanKernel<float, int32_t>(data_files, DT_FLOAT, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, FLOAT16_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3}, {2}, {3, 3}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_float16_input1.txt",
                                      "reduce_mean/data/reduce_mean_float16_input2.txt",
                                      "reduce_mean/data/reduce_mean_float16_output.txt"};
  RunReduceMeanKernel<Eigen::half, int32_t>(data_files, DT_FLOAT16, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, DOUBLE_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2, 3, 2, 3}, {4}, {2, 2, 2, 2}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_double_input1.txt",
                                      "reduce_mean/data/reduce_mean_double_input2.txt",
                                      "reduce_mean/data/reduce_mean_double_output.txt"};
  RunReduceMeanKernel<double, int32_t>(data_files, DT_DOUBLE, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, DOUBLE2_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2, 3, 2, 3}, {4}, {3, 3, 3, 3}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_double2_input1.txt",
                                      "reduce_mean/data/reduce_mean_double2_input2.txt",
                                      "reduce_mean/data/reduce_mean_double2_output.txt"};
  RunReduceMeanKernel<double, int32_t>(data_files, DT_DOUBLE, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, INT8_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2, 3, 2}, {4}, {3, 3, 3}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_int8_input1.txt",
                                      "reduce_mean/data/reduce_mean_int8_input2.txt",
                                      "reduce_mean/data/reduce_mean_int8_output.txt"};
  RunReduceMeanKernel<int8_t, int32_t>(data_files, DT_INT8, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, UINT8_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2, 3, 2}, {3}, {2, 2, 2, 2}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_uint8_input1.txt",
                                      "reduce_mean/data/reduce_mean_uint8_input2.txt",
                                      "reduce_mean/data/reduce_mean_uint8_output.txt"};
  RunReduceMeanKernel<uint8_t, int32_t>(data_files, DT_UINT8, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, INT16_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2, 3}, {3}, {3, 3, 3}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_int16_input1.txt",
                                      "reduce_mean/data/reduce_mean_int16_input2.txt",
                                      "reduce_mean/data/reduce_mean_int16_output.txt"};
  RunReduceMeanKernel<int16_t, int32_t>(data_files, DT_INT16, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, UINT16_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2, 3}, {3}, {2, 2, 2}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_uint16_input1.txt",
                                      "reduce_mean/data/reduce_mean_uint16_input2.txt",
                                      "reduce_mean/data/reduce_mean_uint16_output.txt"};
  RunReduceMeanKernel<uint16_t, int32_t>(data_files, DT_UINT16, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, UINT32_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2}, {3}, {3, 3}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_uint32_input1.txt",
                                      "reduce_mean/data/reduce_mean_uint32_input2.txt",
                                      "reduce_mean/data/reduce_mean_uint32_output.txt"};
  RunReduceMeanKernel<uint32_t, int32_t>(data_files, DT_UINT32, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, UINT64_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{2, 3, 2, 3, 2}, {2}, {2, 2, 2}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_uint64_input1.txt",
                                      "reduce_mean/data/reduce_mean_uint64_input2.txt",
                                      "reduce_mean/data/reduce_mean_uint64_output.txt"};
  RunReduceMeanKernel<uint64_t, int32_t>(data_files, DT_UINT64, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, COMPLEX64_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{4, 5, 3, 7}, {1}, {5, 3, 7}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_complex64_input1.txt",
                                      "reduce_mean/data/reduce_mean_complex64_input2.txt",
                                      "reduce_mean/data/reduce_mean_complex64_output.txt"};
  RunReduceMeanKernel<std::complex<float>, int32_t>(data_files, DT_COMPLEX64, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, COMPLEX128_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{4, 5, 1, 3}, {1}, {4, 1, 3}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_complex128_input1.txt",
                                      "reduce_mean/data/reduce_mean_complex128_input2.txt",
                                      "reduce_mean/data/reduce_mean_complex128_output.txt"};
  RunReduceMeanKernel<std::complex<double>, int32_t>(data_files, DT_COMPLEX128, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, COMPLEX64_2_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{4, 5, 3, 6}, {2}, {5, 6}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_complex64_2_input1.txt",
                                      "reduce_mean/data/reduce_mean_complex64_2_input2.txt",
                                      "reduce_mean/data/reduce_mean_complex64_2_output.txt"};
  RunReduceMeanKernel<std::complex<float>, int32_t>(data_files, DT_COMPLEX64, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, COMPLEX128_2_INT32_SUCC) {
  std::vector<std::vector<int64_t>> shapes = {{4, 5, 2, 1}, {2}, {4, 2}};
  std::vector<std::string> data_files{"reduce_mean/data/reduce_mean_complex128_2_input1.txt",
                                      "reduce_mean/data/reduce_mean_complex128_2_input2.txt",
                                      "reduce_mean/data/reduce_mean_complex128_2_output.txt"};
  RunReduceMeanKernel<std::complex<double>, int32_t>(data_files, DT_COMPLEX128, DT_INT32, shapes);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_INT8) {
  DataType data_type1 = DT_INT8;
  vector<int64_t> input1_shape{0};
  int8_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  int8_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_INT16) {
  DataType data_type1 = DT_INT16;
  vector<int64_t> input1_shape{0};
  int16_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  int16_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_INT32) {
  DataType data_type1 = DT_INT32;
  vector<int64_t> input1_shape{0};
  int32_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  int32_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_INT64) {
  DataType data_type1 = DT_INT64;
  vector<int64_t> input1_shape{0};
  int64_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  int64_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_UINT8) {
  DataType data_type1 = DT_UINT8;
  vector<int64_t> input1_shape{0};
  uint8_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  uint8_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_UINT16) {
  DataType data_type1 = DT_UINT16;
  vector<int64_t> input1_shape{0};
  uint16_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  uint16_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_UINT32) {
  DataType data_type1 = DT_UINT32;
  vector<int64_t> input1_shape{0};
  uint32_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  uint32_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_UINT64) {
  DataType data_type1 = DT_UINT64;
  vector<int64_t> input1_shape{0};
  uint64_t input1[] = {0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  uint64_t output_tensor[] = {0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_FLOAT16) {
  DataType data_type1 = DT_FLOAT16;
  vector<int64_t> input1_shape{0};
  Eigen::half input1[] = {(Eigen::half)0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  Eigen::half output_tensor[] = {(Eigen::half)0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_FLOAT) {
  DataType data_type1 = DT_FLOAT;
  vector<int64_t> input1_shape{0};
  float input1[] = {0.0f};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  float output_tensor[] = {0.0f};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_DOUBLE) {
  DataType data_type1 = DT_DOUBLE;
  vector<int64_t> input1_shape{0};
  double input1[] = {0.0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  double output_tensor[] = {0.0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_COMPLEX64) {
  DataType data_type1 = DT_COMPLEX64;
  vector<int64_t> input1_shape{0};
  std::complex<float> input1[] = {0.0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  std::complex<float> output_tensor[] = {0.0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, EMPTY_TENSOR_COMPLEX128) {
  DataType data_type1 = DT_COMPLEX128;
  vector<int64_t> input1_shape{0};
  std::complex<double> input1[] = {0.0};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  std::complex<double> output_tensor[] = {0.0};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(ReduceMeanKernelTest, DTYPE_FAILED) {
  DataType data_type1 = DT_BOOL;
  vector<int64_t> input1_shape{2};
  bool input1[] = {true, true};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {0};
  vector<int64_t> output_shape{1};
  bool output_tensor[1];
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(ReduceMeanKernelTest, AXES_FAILED) {
  DataType data_type1 = DT_FLOAT;
  vector<int64_t> input1_shape{2};
  float input1[] = {1.0f, 2.0f};
  DataType data_type2 = DT_INT32;
  vector<int64_t> input2_shape{1};
  int32_t input2[] = {1};
  vector<int64_t> output_shape{1};
  float output_tensor[] = {0.0f};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(ReduceMeanKernelTest, AXES_DTYPE_FAILED) {
  DataType data_type1 = DT_FLOAT;
  vector<int64_t> input1_shape{2};
  float input1[] = {1.0f, 2.0f};
  DataType data_type2 = DT_FLOAT;
  vector<int64_t> input2_shape{1};
  float input2[] = {0.0f};
  vector<int64_t> output_shape{1};
  float output_tensor[] = {0.0f};
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "ReduceMean", "ReduceMean")
      .Input({"x", data_type1, input1_shape, (void*)input1})
      .Input({"axes", data_type2, input2_shape, (void*)input2})
      .Output({"y", data_type1, output_shape, (void*)output_tensor})
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}