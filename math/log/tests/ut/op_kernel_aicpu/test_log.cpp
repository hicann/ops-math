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
#include "utils/aicpu_read_file.h"
#undef private
#undef protected
#include "utils/eigen_tensor.h"

using namespace std;
using namespace aicpu;

const std::string ktestcaseFilePath =
    "../../../../math/log/tests/ut/op_kernel_aicpu/";

static void GenData() {
  system(("cp -r " + ktestcaseFilePath + "script ./").c_str());
  system("chmod -R 755 ./script");
  system("cd ./script && python3 log_gen_data.py");
  char * path_ = get_current_dir_name();
  string path(path_);
  system(("mkdir -p " + ktestcaseFilePath + "data").c_str());
  system(("cp -r " + path + "/log/data/* " + ktestcaseFilePath + "data").c_str());
}

class TEST_LOG_UT : public testing::Test {
  protected:
    static void SetUpTestCase() {
      cout << "log_test SetUp\n" << endl;
      cout << "begin to gen test data\n" << endl;
      GenData();
    }
    static void TearDownTestCase() {
      cout << "log_test TearDown\n" << endl;
    }
};

#define CREATE_NODEDEF(shapes, data_types, datas, base, shift, scale)  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  NodeDefBuilder(node_def.get(), "Log", "Log")                         \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("base", base)                                              \
      .Attr("shift", shift)                                            \
      .Attr("scale", scale)


template <typename Tin, typename Tout>
void RunTestLog(char *dtype, vector<DataType> data_types, vector<vector<int64_t>>& shapes) {
  uint64_t data_size = CalTotalElements(shapes, 0);
  Tin input_data[data_size];
  Tout output_exp_data[data_size];
  Tout output_data[data_size];
  std::string input_data_path{ktestcaseFilePath + "data/log_data_input_" +
                              dtype + ".txt"};
  std::string output_exp_data_path{ktestcaseFilePath +
                               "data/log_data_output_" + dtype + ".txt"};

  EXPECT_EQ(ReadFile(input_data_path, input_data, data_size), true);
  EXPECT_EQ(ReadFile(output_exp_data_path, output_exp_data, data_size), true);

  vector<void*> datas = {(void*)input_data, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, output_exp_data, data_size);
  EXPECT_EQ(compare, true);
}


TEST_F(TEST_LOG_UT, DT_UINT8_SUCCESS) {
  vector<DataType> data_types = {DT_UINT8, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<uint8_t, float>((char *)"uint8", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_INT8_SUCCESS) {
  vector<DataType> data_types = {DT_INT8, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<int8_t, float>((char *)"int8", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_INT16_SUCCESS) {
  vector<DataType> data_types = {DT_INT16, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<int16_t, float>((char *)"int16", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_INT32_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<int32_t, float>((char *)"int32", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_INT64_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<int64_t, float>((char *)"int64", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_FLOAT_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<float, float>((char *)"float32", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_DOUBLE_SUCCESS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes{{5, 4, 3}, {5, 4, 3}};
  RunTestLog<double, double>((char *)"float64", data_types, shapes);
}

TEST_F(TEST_LOG_UT, DT_BOOL_SUCCESS) {
  bool input_data[4] = {true, false, true, false};
  float output_data[4];
  float expect_output[4] = {0.0, -std::numeric_limits<float>::infinity(),
                            0.0, -std::numeric_limits<float>::infinity()};
  vector<DataType> data_types = {DT_BOOL, DT_FLOAT};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_FLOAT16_SUCCESS) {
  Eigen::half input_data[4] = {(Eigen::half)1.0, (Eigen::half)100, (Eigen::half)3.0, (Eigen::half)0.0};
  Eigen::half output_data[4];
  Eigen::half expect_output[4] = {(Eigen::half)0.0, (Eigen::half)4.605, (Eigen::half)1.099,
                                  -std::numeric_limits<Eigen::half>::infinity()};
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_BFLOAT16_SUCCESS) {
  Eigen::bfloat16 input_data[4] = {(Eigen::bfloat16)1.0, (Eigen::bfloat16)100,
                               (Eigen::bfloat16)3.0, (Eigen::bfloat16)0.0};
  Eigen::bfloat16 output_data[4];
  Eigen::bfloat16 expect_output[4] = {(Eigen::bfloat16)0.0, (Eigen::bfloat16)4.605, (Eigen::bfloat16)1.099,
                                  -std::numeric_limits<Eigen::bfloat16>::infinity()};
  vector<DataType> data_types = {DT_BFLOAT16, DT_BFLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_COMPLEX64_SUCCESS) {
  std::complex<float> input_data[4] = {{10.5, 7.8}, {10.5, -7.8}, {-10.5, 7.8}, {-10.5, -7.8}};
  std::complex<float> output_data[4];
  std::complex<float> expect_output[4] = {{2.5710948, 0.638914}, {2.5710948, -0.638914},
                                          {2.5710948, 2.5026786}, {2.5710948, -2.5026786}};
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_COMPLEX128_SUCCESS) {
  std::complex<double> input_data[4] = {{10.5, 7.8}, {10.5, -7.8}, {-10.5, 7.8}, {-10.5, -7.8}};
  std::complex<double> output_data[4];
  std::complex<double> expect_output[4] =  {{2.5710948, 0.638914}, {2.5710948, -0.638914},
                                            {2.5710948, 2.5026786}, {2.5710948, -2.5026786}};
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG2_SUCCESSED_01) {
  float input_data[4] = {32938632.0, 23806808.0, 57605544.0, 54950860.0};
  float output_data[4];
  float expect_output[4] =  {24.9732780456, 24.5048713684, 25.77970504760, 25.711639404296};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 2.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG2_SUCCESSED_02) {
  Eigen::half input_data[4] = {(Eigen::half)32938.0, (Eigen::half)23808.0,
                              (Eigen::half)5760.0, (Eigen::half)5495.0};
  Eigen::half output_data[4];
  Eigen::half expect_output[4] =  {(Eigen::half)15.0078125, (Eigen::half)14.5390625,
                                  (Eigen::half)12.4921875, (Eigen::half)12.421875};
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 2.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG2_SUCCESSED_03) {
  std::complex<float> input_data[5] = {{10.5, 7.8}, {10.5, -7.8}, {-10.5, 7.8}, {-10.5, -7.8},
                                       {6765867.706626937772464990616, 0}};
  std::complex<float> output_data[5];
  std::complex<float> expect_output[5] = {{3.70930554826, 0.92175805568}, {3.70930554826, -0.92175805568},
                                          {3.70930554826, 3.6106021404266}, {3.70930554826, -3.6106021404266},
                                          {22.68984353763857697572, 0}};
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes{{5}, {5}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 2.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG2_SUCCESSED_04) {
  int32_t input_data[4] = {32938, 23806, 57605, 1234};
  float output_data[4];
  float expect_output[4] =  {15.00746536254882812, 14.53903770446777344,
                             15.81390666961669922, 10.26912689208984375};
  vector<DataType> data_types = {DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 2.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG10_SUCCESSED_01) {
  float input_data[4] = {96120504.0, 41587040.0, 62258544.0, 22114832.0};
  float output_data[4];
  float expect_output[4] =  {7.9828162193, 7.6189579963, 7.79419898986, 7.3446836471};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 10.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG10_SUCCESSED_02) {
  Eigen::half input_data[4] = {(Eigen::half)32938.0, (Eigen::half)23806.0,
                              (Eigen::half)57605.0, (Eigen::half)57605.0};
  Eigen::half output_data[4];
  Eigen::half expect_output[4] =  {(Eigen::half)4.515625, (Eigen::half)4.37500,
                                  (Eigen::half)4.761718750, (Eigen::half)4.761718750};
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 10.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG10_SUCCESSED_03) {
  std::complex<float> input_data[4] = {{10.5, 7.8}, {10.5, -7.8}, {-10.5, 7.8}, {-10.5, -7.8}};
  std::complex<float> output_data[4];
  std::complex<float> expect_output[4] = {{1.1166121959686, 0.2774768173}, {1.1166121959686, -0.2774768173},
                                          {1.1166121959686, 1.0868995189}, {1.1166121959686, -1.0868995189}};
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 10.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_LOG10_SUCCESSED_04) {
  int32_t input_data[4] = {32938, 23806, 57605, 1234};
  float output_data[4];
  float expect_output[4] =  {4.51769733428955078, 4.37668657302856445,
                             4.76046037673950195, 3.09131526947021484};
  vector<DataType> data_types = {DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 10.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_BASE_01) {
  float input_data[5] = {1.0, 8.0, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<Eigen::half>::infinity(), std::nanf("")};
  float output_data[5];
  float expect_output[5] = {0.0, 3.0, std::numeric_limits<float>::infinity(),
                              std::nanf(""),   std::nanf("")};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5}, {5}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 2.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_BASE_02) {
  Eigen::half input_data[4] = {(Eigen::half)1.0, (Eigen::half)100, (Eigen::half)3.0, (Eigen::half)0.0};
  Eigen::half output_data[4];
  Eigen::half expect_output[4] = {(Eigen::half)0.0, (Eigen::half)4.605, (Eigen::half)1.099,
                                  -std::numeric_limits<Eigen::half>::infinity()};
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Log", "Log")
      .Input({"x", data_types[0], shapes[0], datas[0]})
      .Output({"y", data_types[1], shapes[1], datas[1]})
      .Attr("shift", 0.0)
      .Attr("scale", 1.0);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_SHIFT_01) {
  float input_data[5] = {1.0, 2.0, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<Eigen::half>::infinity(), std::nanf("")};
  float output_data[5];
  float expect_output[5] = {0.69314718, 1.09861231, std::numeric_limits<float>::infinity(),
                              std::nanf(""),   std::nanf("")};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5}, {5}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 1.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_SHIFT_02) {
  Eigen::half input_data[4] = {(Eigen::half)1.0, (Eigen::half)100, (Eigen::half)3.0, (Eigen::half)0.0};
  Eigen::half output_data[4];
  Eigen::half expect_output[4] = {(Eigen::half)0.0, (Eigen::half)4.605, (Eigen::half)1.099,
                                  -std::numeric_limits<Eigen::half>::infinity()};
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Log", "Log")
      .Input({"x", data_types[0], shapes[0], datas[0]})
      .Output({"y", data_types[1], shapes[1], datas[1]})
      .Attr("base", -1.0)
      .Attr("scale", 1.0);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_SHIFT_03) {
  int32_t input_data[4] = {1, 2, 3, 4};
  float output_data[4];
  float expect_output[4] = {0.69314718246459961, 1.09861230850219727,
                            1.38629436492919922, 1.60943794250488281};
  vector<DataType> data_types = {DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 1.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_SCALE_01) {
  float input_data[5] = {0.5, 1.5, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<Eigen::half>::infinity(), std::nanf("")};
  float output_data[5];
  float expect_output[5] = {0.0, 1.09861231, std::numeric_limits<float>::infinity(),
                              std::nanf(""),   std::nanf("")};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5}, {5}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 2.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, DT_SCALE_02) {
  Eigen::half input_data[4] = {(Eigen::half)1.0, (Eigen::half)100, (Eigen::half)3.0, (Eigen::half)0.0};
  Eigen::half output_data[4];
  Eigen::half expect_output[4] = {(Eigen::half)0.0, (Eigen::half)4.605, (Eigen::half)1.099,
                                  -std::numeric_limits<Eigen::half>::infinity()};
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4}, {4}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Log", "Log")
      .Input({"x", data_types[0], shapes[0], datas[0]})
      .Output({"y", data_types[1], shapes[1], datas[1]})
      .Attr("base", -1.0)
      .Attr("shift", 0.0);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, NOT_DEFAULT) {
  float input_data[5] = {1.0, 2.0, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<Eigen::half>::infinity(), std::nanf("")};
  float output_data[5];
  float expect_output[5] = {0.82708752, 1.06862164, std::numeric_limits<float>::infinity(),
                              std::nanf(""),   std::nanf("")};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{5}, {5}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, 7.0, 2.0, 3.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool compare = CompareResult(output_data, expect_output, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LOG_UT, EMPTY_TENSOR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{0}, {0}};
  vector<void*> datas = {(void*)nullptr, (void*)nullptr};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_LOG_UT, NOT_SUPPORT_TYPE_FAILED_01) {
  std::string input_data[1] = {"1"};
  std::string  output_data[1];
  vector<DataType> data_types = {DT_STRING, DT_STRING};
  vector<vector<int64_t>> shapes{{1}, {1}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOG_UT, NOT_SUPPORT_TYPE_FAILED_02) {
  uint16_t input_data[1] = {1};
  double output_data[1];
  vector<DataType> data_types = {DT_UINT16, DT_DOUBLE};
  vector<vector<int64_t>> shapes{{1}, {1}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOG_UT, NOT_SUPPORT_TYPE_FAILED_03) {
  int32_t input_data[1] = {1};
  int32_t output_data[1];
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes{{1}, {1}};
  vector<void*> datas = {(void*)input_data, (void*)output_data};

  CREATE_NODEDEF(shapes, data_types, datas, -1.0, 0.0, 1.0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}