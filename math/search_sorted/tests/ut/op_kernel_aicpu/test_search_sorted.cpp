/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_SearchSorted_UTest : public testing::Test {};

#define ADD_CASE(aicpu_type, base_type, out_type, out_dtype)                        \
    TEST_F(TEST_SearchSorted_UTest, SearchSorted_Success_##aicpu_type##_##out_type) \
    {                                                                               \
        aicpu_type sorted_sequence[2 * 5] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};        \
        aicpu_type values[2 * 3] = {3, 6, 9, 3, 6, 9};                              \
        out_type output[2 * 3] = {0};                                               \
        out_type expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};                       \
        auto nodeDef = CpuKernelUtils::CreateNodeDef();                             \
        nodeDef->SetOpType("SearchSorted");                                         \
        auto right = CpuKernelUtils::CreateAttrValue();                             \
        right->SetBool(false);                                                      \
        nodeDef->AddAttrs("right", right.get());                                    \
        auto side = CpuKernelUtils::CreateAttrValue();                              \
        side->SetString("");                                                        \
        nodeDef->AddAttrs("side", side.get());                                      \
        auto inputTensor0 = nodeDef->AddInputs();                                   \
        EXPECT_NE(inputTensor0, nullptr);                                           \
        auto aicpuShape0 = inputTensor0->GetTensorShape();                          \
        std::vector<int64_t> shapes0 = {2, 5};                                      \
        aicpuShape0->SetDimSizes(shapes0);                                          \
        inputTensor0->SetDataType(base_type);                                       \
        inputTensor0->SetData(sorted_sequence);                                     \
        inputTensor0->SetDataSize(2 * 5 * sizeof(aicpu_type));                      \
        auto inputTensor1 = nodeDef->AddInputs();                                   \
        EXPECT_NE(inputTensor1, nullptr);                                           \
        auto aicpuShape1 = inputTensor1->GetTensorShape();                          \
        std::vector<int64_t> shapes1 = {2, 3};                                      \
        aicpuShape1->SetDimSizes(shapes1);                                          \
        inputTensor1->SetDataType(base_type);                                       \
        inputTensor1->SetData(values);                                              \
        inputTensor1->SetDataSize(2 * 3 * sizeof(aicpu_type));                      \
        auto outputTensor1 = nodeDef->AddOutputs();                                 \
        EXPECT_NE(outputTensor1, nullptr);                                          \
        auto aicpuShape3 = outputTensor1->GetTensorShape();                         \
        std::vector<int64_t> shapes2 = {2, 3};                                      \
        aicpuShape3->SetDimSizes(shapes2);                                          \
        outputTensor1->SetDataType(out_dtype);                                      \
        outputTensor1->SetData(output);                                             \
        outputTensor1->SetDataSize(2 * 3 * sizeof(out_type));                       \
        CpuKernelContext ctx(DEVICE);                                               \
        EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);                       \
        uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);             \
        EXPECT_EQ(ret, KERNEL_STATUS_OK);                                           \
        EXPECT_EQ(0, std::memcmp(output, expected_output, sizeof(output)));         \
    }

TEST_F(TEST_SearchSorted_UTest, SearchSorted_Input_Type_Error)
{
    uint32_t sorted_sequence[2 * 5] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    uint32_t values[2 * 3] = {3, 6, 9, 3, 6, 9};
    int32_t output[2 * 3] = {0};
    int32_t expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SearchSorted");
    auto right = CpuKernelUtils::CreateAttrValue();
    right->SetBool(false);
    nodeDef->AddAttrs("right", right.get());
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {2, 5};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_UINT32);
    inputTensor0->SetData(sorted_sequence);
    inputTensor0->SetDataSize(2 * 5 * sizeof(uint32_t));
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {2, 3};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_UINT32);
    inputTensor1->SetData(values);
    inputTensor1->SetDataSize(2 * 3 * sizeof(uint32_t));
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    auto aicpuShape3 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes2 = {2, 3};
    aicpuShape3->SetDimSizes(shapes2);
    outputTensor1->SetDataType(DT_INT32);
    outputTensor1->SetData(output);
    outputTensor1->SetDataSize(2 * 3 * sizeof(int32_t));
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SearchSorted_Input_Unsorted_Error)
{
    float sorted_sequence[2 * 5] = {9, 3, 5, 7, 9, 2, 4, 6, 8, 1};
    float values[2 * 3] = {3, 6, 9, 3, 6, 9};
    int32_t output[2 * 3] = {0};
    int32_t expected_output[2 * 3] = {0, 3, 4, 1, 2, 5};
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SearchSorted");
    auto right = CpuKernelUtils::CreateAttrValue();
    right->SetBool(false);
    nodeDef->AddAttrs("right", right.get());
    auto side = CpuKernelUtils::CreateAttrValue();
    side->SetString("");
    nodeDef->AddAttrs("side", side.get());
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {2, 5};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(sorted_sequence);
    inputTensor0->SetDataSize(2 * 5 * sizeof(float));
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {2, 3};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(values);
    inputTensor1->SetDataSize(2 * 3 * sizeof(float));
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    auto aicpuShape3 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes2 = {2, 3};
    aicpuShape3->SetDimSizes(shapes2);
    outputTensor1->SetDataType(DT_INT32);
    outputTensor1->SetData(output);
    outputTensor1->SetDataSize(2 * 3 * sizeof(int32_t));
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(output, expected_output, sizeof(output)));
}

ADD_CASE(float, DT_FLOAT, int, DT_INT32)

ADD_CASE(double, DT_DOUBLE, int, DT_INT32)

ADD_CASE(int32_t, DT_INT32, int, DT_INT32)

ADD_CASE(int64_t, DT_INT64, int, DT_INT32)

ADD_CASE(float, DT_FLOAT, int64_t, DT_INT64)

ADD_CASE(double, DT_DOUBLE, int64_t, DT_INT64)

ADD_CASE(int32_t, DT_INT32, int64_t, DT_INT64)

ADD_CASE(int64_t, DT_INT64, int64_t, DT_INT64)

template <typename T1, typename T2>
void RunSearchSortedKernel(
    vector<string> data_files, vector<DataType> data_type, vector<vector<int64_t>>& shapes, const bool right)
{
    string data_path = ktestcaseFilePath + data_files[0];
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1_tensor_buffer = new T1[input1_size];
    bool status = ReadFile(data_path, input1_tensor_buffer, input1_size);
    EXPECT_EQ(status, true);
    data_path = ktestcaseFilePath + data_files[1];
    uint64_t input2_size = CalTotalElements(shapes, 1);
    T1* input2_tensor_buffer = new T1[input2_size];
    status = ReadFile(data_path, input2_tensor_buffer, input2_size);
    EXPECT_EQ(status, true);
    uint64_t output_size = CalTotalElements(shapes, 2);
    T2* output_tensor_buffer = new T2[output_size];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", right);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // read data from file for expect ouput
    data_path = ktestcaseFilePath + data_files[2];
    T2* output_exp = new T2[output_size];
    status = ReadFile(data_path, output_exp, output_size);
    EXPECT_EQ(status, true);
    bool compare = CompareResult(output_tensor_buffer, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1_tensor_buffer;
    delete[] input2_tensor_buffer;
    delete[] output_tensor_buffer;
    delete[] output_exp;
}

template <typename T1, typename T2>
void RunSearchSortedWithSorterKernel(
    vector<string> data_files, vector<DataType> data_type, vector<vector<int64_t>>& shapes, const bool right)
{
    string data_path = ktestcaseFilePath + data_files[0];
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1_tensor_buffer = new T1[input1_size];
    bool status = ReadFile(data_path, input1_tensor_buffer, input1_size);
    EXPECT_EQ(status, true);
    data_path = ktestcaseFilePath + data_files[1];
    uint64_t input2_size = CalTotalElements(shapes, 1);
    T1* input2_tensor_buffer = new T1[input2_size];
    status = ReadFile(data_path, input2_tensor_buffer, input2_size);
    EXPECT_EQ(status, true);
    int64_t* input3_tensor_buffer = new int64_t[input1_size];
    data_path = ktestcaseFilePath + data_files[2];
    status = ReadFile(data_path, input3_tensor_buffer, input1_size);
    EXPECT_EQ(status, true);
    uint64_t output_size = CalTotalElements(shapes, 2);
    T2* output_tensor_buffer = new T2[output_size];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", right);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // read data from file for expect ouput
    data_path = ktestcaseFilePath + data_files[3];
    T2* output_exp = new T2[output_size];
    status = ReadFile(data_path, output_exp, output_size);
    EXPECT_EQ(status, true);
    bool compare = CompareResult(output_tensor_buffer, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1_tensor_buffer;
    delete[] input2_tensor_buffer;
    delete[] input3_tensor_buffer;
    delete[] output_tensor_buffer;
    delete[] output_exp;
}

TEST_F(TEST_SearchSorted_UTest, FLOAT_INT64_FALSE_SUCC)
{
    bool right = false;
    vector<DataType> data_type = {DT_FLOAT, DT_INT64};
    vector<vector<int64_t>> shapes = {{4, 3, 10}, {4, 3, 3}, {4, 3, 3}};
    vector<string> data_files{
        "search_sorted/data/search_sorted_float_input1.txt", "search_sorted/data/search_sorted_float_input2.txt",
        "search_sorted/data/search_sorted_float_output.txt"};
    RunSearchSortedKernel<float, int64_t>(data_files, data_type, shapes, right);
}

TEST_F(TEST_SearchSorted_UTest, DOUBLE_INT32_TRUE_SUCC)
{
    bool right = false;
    vector<DataType> data_type = {DT_DOUBLE, DT_INT32};
    vector<vector<int64_t>> shapes = {{100}, {200}, {200}};
    vector<string> data_files{
        "search_sorted/data/search_sorted_double_input1.txt", "search_sorted/data/search_sorted_double_input2.txt",
        "search_sorted/data/search_sorted_double_output.txt"};
    RunSearchSortedKernel<double, int32_t>(data_files, data_type, shapes, right);
}

TEST_F(TEST_SearchSorted_UTest, INT8_INT32_FALSE_SUCC)
{
    bool right = false;
    vector<DataType> data_type = {DT_INT8, DT_INT32};
    vector<vector<int64_t>> shapes = {{20, 10}, {20, 100}, {20, 100}};
    vector<string> data_files{
        "search_sorted/data/search_sorted_int8_input1.txt", "search_sorted/data/search_sorted_int8_input2.txt",
        "search_sorted/data/search_sorted_int8_input3.txt", "search_sorted/data/search_sorted_int8_output.txt"};
    RunSearchSortedWithSorterKernel<int8_t, int32_t>(data_files, data_type, shapes, right);
}

TEST_F(TEST_SearchSorted_UTest, INT32_INT64_TRUE_SUCC)
{
    bool right = true;
    vector<DataType> data_type = {DT_INT32, DT_INT64};
    vector<vector<int64_t>> shapes = {{4, 3, 10}, {4, 3, 3}, {4, 3, 3}};
    vector<string> data_files{
        "search_sorted/data/search_sorted_int32_input1.txt", "search_sorted/data/search_sorted_int32_input2.txt",
        "search_sorted/data/search_sorted_int32_output.txt"};
    RunSearchSortedKernel<int32_t, int64_t>(data_files, data_type, shapes, right);
}

TEST_F(TEST_SearchSorted_UTest, INT64_INT64_TRUE_SUCC)
{
    bool right = true;
    vector<DataType> data_type = {DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{4, 3, 10}, {4, 3, 3}, {4, 3, 3}};
    vector<string> data_files{
        "search_sorted/data/search_sorted_int64_input1.txt", "search_sorted/data/search_sorted_int64_input2.txt",
        "search_sorted/data/search_sorted_int64_input3.txt", "search_sorted/data/search_sorted_int64_output.txt"};
    RunSearchSortedWithSorterKernel<int64_t, int64_t>(data_files, data_type, shapes, right);
}

TEST_F(TEST_SearchSorted_UTest, SORTER_DTYPE_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {3}, {3}};
    int32_t input1_tensor_buffer[5] = {1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[3] = {3, 6, 9};
    int32_t input3_tensor_buffer[5] = {0, 1, 2, 4, 3};
    int32_t output_tensor_buffer[3];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT32, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SORTER_NUMBER_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {3}, {3}};
    int32_t input1_tensor_buffer[5] = {1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[3] = {3, 6, 9};
    int64_t input3_tensor_buffer[3] = {0, 1, 2};
    int32_t output_tensor_buffer[3];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[2], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SORTER_OUT_OF_RANGE_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 5}, {2, 3}, {2, 3}};
    int32_t input1_tensor_buffer[2 * 5] = {1, 3, 5, 9, 7, 1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[2 * 3] = {3, 6, 9, 3, 6, 9};
    int64_t input3_tensor_buffer[2 * 5] = {-1, 1, 2, 4, 3, 0, 1, 2, 4, 10};
    int32_t output_tensor_buffer[2 * 3];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SEQUENCE_VALUE_DIMS_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 5}, {6}, {6}};
    int32_t input1_tensor_buffer[2 * 5] = {1, 3, 5, 9, 7, 1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[6] = {3, 6, 9, 3, 6, 9};
    int64_t input3_tensor_buffer[2 * 5] = {0, 1, 2, 4, 3, 0, 1, 2, 4, 3};
    int32_t output_tensor_buffer[6];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, EMPTY_SEQUENCE_TENSOR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 0}, {2, 3}, {2, 3}};
    int32_t input1_tensor_buffer[] = {};
    int32_t input2_tensor_buffer[2 * 3] = {3, 6, 9, 3, 6, 9};
    int32_t output_tensor_buffer[2 * 3] = {0, 0, 0, 0, 0, 0};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[2 * 3] = {0, 0, 0, 0, 0, 0};
    bool compare = CompareResult(output_tensor_buffer, output_exp, 2 * 3);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SearchSorted_UTest, EMPTY_VALUE_TENSOR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {0}, {0}};
    int32_t input1_tensor_buffer[5] = {1, 3, 5, 7, 9};
    ;
    int32_t input2_tensor_buffer[] = {};
    int32_t output_tensor_buffer[] = {};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}