/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
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

using namespace std;
using namespace aicpu;

class TEST_CLIP_BY_VALUE_V2_UT : public testing::Test {};

template <typename T>
bool CompareResultCompatibleWithNAN(T output[], T expect_output[], uint64_t num)
{
    bool result = true;
    for (uint64_t i = 0; i < num; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(output[i]) && std::isnan(expect_output[i])) {
                continue;
            }
        }
        if (output[i] != expect_output[i]) {
            std::cout << "output[" << i << "] = ";
            std::cout << output[i];
            std::cout << ", expect_output[" << i << "] = ";
            std::cout << expect_output[i] << std::endl;
            result = false;
        }
    }
    return result;
}

#define CREATE_NODEDEF(shapes, data_types, datas)                      \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
    NodeDefBuilder(node_def.get(), "ClipByValueV2", "ClipByValueV2")   \
        .Input({"x", data_types[0], shapes[0], datas[0]})              \
        .Input({"clip_value_min", data_types[1], shapes[1], datas[1]}) \
        .Input({"clip_value_max", data_types[2], shapes[2], datas[2]}) \
        .Output({"y", data_types[3], shapes[3], datas[3]});

#define ADD_CASE(base_type, aicpu_type)                                                        \
    TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TestClipByValueV2_##aicpu_type)                           \
    {                                                                                          \
        vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type, aicpu_type};        \
        vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {1, 4}, {1, 4}};                     \
        base_type input0[4] = {(base_type)1, (base_type)2, (base_type)5, (base_type)4};        \
        base_type input1[4] = {(base_type)1, (base_type)0, (base_type)1, (base_type)0};        \
        base_type input2[4] = {(base_type)3, (base_type)1, (base_type)4, (base_type)3};        \
        base_type expect_output[4] = {(base_type)1, (base_type)1, (base_type)4, (base_type)3}; \
        base_type output[4] = {(base_type)0};                                                  \
        vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};    \
        CREATE_NODEDEF(shapes, data_types, datas);                                             \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                          \
        EXPECT_EQ(CompareResult<base_type>(output, expect_output, 4), true);                   \
    }

ADD_CASE(int8_t, DT_INT8)
ADD_CASE(int16_t, DT_INT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
ADD_CASE(uint8_t, DT_UINT8)
ADD_CASE(uint16_t, DT_UINT16)
ADD_CASE(uint32_t, DT_UINT32)
ADD_CASE(uint64_t, DT_UINT64)
ADD_CASE(uint8_t, DT_QUINT8)
ADD_CASE(int8_t, DT_QINT8)
ADD_CASE(int32_t, DT_QINT32)
ADD_CASE(double, DT_DOUBLE)
ADD_CASE(float, DT_FLOAT)
ADD_CASE(Eigen::bfloat16, DT_BFLOAT16)
ADD_CASE(Eigen::half, DT_FLOAT16)

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_COMPLEX64_SUCCESS)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {1, 4}, {1, 4}};
    std::complex<float> input0[4] = {
        std::complex<float>(1, 1), std::complex<float>(2, 2), std::complex<float>(3, 3), std::complex<float>(4, 4)};
    std::complex<float> input1[4] = {
        std::complex<float>(2, 1), std::complex<float>(2, 2), std::complex<float>(5, 4), std::complex<float>(4, 4)};
    std::complex<float> input2[4] = {
        std::complex<float>(1, 1), std::complex<float>(3, 2), std::complex<float>(4, 3), std::complex<float>(3, 4)};
    std::complex<float> expect_output[4] = {
        std::complex<float>(1, 1), std::complex<float>(2, 2), std::complex<float>(4, 3), std::complex<float>(3, 4)};
    std::complex<float> output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<std::complex<float>>(output, expect_output, 4), true);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_COMPLEX64_SCALAR_SUCCESS)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
    std::complex<float> input0 = std::complex<float>(3, 3);
    std::complex<float> input1 = std::complex<float>(5, 4);
    std::complex<float> input2 = std::complex<float>(4, 3);
    std::complex<float> expect_output = std::complex<float>(4, 3);
    std::complex<float> output = {std::complex<float>(0, 0)};
    vector<void*> datas = {(void*)&input0, (void*)&input1, (void*)&input2, (void*)&output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(std::abs(expect_output), std::abs(output));
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_COMPLEX64_MIN_AND_MAX_SUCCESS)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{1, 4}, {}, {}, {1, 4}};
    std::complex<float> input0[4] = {
        std::complex<float>(1, 1), std::complex<float>(2, 2), std::complex<float>(3, 3), std::complex<float>(4, 4)};
    std::complex<float> input1 = std::complex<float>(5, 4);
    std::complex<float> input2 = std::complex<float>(4, 3);
    std::complex<float> expect_output[4] = {
        std::complex<float>(4, 3), std::complex<float>(4, 3), std::complex<float>(4, 3), std::complex<float>(4, 3)};
    std::complex<float> output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)&input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<std::complex<float>>(output, expect_output, 4), true);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_COMPLEX128_SUCCESS)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {1, 4}, {1, 4}};
    std::complex<double> input0[4] = {
        std::complex<double>(1, 1), std::complex<double>(2, 2), std::complex<double>(3, 3), std::complex<double>(4, 4)};
    std::complex<double> input1[4] = {
        std::complex<double>(2, 1), std::complex<double>(2, 2), std::complex<double>(5, 4), std::complex<double>(4, 4)};
    std::complex<double> input2[4] = {
        std::complex<double>(1, 1), std::complex<double>(3, 2), std::complex<double>(4, 3), std::complex<double>(3, 4)};
    std::complex<double> expect_output[4] = {
        std::complex<double>(1, 1), std::complex<double>(2, 2), std::complex<double>(4, 3), std::complex<double>(3, 4)};
    std::complex<double> output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<std::complex<double>>(output, expect_output, 4), true);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_MIN_AND_MAX_SCALAR_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 4}, {}, {}, {1, 4}};
    float input0[4] = {(float)33.3, (float)44.4, (float)55.5, (float)66.6};
    float input1 = (float)35.1;
    float input2 = (float)56.1;
    float expect_output[4] = {(float)35.1, (float)44.4, (float)55.5, (float)56.1};
    float output[4] = {(float)0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)&input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<float>(output, expect_output, 4), true);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_SCALAR_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
    float input0 = (float)33.3;
    float input1 = (float)35.1;
    float input2 = (float)56.1;
    float expect_output = (float)35.1;
    float output = {(float)0};
    vector<void*> datas = {(void*)&input0, (void*)&input1, (void*)&input2, (void*)&output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(expect_output, output);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_SCALAR_MAX_SMALLER_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 4}, {}, {}, {1, 4}};
    float input0[4] = {(float)33.3, (float)44.4, (float)55.5, (float)66.6};
    float input1 = (float)56.1;
    float input2 = (float)35.1;
    float expect_output[4] = {(float)35.1, (float)35.1, (float)35.1, (float)35.1};
    float output[4] = {(float)0};
    vector<void*> datas = {(void*)input0, (void*)&input1, (void*)&input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<float>(output, expect_output, 4), true);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_TENSOR_MAX_SMALLER_SUCCESS)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {1, 4}, {1, 4}};
    int8_t input0[4] = {1, 2, 5, 4};
    int8_t input1[4] = {3, 1, 4, 3};
    int8_t input2[4] = {1, 0, 1, 0};
    int8_t expect_output[4] = {1, 0, 1, 0};
    int8_t output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<int8_t>(output, expect_output, 4), true);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_EMPTY_TENSOR_SUCCESS)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{0, 0}, {1, 4}, {1, 4}, {1, 4}};
    int8_t input1[4] = {3, 1, 4, 3};
    int8_t input2[4] = {1, 0, 1, 0};
    int8_t expect_output[4] = {1, 0, 1, 0};
    int8_t output[4] = {0};
    vector<void*> datas = {nullptr, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_BOOL_FAILED)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1}, {}, {}, {1}};
    bool input0[1] = {true};
    bool input1 = false;
    bool input2 = true;
    bool output[1] = {true};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_ONLY_MIN_FAILED)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {0}, {1, 4}};
    int8_t input0[4] = {1, 2, 5, 4};
    int8_t input1[4] = {1, 2, 5, 4};
    int8_t output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_ONLY_MAX_FAILED)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {0}, {1, 4}};
    int8_t input0[4] = {1, 2, 5, 4};
    int8_t input1[4] = {1, 2, 5, 4};
    int8_t output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)nullptr, (void*)input1, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_DTYPE_NOT_SAME_FAILED)
{
    vector<DataType> data_types = {DT_INT8, DT_UINT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {1, 4}, {1, 4}};
    int8_t input0[4] = {1, 2, 5, 4};
    int8_t input1[4] = {1, 2, 5, 4};
    int8_t input2[4] = {1, 2, 5, 4};
    int8_t output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_CHECK_SHAPE_FAILED)
{
    vector<DataType> data_types = {DT_INT8, DT_UINT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 3}, {2, 3}};
    int8_t input0[6] = {1, 2, 3, 4, 5, 6};
    int8_t input1[6] = {1, 2, 3, 4, 5, 6};
    int8_t input2[6] = {1, 2, 3, 4, 5, 6};
    int8_t output[6] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CLIP_BY_VALUE_V2_UT, TEST_NAN_SUCCESS)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{1, 4}, {1, 4}, {1, 4}, {1, 4}};
    double input0[4] = {NAN, NAN, NAN, NAN};
    double input1[4] = {1, 1, 1, 1};
    double input2[4] = {5, 5, 5, 5};
    double expect_output[4] = {NAN, NAN, NAN, NAN};
    double output[4] = {0};
    vector<void*> datas = {(void*)input0, (void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResultCompatibleWithNAN<double>(output, expect_output, 4), true);
}