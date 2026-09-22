/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
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

class TEST_GREATER_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "Greater", "Greater")             \
        .Input({"x1", data_types[0], shapes[0], datas[0]})           \
        .Input({"x2", data_types[1], shapes[1], datas[1]})           \
        .Output({"y", data_types[2], shapes[2], datas[2]})

#define ADD_CASE(base_type, aicpu_type)                                                  \
    TEST_F(TEST_GREATER_UT, TestGreater_##aicpu_type)                                    \
    {                                                                                    \
        vector<DataType> data_types = {aicpu_type, aicpu_type, DT_BOOL};                 \
        vector<vector<int64_t>> shapes = {{2, 4}, {4}, {2, 4}};                          \
        base_type input0[8] = {(base_type)5, (base_type)4, (base_type)6, (base_type)7,   \
                               (base_type)5, (base_type)4, (base_type)6, (base_type)7};  \
        base_type input1[4] = {(base_type)5, (base_type)2, (base_type)5, (base_type)10}; \
        bool output[40] = {0};                                                           \
        vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};             \
        CREATE_NODEDEF(shapes, data_types, datas);                                       \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                    \
        bool output_exp[8] = {0, 1, 1, 0, 0, 1, 1, 0};                                   \
        EXPECT_EQ(CompareResult<bool>(output, output_exp, 8), true);                     \
    }

ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(float, DT_FLOAT)
ADD_CASE(double, DT_DOUBLE)
ADD_CASE(int8_t, DT_INT8)
ADD_CASE(int16_t, DT_INT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
ADD_CASE(uint8_t, DT_UINT8)
ADD_CASE(uint16_t, DT_UINT16)
ADD_CASE(uint32_t, DT_UINT32)
ADD_CASE(uint64_t, DT_UINT64)

template <typename T1, typename T2, typename T3>
void RunGreaterKernel(vector<DataType> data_types, vector<vector<int64_t>>& shapes, const T1* input1_data,
                      const T2* input2_data, const T3* output_exp_data)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1 = new T1[input1_size];

    uint64_t input2_size = CalTotalElements(shapes, 1);
    T2* input2 = new T2[input2_size];

    for (uint64_t i = 0; i < input1_size; ++i) {
        input1[i] = input1_data[i];
    }
    for (uint64_t i = 0; i < input2_size; ++i) {
        input2[i] = input2_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T3* output_exp = new T3[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1;
    delete[] input2;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_GREATER_UT, X_SCALAR_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{}, {4}, {4}};
    const float input1_data[] = {81.3377f};
    const float input2_data[] = {42.911854f, 67.3148f, 23.028294f, 35.85503f};
    const bool output_exp_data[] = {true, true, true, true};

    RunGreaterKernel<float, float, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_GREATER_UT, Y_SCALAR_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{4}, {}, {4}};
    const float input1_data[] = {81.3377f, 42.911854f, 67.3148f, 23.028294f};
    const float input2_data[] = {35.85503f};
    const bool output_exp_data[] = {true, true, true, false};

    RunGreaterKernel<float, float, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_GREATER_UT, SAME_SHAPE_SMALL_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const float input1_data[] = {1.0f, 5.0f, 3.0f, -1.0f, 0.0f, 7.5f};
    const float input2_data[] = {2.0f, 4.0f, 3.0f, -2.0f, 0.0f, 7.4f};
    const bool output_exp_data[] = {false, true, false, true, false, true};

    RunGreaterKernel<float, float, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_GREATER_UT, SAME_SHAPE_LARGE_SUCCESS)
{
    const int64_t n = 40000;
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{n}, {n}, {n}};
    vector<float> in1(static_cast<size_t>(n));
    vector<float> in2(static_cast<size_t>(n));
    vector<uint8_t> exp(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        in1[static_cast<size_t>(i)] = static_cast<float>(i % 7);
        in2[static_cast<size_t>(i)] = static_cast<float>(i % 5);
        exp[static_cast<size_t>(i)] = static_cast<uint8_t>((i % 7) > (i % 5));
    }
    RunGreaterKernel<float, float, bool>(data_types, shapes, in1.data(), in2.data(),
                                         reinterpret_cast<const bool*>(exp.data()));
}

TEST_F(TEST_GREATER_UT, INNER_BROADCAST_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3, 1}, {2, 3, 4}, {2, 3, 4}};
    const float input1_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float input2_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f,
                                 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    const bool output_exp_data[] = {true, false, false, false, true, true, false, false, true, true, true, false,
                                    true, true,  true,  true,  true, true, true,  true,  true, true, true, true};

    RunGreaterKernel<float, float, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_GREATER_UT, BOTH_SCALAR_SUCCESS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    const float input1_data[] = {3.0f};
    const float input2_data[] = {2.0f};
    const bool output_exp_data[] = {true};

    RunGreaterKernel<float, float, bool>(data_types, shapes, input1_data, input2_data, output_exp_data);
}

TEST_F(TEST_GREATER_UT, EMPTY_TENSOR_SUCCESS)
{
    float input1[1] = {0.0f};
    float input2[1] = {0.0f};
    bool output[1] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_GREATER_UT, EMPTY_TENSOR_RANK3_SUCCESS)
{
    float input1[1] = {0.0f};
    float input2[1] = {0.0f};
    bool output[1] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 0, 3}, {2, 0, 3}, {2, 0, 3}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_GREATER_UT, INVALID_BROADCAST_FAILS)
{
    float input1[3] = {0.0f};
    float input2[4] = {0.0f};
    bool output[4] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{3}, {4}, {4}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, DTYPE_MISMATCH_FAILS)
{
    float input1[4] = {0.0f};
    int32_t input2[4] = {0};
    bool output[4] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, COLLAPSED_RANK_OVER_LIMIT_FAILS)
{
    float input1[32] = {0.0f};
    float input2[16] = {0.0f};
    bool output[512] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {
        {2, 1, 2, 1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, EMPTY_INVALID_BROADCAST_FAILS)
{
    float input1[1] = {0.0f};
    float input2[1] = {0.0f};
    bool output[1] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{0, 3}, {0, 4}, {0, 4}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, EMPTY_UNSUPPORTED_DTYPE_FAILS)
{
    bool input1[1] = {false};
    bool input2[1] = {false};
    bool output[1] = {false};
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, EMPTY_DTYPE_MISMATCH_FAILS)
{
    float input1[1] = {0.0f};
    int32_t input2[1] = {0};
    bool output[1] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, EMPTY_COLLAPSED_RANK_OVER_LIMIT_FAILS)
{
    float input1[1] = {0.0f};
    float input2[1] = {0.0f};
    bool output[1] = {false};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {
        {0, 1, 2, 1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2, 1}, {0, 2, 2, 2, 2, 2, 2, 2, 2}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_UT, FLOAT16_SPECIAL_VALUES_SUCCESS)
{
    Eigen::half input1[5] = {Eigen::half(-1.0f), Eigen::half(-0.0f), Eigen::half(0.0f),
                             Eigen::half(std::numeric_limits<float>::infinity()),
                             Eigen::half(std::numeric_limits<float>::quiet_NaN())};
    Eigen::half input2[5] = {Eigen::half(-2.0f), Eigen::half(0.0f), Eigen::half(-0.0f), Eigen::half(1.0f),
                             Eigen::half(0.0f)};
    bool output[5] = {false};
    bool output_exp[5] = {true, false, false, true, false};
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_BOOL};
    vector<vector<int64_t>> shapes = {{5}, {5}, {5}};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<bool>(output, output_exp, 5), true);
}

TEST_F(TEST_GREATER_UT, COLLAPSIBLE_HIGH_RANK_INT8_SUCCESS)
{
    constexpr size_t kRankBeyondPlan = 9;
    constexpr size_t kRows = 3;
    constexpr size_t kColumns = 5;
    constexpr size_t kOutputElements = kRows * kColumns;
    constexpr size_t kRowDimension = kRankBeyondPlan - 2;
    constexpr size_t kColumnDimension = kRankBeyondPlan - 1;
    int8_t x[kRows] = {-1, 0, 1};
    int8_t y[kColumns] = {-2, -1, 0, 1, 2};
    bool expected[kOutputElements] = {};
    for (size_t row = 0; row < kRows; ++row) {
        for (size_t col = 0; col < kColumns; ++col) {
            expected[row * kColumns + col] = x[row] > y[col];
        }
    }
    std::vector<int64_t> x_shape(kRankBeyondPlan, 1);
    std::vector<int64_t> y_shape(kRankBeyondPlan, 1);
    std::vector<int64_t> output_shape(kRankBeyondPlan, 1);
    x_shape[kRowDimension] = kRows;
    y_shape[kColumnDimension] = kColumns;
    output_shape[kRowDimension] = kRows;
    output_shape[kColumnDimension] = kColumns;
    std::vector<std::vector<int64_t>> shapes = {x_shape, y_shape, output_shape};
    RunGreaterKernel<int8_t, int8_t, bool>({DT_INT8, DT_INT8, DT_BOOL}, shapes, x, y, expected);
}

TEST_F(TEST_GREATER_UT, HIGH_RANK_OUTPUT_SIZE_MISMATCH_FAILS_WITHOUT_WRITE)
{
    constexpr size_t kRankBeyondPlan = 9;
    constexpr size_t kRows = 3;
    constexpr size_t kColumns = 5;
    constexpr size_t kBroadcastElements = kRows * kColumns;
    constexpr size_t kRowDimension = kRankBeyondPlan - 2;
    constexpr size_t kColumnDimension = kRankBeyondPlan - 1;
    int8_t x[kRows] = {};
    int8_t y[kColumns] = {1, 1, 1, 1, 1};
    bool output[kBroadcastElements];
    std::fill_n(output, kBroadcastElements, true);

    std::vector<int64_t> x_shape(kRankBeyondPlan, 1);
    std::vector<int64_t> y_shape(kRankBeyondPlan, 1);
    x_shape[kRowDimension] = kRows;
    y_shape[kColumnDimension] = kColumns;
    std::vector<std::vector<int64_t>> shapes = {x_shape, y_shape, {1}};
    std::vector<DataType> data_types = {DT_INT8, DT_INT8, DT_BOOL};
    std::vector<void*> datas = {x, y, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);

    for (bool value : output) {
        EXPECT_TRUE(value);
    }
}

TEST_F(TEST_GREATER_UT, SAME_SHAPE_OUTPUT_SIZE_MISMATCH_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kInputElements = 2;
    constexpr int64_t kDeclaredOutputElements = 3;
    float input1[kInputElements] = {};
    float input2[kInputElements] = {};
    bool output[kDeclaredOutputElements];
    std::fill_n(output, kDeclaredOutputElements, true);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{kInputElements}, {kInputElements}, {kDeclaredOutputElements}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);

    for (bool value : output) {
        EXPECT_TRUE(value);
    }
}

TEST_F(TEST_GREATER_UT, SAME_SHAPE_NEGATIVE_DIM_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kInvalidDimension = -1;
    float input1[1] = {};
    float input2[1] = {};
    bool output[1] = {true};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{kInvalidDimension}, {kInvalidDimension}, {kInvalidDimension}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    EXPECT_TRUE(output[0]);
}

TEST_F(TEST_GREATER_UT, X_SCALAR_NEGATIVE_DIM_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kInvalidDimension = -1;
    float input1[1] = {};
    float input2[1] = {};
    bool output[1] = {true};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{}, {kInvalidDimension}, {kInvalidDimension}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    EXPECT_TRUE(output[0]);
}

TEST_F(TEST_GREATER_UT, X_SCALAR_OUTPUT_SIZE_MISMATCH_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kInputElements = 2;
    constexpr int64_t kDeclaredOutputElements = 3;
    float input1[1] = {};
    float input2[kInputElements] = {};
    bool output[kDeclaredOutputElements];
    std::fill_n(output, kDeclaredOutputElements, true);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{}, {kInputElements}, {kDeclaredOutputElements}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);

    for (bool value : output) {
        EXPECT_TRUE(value);
    }
}

TEST_F(TEST_GREATER_UT, Y_SCALAR_OUTPUT_SIZE_MISMATCH_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kInputElements = 2;
    constexpr int64_t kDeclaredOutputElements = 3;
    float input1[kInputElements] = {};
    float input2[1] = {};
    bool output[kDeclaredOutputElements];
    std::fill_n(output, kDeclaredOutputElements, true);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{kInputElements}, {}, {kDeclaredOutputElements}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);

    for (bool value : output) {
        EXPECT_TRUE(value);
    }
}

TEST_F(TEST_GREATER_UT, EMPTY_OUTPUT_SIZE_MISMATCH_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kInputElements = 2;
    float input1[kInputElements] = {};
    float input2[kInputElements] = {};
    bool output[1] = {true};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
    vector<vector<int64_t>> shapes = {{kInputElements}, {kInputElements}, {0}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    EXPECT_TRUE(output[0]);
}

TEST_F(TEST_GREATER_UT, EMPTY_OUTPUT_OVERSIZED_BROADCAST_FAILS_WITHOUT_OVERFLOW)
{
    constexpr int64_t kUnitDimension = 1;
    constexpr int64_t kOtherDimension = 3;
    constexpr int64_t kLargeDimension = std::numeric_limits<int64_t>::max() / 2 + 1;
    uint8_t input1[kUnitDimension] = {};
    uint8_t input2[kOtherDimension] = {};
    bool output[1] = {true};
    vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_BOOL};
    vector<vector<int64_t>> shapes = {{kLargeDimension, kUnitDimension}, {kUnitDimension, kOtherDimension}, {0}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    EXPECT_TRUE(output[0]);
}

TEST_F(TEST_GREATER_UT, NONEMPTY_BROADCAST_RESULT_OVERFLOW_FAILS_WITHOUT_WRITE)
{
    constexpr int64_t kShapeDivisor = 8;
    constexpr int64_t kBeyondSafeQuotient = 2;
    constexpr int64_t kOutputElements = 16;
    constexpr int64_t kLargeDimension = std::numeric_limits<int64_t>::max() / kShapeDivisor + kBeyondSafeQuotient;
    uint8_t input1[1] = {};
    uint8_t input2[kOutputElements] = {};
    bool output[kOutputElements];
    std::fill_n(output, kOutputElements, true);
    vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1, kLargeDimension}, {kOutputElements, 1}, {kOutputElements}};
    vector<void*> datas = {input1, input2, output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);

    for (bool value : output) {
        EXPECT_TRUE(value);
    }
}
