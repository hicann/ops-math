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

class TEST_SQUAREDDIFFERENCE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
    NodeDefBuilder(node_def.get(), "SquaredDifference", "SquaredDifference") \
        .Input({"x1", data_types[0], shapes[0], datas[0]})                   \
        .Input({"x2", data_types[1], shapes[1], datas[1]})                   \
        .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_X_NUM_ONE_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {1, 3}, {1, 3}};

    constexpr uint64_t input1_size = 1;
    int32_t input1[input1_size] = {2};

    constexpr uint64_t input2_size = 3;
    int32_t input2[input2_size] = {7, 0, 4};

    constexpr uint64_t output_size = 3;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size] = {25, 4, 4};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_Y_NUM_ONESUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 3}, {1}, {1, 3}};

    constexpr uint64_t input1_size = 3;
    int32_t input1[input1_size] = {2, 7, 0};

    constexpr uint64_t input2_size = 1;
    int32_t input2[input2_size] = {4};

    constexpr uint64_t output_size = 3;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size] = {4, 9, 16};
    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4, 16}, {1, 16}, {4, 16}};

    constexpr uint64_t input1_size = 4 * 16;
    int32_t input1[input1_size] = {2, 7, 0, 4, 2, 1, 3, 3, 3, 9, 1, 2, 0, 7, 0, 5, 3, 9, 4, 7, 8, 5,
                                   6, 6, 7, 7, 6, 4, 0, 1, 5, 7, 5, 3, 7, 0, 8, 9, 8, 9, 6, 3, 7, 6,
                                   9, 5, 9, 4, 4, 2, 7, 2, 1, 9, 6, 8, 8, 9, 6, 2, 7, 9, 1, 2};

    constexpr uint64_t input2_size = 16;
    int32_t input2[input2_size] = {6, 3, 1, 2, 5, 6, 4, 8, 6, 6, 1, 8, 5, 7, 2, 3};

    constexpr uint64_t output_size = 4 * 16;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size] = {16, 16, 1,  4,  9,  25, 1,  25, 9, 9, 0,  36, 25, 0,  4,  4,
                                       9,  36, 9,  25, 9,  1,  4,  4,  1, 1, 25, 16, 25, 36, 9,  16,
                                       1,  0,  36, 4,  9,  9,  16, 1,  0, 9, 36, 4,  16, 4,  49, 1,
                                       4,  1,  36, 0,  16, 9,  4,  0,  4, 9, 25, 36, 4,  4,  1,  1};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}
