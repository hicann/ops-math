/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_adjacent_difference.cpp
 * \brief Unit test for l0op::AdjacentDifference op_api interface.
 * The operator compares adjacent elements: y[0]=0, y[i]=(x[i]==x[i-1]?0:1)
 */

#include <vector>
#include "gtest/gtest.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "math/adjacent_difference/op_api/adjacent_difference.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 64;

class AdjacentDifferenceL0Test : public ::testing::Test {
public:
    AdjacentDifferenceL0Test() : exe(nullptr) {}

    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override { delete exe; }

public:
    aclOpExecutor* exe;
    int64_t data[DATA_SIZE] = {1};
};

// ==================== dtype 覆盖：输出 INT32 ====================

// float 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_FLOAT_OUTPUT_INT32)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({3, 4, 5});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// float16 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_FLOAT16_OUTPUT_INT32)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT16);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({3, 4, 5});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// bfloat16 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_BFLOAT16_OUTPUT_INT32)
{
    auto self = CreateAclTensor({2, 8}, ACL_BF16);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({2, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// int8 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_INT8_OUTPUT_INT32)
{
    auto self = CreateAclTensor({4, 8}, ACL_INT8);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({4, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// int16 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_INT16_OUTPUT_INT32)
{
    auto self = CreateAclTensor({4, 8}, ACL_INT16);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({4, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// int32 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_INT32_OUTPUT_INT32)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_INT32);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({3, 4, 5});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// int64 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_INT64_OUTPUT_INT32)
{
    auto self = CreateAclTensor({2, 3, 4}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({2, 3, 4});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// uint8 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_UINT8_OUTPUT_INT32)
{
    auto self = CreateAclTensor({4, 8}, ACL_UINT8);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({4, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// uint16 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_UINT16_OUTPUT_INT32)
{
    auto self = CreateAclTensor({4, 8}, ACL_UINT16);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({4, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// uint32 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_UINT32_OUTPUT_INT32)
{
    auto self = CreateAclTensor({4, 8}, ACL_UINT32);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({4, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// uint64 类型输入, 输出 int32
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_UINT64_OUTPUT_INT32)
{
    auto self = CreateAclTensor({4, 8}, ACL_UINT64);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({4, 8});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT32);
}

// ==================== dtype 覆盖：输出 INT64 ====================

// float 类型输入, 输出 int64
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_FLOAT_OUTPUT_INT64)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT64, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({3, 4, 5});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT64);
}

// int32 类型输入, 输出 int64
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_INT32_OUTPUT_INT64)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_INT32);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT64, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({3, 4, 5});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT64);
}

// int64 类型输入, 输出 int64
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_INT64_OUTPUT_INT64)
{
    auto self = CreateAclTensor({2, 3, 4}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT64, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({2, 3, 4});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
    EXPECT_EQ(out->GetDataType(), DataType::DT_INT64);
}

// ==================== shape 边界覆盖 ====================

// 1维 tensor
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_1D)
{
    auto self = CreateAclTensor({10}, ACL_FLOAT);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({10});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
}

// 5维 tensor
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_5D)
{
    auto self = CreateAclTensor({2, 2, 2, 2, 2}, ACL_FLOAT);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({2, 2, 2, 2, 2});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
}

// 空 tensor (shape 含 0 维)
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_EMPTY_TENSOR)
{
    auto self = CreateAclTensor({0, 3}, ACL_FLOAT);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({0, 3});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
}

// 单元素 tensor: y[0]=0, 无相邻元素可比
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_SINGLE_ELEMENT)
{
    auto self = CreateAclTensor({1}, ACL_FLOAT);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({1});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
}

// 非连续 stride tensor
TEST_F(AdjacentDifferenceL0Test, AdjacentDifference_SUCC_NOT_CONTIGUOUS)
{
    std::vector<int64_t> shape = {2, 5};
    std::vector<int64_t> strides = {5, 1};
    auto self = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                shape.size(), data);
    ASSERT_NE(self, nullptr);
    auto out = l0op::AdjacentDifference(self, DataType::DT_INT32, exe);
    ASSERT_NE(out, nullptr);

    op::ShapeVector expectShape({2, 5});
    EXPECT_EQ(op::ToShapeVector(out->GetViewShape()), expectShape);
}
