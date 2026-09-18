/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_reshape_infershape.cpp
 * \brief InferShape and InferDataType UT for Reshape operator.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

using namespace ge;

namespace {
gert::InfershapeContextPara MakePara(const gert::StorageShape& xShape, ge::DataType xDtype,
                                     const gert::StorageShape& shapeShape, ge::DataType shapeDtype, void* shapeValue,
                                     bool isConst = true)
{
    return gert::InfershapeContextPara("Reshape",
                                       {
                                           {xShape, xDtype, ge::FORMAT_ND},
                                           {shapeShape, shapeDtype, ge::FORMAT_ND, isConst, shapeValue},
                                       },
                                       {
                                           {{{}, {}}, xDtype, ge::FORMAT_ND},
                                       });
}
} // namespace

class ReshapeInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReshapeInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReshapeInferShapeTest TearDown" << std::endl; }
};

TEST_F(ReshapeInferShapeTest, int32_shape)
{
    int32_t shapeValue[] = {3, 2};
    auto para = MakePara({{2, 3}, {2, 3}}, ge::DT_FLOAT, {{2}, {2}}, ge::DT_INT32, shapeValue);
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReshapeInferShapeTest, int64_shape_with_unknown_dim)
{
    int64_t shapeValue[] = {6, -1};
    auto para = MakePara({{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, {{2}, {2}}, ge::DT_INT64, shapeValue);
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReshapeInferShapeTest, zero_dim_copies_input_dim)
{
    int64_t shapeValue[] = {0, 12};
    auto para = MakePara({{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, {{2}, {2}}, ge::DT_INT64, shapeValue);
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 12}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReshapeInferShapeTest, empty_tensor_with_zero_dim)
{
    int64_t shapeValue[] = {0, 3};
    auto para = MakePara({{2, 0, 3}, {2, 0, 3}}, ge::DT_FLOAT, {{2}, {2}}, ge::DT_INT64, shapeValue);
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReshapeInferShapeTest, empty_tensor_with_unknown_dim)
{
    int64_t shapeValue[] = {-1, 3};
    auto para = MakePara({{2, 0, 3}, {2, 0, 3}}, ge::DT_FLOAT, {{2}, {2}}, ge::DT_INT64, shapeValue);
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReshapeInferShapeTest, duplicate_unknown_dim_fails)
{
    int64_t shapeValue[] = {-1, -1};
    auto para = MakePara({{2, 3}, {2, 3}}, ge::DT_FLOAT, {{2}, {2}}, ge::DT_INT64, shapeValue);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(ReshapeInferShapeTest, element_count_mismatch_fails)
{
    int32_t shapeValue[] = {5};
    auto para = MakePara({{2, 3}, {2, 3}}, ge::DT_FLOAT, {{1}, {1}}, ge::DT_INT32, shapeValue);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(ReshapeInferShapeTest, zero_dim_index_out_of_range_fails)
{
    int64_t shapeValue[] = {0, 0, 0};
    auto para = MakePara({{2, 3}, {2, 3}}, ge::DT_FLOAT, {{3}, {3}}, ge::DT_INT64, shapeValue);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(ReshapeInferShapeTest, non_const_shape_data_fails)
{
    auto para = MakePara({{2, 3}, {2, 3}}, ge::DT_FLOAT, {{2}, {2}}, ge::DT_INT64, nullptr, false);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
