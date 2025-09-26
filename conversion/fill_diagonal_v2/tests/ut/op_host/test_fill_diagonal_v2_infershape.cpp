/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fill_diagonal_v2_proto.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FillDiagonalV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fill_diagonal_v2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fill_diagonal_v2 Proto Test TearDown" << std::endl;
  }
};

static std::vector<int64_t> ToVectorForFillDiagonalV2(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForFillDiagonalV2(
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS,
    bool attr = false)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& selfStorageShape = inputShapes[0];
    const auto& fillValueStorageShape = inputShapes[1];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType outputDtype = dtypes[2];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&selfStorageShape,
        (gert::Tensor *)&fillValueStorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("FillDiagonalV2")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Attr("wrap", attr)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("FillDiagonalV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(FillDiagonalV2Test, fill_diagonal_v2_infer_shape)
{
    size_t size1 = 16;

    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{size1, size1}, {size1, size1}},    // self_shape
        {{size1, size1}, {size1, size1}}     // fill_value_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT16,  // input1Dtype
        ge::DT_FLOAT,    // input2Dtype
        ge::DT_FLOAT16   // outputDtype
    };

    std::vector<int64_t> expectResult = {size1, size1};
    gert::StorageShape outStorageShape = {};
    bool attr = false;
    // 简化后的函数调用
    ExeTestCaseForFillDiagonalV2(inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS, attr);
    EXPECT_EQ(ToVectorForFillDiagonalV2(outStorageShape.GetOriginShape()), expectResult);
}