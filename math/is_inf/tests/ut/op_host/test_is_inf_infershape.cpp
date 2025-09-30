/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

// ----------------IsInf--------------
class is_inf : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "is_inf SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "is_inf TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(
    const std::vector<gert::StorageShape>& inputShapes, // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,            // 存储所有DataType参数
    gert::StorageShape& outStorageShape, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& xStorageShape = inputShapes[0];

    ge::DataType input1Dtype = dtypes[0];
    ge::DataType outputDtype = dtypes[1];

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xStorageShape,
    };
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("IsInf")
                             .NodeIoNum(1, 1)
                             .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("IsInf")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(is_inf, is_inf_infershape_cast_0)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{3, 4}, {3, 4}}, // x_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT, // input1Dtype
        ge::DT_BOOL   // outputDtype
    };

    std::vector<int64_t> expectResult = {3, 4};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase(inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(outStorageShape.GetOriginShape()), expectResult);
}
