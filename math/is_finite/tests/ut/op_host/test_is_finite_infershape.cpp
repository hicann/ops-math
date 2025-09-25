/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class IsFiniteInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "IsFiniteInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IsFiniteInfershape TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const gert::StorageShape& xStorageShape,
                        ge::DataType inputDtype,
                        ge::DataType outputDtype,
                        gert::StorageShape& yStorageShape,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {(gert::Tensor *)&xStorageShape};
    std::vector<gert::StorageShape *> outputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("IsFinite")
        .NodeIoNum(1, 1)
        .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("IsFinite")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
}

TEST_F(IsFiniteInfershape, is_finite_infershape_test1)
{
    gert::StorageShape xStorageShape = {{3, 4}, {3, 4}};
    gert::StorageShape yStorageShape = {};
    std::vector<int64_t> expectResult = {3, 4};
    ExeTestCase(xStorageShape, ge::DT_FLOAT, ge::DT_BOOL, yStorageShape, ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(yStorageShape.GetOriginShape()), expectResult);
}

TEST_F(IsFiniteInfershape, is_finite_infershape_test2)
{
    gert::StorageShape xStorageShape = {{5, -1}, {5, -1}};
    gert::StorageShape yStorageShape = {};
    std::vector<int64_t> expectResult = {5, -1};
    ExeTestCase(xStorageShape, ge::DT_FLOAT, ge::DT_BOOL, yStorageShape, ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(yStorageShape.GetOriginShape()), expectResult);
}

//TEST_F(is_finite, is_finite_infershape_test3)
//{
//    ge::op::IsFinite op;
//    op.UpdateInputDesc("x", create_desc_with_ori({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {-2}, ge::FORMAT_ND));
//
//    auto ret = op.InferShapeAndType();
//    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//
//    auto output_desc = op.GetOutputDescByName("y");
//    EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
//
//    std::vector<int64_t> expected_output_y_shape = {-2};
//    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
//}
//
//TEST_F(is_finite, is_finite_infershape_test4)
//{
//    ge::op::IsFinite op;
//    op.UpdateInputDesc(
//        "x", create_desc_with_ori(
//                 {4, 5, 6, 7, 8, 1, 5, 9}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 5, 6, 7, 8, 1, 5, 9}, ge::FORMAT_ND));
//
//    auto ret = op.InferShapeAndType();
//    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//
//    auto output_desc = op.GetOutputDescByName("y");
//    EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
//
//    std::vector<int64_t> expected_output_y_shape = {4, 5, 6, 7, 8, 1, 5, 9};
//    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
//}
//
//static ge::Operator BuildGraph4InferAxisType(
//    const std::initializer_list<int64_t>& dims1, const ge::Format& format, const ge::DataType& dataType)
//{
//    auto apply_op = op::IsFinite("IsFinite");
//    TENSOR_INPUT_WITH_SHAPE(apply_op, x, dims1, dataType, format, {});
//    apply_op.InferShapeAndType();
//    return apply_op;
//}
//
//TEST_F(is_finite, infer_axis_type_case1)
//{
//    auto op1 = BuildGraph4InferAxisType({-1, -1}, ge::FORMAT_ND, ge::DT_FLOAT16);
//
//    ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("IsFinite");
//    EXPECT_NE(infer_func, nullptr);
//
//    std::vector<ge::AxisTypeInfo> infos;
//    const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
//    EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);
//
//    std::vector<ge::AxisTypeInfo> expect_infos = {
//        ge::AxisTypeInfoBuilder()
//            .AxisType(ge::AxisType::ELEMENTWISE)
//            .AddInputCutInfo({0, {0}})
//            .AddOutputCutInfo({0, {0}})
//            .Build(),
//        ge::AxisTypeInfoBuilder()
//            .AxisType(ge::AxisType::ELEMENTWISE)
//            .AddInputCutInfo({0, {1}})
//            .AddOutputCutInfo({0, {1}})
//            .Build(),
//    };
//
//    EXPECT_STREQ(AxisTypeInfoToString(infos).c_str(), AxisTypeInfoToString(expect_infos).c_str());
//}
//
//TEST_F(is_finite, infer_axis_type_case2)
//{
//    auto op1 = BuildGraph4InferAxisType(
//        {
//            -2,
//        },
//        ge::FORMAT_ND, ge::DT_FLOAT16);
//
//    ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("IsFinite");
//    EXPECT_NE(infer_func, nullptr);
//
//    std::vector<ge::AxisTypeInfo> infos;
//    const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
//    EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);
//
//    EXPECT_EQ(infos.size(), 0);
//}