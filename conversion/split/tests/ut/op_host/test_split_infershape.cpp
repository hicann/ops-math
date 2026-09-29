/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "op_infer_shape_range_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SplitInfershape : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// Test: Split infershape with same shape
TEST_F(SplitInfershape, split_infershape_same_shape)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{11, 16}, {11, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{11, 8}, {11, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with invalid shape
TEST_F(SplitInfershape, split_infershape_invalid_xshape)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with invalid shape
TEST_F(SplitInfershape, split_infershape_invalid_split_dim)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with invalid dtype
TEST_F(SplitInfershape, split_infershape_invalid_dtype)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &split_dim},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with negative split_dim (relative indexing)
TEST_F(SplitInfershape, split_infershape_negative_split_dim)
{
    int32_t split_dim = -1;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with split_dim not get const (unknown split_dim)
TEST_F(SplitInfershape, split_infershape_unknown_split_dim)
{
    std::vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND, true, splitDimValue.data()},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8}, {2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with dynamic dim (-1) in split dimension
TEST_F(SplitInfershape, split_infershape_dynamic_dim)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with num_split equals 1
TEST_F(SplitInfershape, split_infershape_single_split)
{
    int32_t split_dim = 0;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with 3D tensor
TEST_F(SplitInfershape, split_infershape_3d_tensor)
{
    int32_t split_dim = 0;
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
                                                          {{{6, 4, 8}, {6, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8}, {2, 4, 8}, {2, 4, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType: x is bfloat16, all dynamic outputs (num_split=2) must be bfloat16
TEST_F(SplitInfershape, split_infer_datatype_bf16_multi_output)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Split");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Split").OpName("Split");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
}

// InferDataType: x is float32, all dynamic outputs (num_split=3) must be float32
TEST_F(SplitInfershape, split_infer_datatype_float_multi_output)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Split");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Split").OpName("Split");
    builder.IONum(2, 3);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(2, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

// Test: num_split is 1 and split_dim is not const, output keeps the input shape
TEST_F(SplitInfershape, split_infershape_num_split_1_split_dim_not_const)
{
    gert::InfershapeContextPara infershapeContextPara("Split",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 用真实 Tensor 对承载输入 shape range：Tensor 的 origin shape 即 range 值。
// min/max Tensor 的 dtype/format/expand_dims_type 保持一致，满足
// OpInferShapeRangeContextBuilder::InputTensorsRange 的一致性校验（CANN 9.3.0 起强制）。
struct XShapeRange {
    gert::Tensor minTensor;
    gert::Tensor maxTensor;
    gert::Range<gert::Tensor> range;
    XShapeRange(std::initializer_list<int64_t> minDims, std::initializer_list<int64_t> maxDims)
        : minTensor(gert::StorageShape(minDims, minDims),
                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                    gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr),
          maxTensor(gert::StorageShape(maxDims, maxDims),
                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                    gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr),
          range(&minTensor, &maxTensor)
    {}
};

static std::vector<int64_t> ShapeToVec(const gert::Shape& shape)
{
    std::vector<int64_t> shapeVec;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shapeVec.push_back(shape.GetDim(i));
    }
    return shapeVec;
}

// Test: InferShapeRange with const negative split_dim (-1, last axis), split axis range is divided
TEST_F(SplitInfershape, split_infer_shape_range_const_negative_dim)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Split");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Split").OpName("Split");
    builder.IONum(2, 2);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(2));

    int32_t splitDimValue = -1;
    gert::Tensor splitDimTensor(gert::StorageShape({1}, {1}),
                                gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                gert::TensorPlacement::kOnHost, ge::DT_INT32, &splitDimValue);
    gert::Range<gert::Tensor> splitDimRange(&splitDimTensor, &splitDimTensor);

    XShapeRange xRange({1, 4}, {4, 8});

    builder.InputTensorsRange({&splitDimRange, &xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    // axis = -1 + 2 = 1: axis0 range keeps [1,4], axis1 range becomes [1, 4]
    for (size_t outIdx = 0; outIdx < 2; outIdx++) {
        auto* yRange = context->GetOutputShapeRange(outIdx);
        ASSERT_NE(yRange, nullptr);
        ASSERT_NE(yRange->GetMax(), nullptr);
        ASSERT_NE(yRange->GetMin(), nullptr);
        EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({1, 2}));
        EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({4, 4}));
    }
}

// Test: InferShapeRange with const split_dim 0, split axis range is divided (min==1 keeps 1)
TEST_F(SplitInfershape, split_infer_shape_range_const_dim0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Split");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Split").OpName("Split");
    builder.IONum(2, 2);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(2));

    int32_t splitDimValue = 0;
    gert::Tensor splitDimTensor(gert::StorageShape({1}, {1}),
                                gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                gert::TensorPlacement::kOnHost, ge::DT_INT32, &splitDimValue);
    gert::Range<gert::Tensor> splitDimRange(&splitDimTensor, &splitDimTensor);

    XShapeRange xRange({1, 4}, {4, 8});

    builder.InputTensorsRange({&splitDimRange, &xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    // axis 0: min keeps 1 (==1), max ceil(4/2)=2; axis 1: [4, 8] copied
    for (size_t outIdx = 0; outIdx < 2; outIdx++) {
        auto* yRange = context->GetOutputShapeRange(outIdx);
        ASSERT_NE(yRange, nullptr);
        ASSERT_NE(yRange->GetMax(), nullptr);
        ASSERT_NE(yRange->GetMin(), nullptr);
        EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({1, 4}));
        EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({2, 8}));
    }
}

// Test: InferShapeRange with non-const split_dim and 1-D x, min is 0 and max is ceil(x_max/num_split)
TEST_F(SplitInfershape, split_infer_shape_range_not_const_1d)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Split");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Split").OpName("Split");
    builder.IONum(2, 2);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(2));

    gert::Tensor splitDimTensor(gert::StorageShape({1}, {1}),
                                gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Range<gert::Tensor> splitDimRange(&splitDimTensor, &splitDimTensor);

    XShapeRange xRange({0}, {100});

    builder.InputTensorsRange({&splitDimRange, &xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    // min is 0 (not ceil(x_max/num_split)), max is ceil(100/2)=50
    for (size_t outIdx = 0; outIdx < 2; outIdx++) {
        auto* yRange = context->GetOutputShapeRange(outIdx);
        ASSERT_NE(yRange, nullptr);
        ASSERT_NE(yRange->GetMax(), nullptr);
        ASSERT_NE(yRange->GetMin(), nullptr);
        EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({0}));
        EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({50}));
    }
}

// Test: InferShapeRange with non-const split_dim and multi-dim x, all dims keep x max with min 0
TEST_F(SplitInfershape, split_infer_shape_range_not_const_multi_dim)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Split");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Split").OpName("Split");
    builder.IONum(2, 2);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(2));

    gert::Tensor splitDimTensor(gert::StorageShape({1}, {1}),
                                gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Range<gert::Tensor> splitDimRange(&splitDimTensor, &splitDimTensor);

    XShapeRange xRange({2, 4}, {4, 8});

    builder.InputTensorsRange({&splitDimRange, &xRange.range});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    for (size_t outIdx = 0; outIdx < 2; outIdx++) {
        auto* yRange = context->GetOutputShapeRange(outIdx);
        ASSERT_NE(yRange, nullptr);
        ASSERT_NE(yRange->GetMax(), nullptr);
        ASSERT_NE(yRange->GetMin(), nullptr);
        EXPECT_EQ(ShapeToVec(*(yRange->GetMin())), std::vector<int64_t>({0, 0}));
        EXPECT_EQ(ShapeToVec(*(yRange->GetMax())), std::vector<int64_t>({4, 8}));
    }
}
