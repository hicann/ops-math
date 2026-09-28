/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PlaceholderWithDefaultTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PlaceholderWithDefault SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PlaceholderWithDefault TearDown" << std::endl; }
};

TEST_F(PlaceholderWithDefaultTest, placeholder_withdefault_infer_shape)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("PlaceholderWithDefault"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("PlaceholderWithDefault")->infer_shape;
    gert::StorageShape input_shape = {{2, 3}, {2, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("PlaceholderWithDefault")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("shape", std::vector<int64_t>{2, 3})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2U);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(PlaceholderWithDefaultTest, placeholder_withdefault_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("PlaceholderWithDefault"), nullptr);
    auto op_impl = space_registry->GetOpImpl("PlaceholderWithDefault");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("PlaceholderWithDefault").OpName("PlaceholderWithDefault");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(std::vector<int64_t>{2, 3});
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
