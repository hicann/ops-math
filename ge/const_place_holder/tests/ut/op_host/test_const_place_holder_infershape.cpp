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

using namespace ge;

class ConstPlaceHolderProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConstPlaceHolder Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConstPlaceHolder Proto Test TearDown" << std::endl; }
};

TEST_F(ConstPlaceHolderProtoTest, test_const_place_holder_rt2_infer_shape)
{
    ge::AscendString op_type("ConstPlaceHolder");
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl(op_type.GetString()), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl(op_type.GetString())->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};
    gert::StorageShape storage_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(op_type.GetString())
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&dummy_input})
                      .OutputShapes({&storage_shape})
                      .Attr("origin_shape", std::vector<int64_t>{1, 1, 2, 2})
                      .Attr("origin_format", static_cast<int64_t>(0))
                      .Attr("storage_shape", std::vector<int64_t>{1, 1, 2, 2})
                      .Attr("storage_format", static_cast<int64_t>(0))
                      .Attr("expand_dim_rules", ge::AscendString(""))
                      .Attr("dtype", static_cast<int64_t>(ge::DT_FLOAT16))
                      .Attr("addr", static_cast<int64_t>(123456))
                      .Attr("size", static_cast<int64_t>(123456))
                      .Build();

    EXPECT_NE(infer_shape_func, nullptr);
    auto ret = infer_shape_func(holder.GetContext());
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto out_shape = holder.GetContext()->GetOutputShape(0);
    EXPECT_EQ(out_shape->GetDimNum(), 4U);
    EXPECT_EQ(out_shape->GetDim(0), 1);
    EXPECT_EQ(out_shape->GetDim(1), 1);
    EXPECT_EQ(out_shape->GetDim(2), 2);
    EXPECT_EQ(out_shape->GetDim(3), 2);
}

TEST_F(ConstPlaceHolderProtoTest, test_const_place_holder_rt2_infer_dtype)
{
    ge::AscendString op_type("ConstPlaceHolder");
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl(op_type.GetString()), nullptr);
    auto op_impl = space_registry->GetOpImpl(op_type.GetString());
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType(op_type).OpName(op_type);
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(std::vector<int64_t>{1, 1, 2, 2});
    builder.AppendAttr(static_cast<int64_t>(0));
    builder.AppendAttr(std::vector<int64_t>{1, 1, 2, 2});
    builder.AppendAttr(static_cast<int64_t>(0));
    builder.AppendAttr(ge::AscendString(""));
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT16));
    builder.AppendAttr(static_cast<int64_t>(123456));
    builder.AppendAttr(static_cast<int64_t>(123456));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto out_dtype = context->GetOutputDataType(0);
    EXPECT_EQ(out_dtype, ge::DT_FLOAT16);
}
