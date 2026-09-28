/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
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
#include "graph/inference_context.h"
#include "base/registry/op_impl_space_registry_v2.h"

class VarHandleOpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "VarHandleOp SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "VarHandleOp TearDown" << std::endl; }
};

// 迁移自原仓 var_handle_op_infer_test_2：dtype 为必填属性，缺失时推导失败
TEST_F(VarHandleOpTest, var_handle_op_infer_test_2)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("VarHandleOp"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("VarHandleOp")->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};
    gert::StorageShape output_shape = {{}, {}};

    // 不设置必填属性 dtype
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("VarHandleOp")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&dummy_input})
                      .OutputShapes({&output_shape})
                      .Attr("container", ge::AscendString(""))
                      .Attr("shared_name", ge::AscendString(""))
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(VarHandleOpTest, var_handle_op_infer_shape)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("VarHandleOp"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("VarHandleOp")->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("VarHandleOp")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&dummy_input})
                      .OutputShapes({&output_shape})
                      .Attr("container", ge::AscendString(""))
                      .Attr("shared_name", ge::AscendString(""))
                      .Attr("dtype", static_cast<int64_t>(ge::DT_FLOAT))
                      .Attr("shape", std::vector<int64_t>{2, 3})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    // 输出为标量资源句柄（shape 属性描述 handle 指向变量的形状，非输出形状）
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0U);
}

TEST_F(VarHandleOpTest, var_handle_op_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("VarHandleOp"), nullptr);
    auto op_impl = space_registry->GetOpImpl("VarHandleOp");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("VarHandleOp").OpName("VarHandleOp");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(ge::AscendString(""));
    builder.AppendAttr(ge::AscendString(""));
    // dtype 属性为变量元素类型（DT_FLOAT），输出固定为 DT_RESOURCE
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT));
    builder.AppendAttr(std::vector<int64_t>{2, 3});
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_RESOURCE);
}

// 验证 RT1.0 HandleShapesAndTypes 传递的 RT2.0 等价实现：
// 经 CtInferShapeContext 扩展输入位（flat 输入 index = inputs_num + 1）注入 InferenceContext，
// 推导后 shape/dtype 属性应被写入 output handle shapes and types
TEST_F(VarHandleOpTest, var_handle_op_infer_shape_handle_shapes_and_types)
{
    auto inferCtx = ge::InferenceContext::Create();
    ASSERT_NE(inferCtx, nullptr);

    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("VarHandleOp"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("VarHandleOp")->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};

    // flat 输入布局：[0]=节点输入，[inputs_num+0]=InferShapeFunc(编译期 null，占位)，[inputs_num+1]=InferenceContext
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("VarHandleOp")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_RESOURCE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors(
                          {(gert::Tensor*)&dummy_input, (gert::Tensor*)&dummy_input, (gert::Tensor*)inferCtx.get()})
                      .Attr("container", ge::AscendString(""))
                      .Attr("shared_name", ge::AscendString(""))
                      .Attr("dtype", static_cast<int64_t>(ge::DT_FLOAT))
                      .Attr("shape", std::vector<int64_t>{2, 3})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    // 输出为标量资源句柄
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0U);

    // handle 所指向变量的 shape/dtype 已传递至 InferenceContext
    const auto& handleShapes = inferCtx->GetOutputHandleShapesAndTypes();
    ASSERT_EQ(handleShapes.size(), 2U);
    ASSERT_EQ(handleShapes[0].size(), 1U);
    EXPECT_EQ(handleShapes[0][0].GetDataType(), ge::DT_FLOAT);
    auto varShape = handleShapes[0][0].GetShape();
    ASSERT_EQ(varShape.GetDimNum(), 2U);
    EXPECT_EQ(varShape.GetDim(0), 2);
    EXPECT_EQ(varShape.GetDim(1), 3);
}
