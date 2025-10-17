/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "register/op_impl_registry_base.h"
#include "kernel_run_context_facker.h"
#include "common/utils/ut_op_common.h"

class RightShiftRt2UTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RightShiftRt2UTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RightShiftRt2UTest TearDown" << std::endl;
    }
};

TEST_F(RightShiftRt2UTest, InferShape_succ)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape y_shape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape z_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape, &y_shape})
                      .OutputShapes({&z_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(2), 4);
}

TEST_F(RightShiftRt2UTest, InferShape_fail_cannot_board)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape y_shape = {{1, 4, 4}, {1, 4, 4}};
    gert::StorageShape z_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape, &y_shape})
                      .OutputShapes({&z_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

TEST_F(RightShiftRt2UTest, InferDataType_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);
    ge::DataType x_type = ge::DT_INT8;
    ge::DataType y_type = ge::DT_UNDEFINED;
    ge::DataType z_type = ge::DT_UNDEFINED;
    int64_t dtype = static_cast<int64_t>(ge::DT_INT64);
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(2, 1)
                              .IrInputNum(2)
                              .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&x_type, &y_type})
                              .OutputDataTypes({&z_type})
                              .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(dtype)}})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

TEST_F(RightShiftRt2UTest, InferShape_dynamic_succ_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape y_shape = {{-2}, {-2}};
    gert::StorageShape z_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape, &y_shape})
                      .OutputShapes({&z_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(RightShiftRt2UTest, InferShape_dynamic_succ_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RightShift")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape y_shape = {{-1, 3, 1}, {-1, 3, 1}};
    gert::StorageShape z_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x_shape, &y_shape})
                      .OutputShapes({&z_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 2);
}