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
 * \file test_unfold_grad_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"
#include "graph/operator_factory_impl.h"
#include "common/utils/ut_op_common.h"
#include "graph/utils/op_desc_utils.h"

class UnfoldGradInferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "UnfoldGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UnfoldGrad Proto Test TearDown" << std::endl;
    }

    template <typename T>
    gert::Tensor* ConstructInputConstTensor(
        std::unique_ptr<uint8_t[]>& input_tensor_holder, const vector<T>& const_value, ge::DataType const_dtype)
    {
        auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
        gert::Tensor tensor(
            {{const_value.size()}, {const_value.size()}}, // shape
            {ge::FORMAT_ND, ge::FORMAT_ND, {}},           // format
            gert::kFollowing,                             // placement
            const_dtype,                                  // dt
            nullptr);
        std::cout << " const_value size = :" << const_value.size() << std::endl;
        std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
        auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
        std::memcpy(tensor_data, &const_value[0], sizeof(T) * const_value.size());
        input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
        return input_tensor;
    }
};

TEST_F(UnfoldGradInferShapeTest, unfold_grad_infershape_case0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UnfoldGrad"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnfoldGrad")->infer_shape;

    vector<int64_t> shape = {1, 3, 3, 658, 658};
    auto inputTensorHolder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + shape.size() * sizeof(int64_t)]);
    auto shapeTensor = ConstructInputConstTensor<int64_t>(inputTensorHolder, shape, ge::DT_INT64);

    gert::StorageShape gradOutShape = {{1, 3, 3, 320, 658, 20}, {1, 3, 3, 320, 658, 20}};
    gert::StorageShape outputShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnfoldGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradOutShape, shapeTensor})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"dim", ge::AnyValue::CreateFrom<int64_t>(3)},
                           {"size", ge::AnyValue::CreateFrom<int64_t>(20)},
                           {"step", ge::AnyValue::CreateFrom<int64_t>(2)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedOutputShape = {1, 3, 3, 658, 658};
    auto gradInShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(ops::ToVector(*gradInShape), expectedOutputShape);
}

TEST_F(UnfoldGradInferShapeTest, unfold_grad_inferdtype_bf16_case0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UnfoldGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("UnfoldGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType gradOut = ge::DT_BF16;
        ge::DataType inputSize = ge::DT_INT64;
        ge::DataType output = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(2, 1)
                                  .IrInstanceNum({1, 1})
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&gradOut, &inputSize})
                                  .OutputDataTypes({&output})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), gradOut);
        EXPECT_EQ(context->GetInputDataType(1), inputSize);
        EXPECT_EQ(context->GetOutputDataType(0), output);
    }
}