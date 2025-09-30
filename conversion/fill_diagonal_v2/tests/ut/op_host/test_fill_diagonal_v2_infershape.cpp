/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file test_fill_diagonal_v2_proto.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "experiment_ops.h"
#include "common/utils/ut_op_common.h"

class fill_diagonal_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "fill_diagonal_v2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "fill_diagonal_v2 TearDown" << std::endl;
    }
};

TEST_F(fill_diagonal_v2_test, fill_diagonal_v2_infer_shape)
{
    ge::op::FillDiagonalV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}, {2, 64}};
    auto tensor_x = create_desc_shape_range({3, 7}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 7}, ge::FORMAT_ND, shape_range);

    op.UpdateInputDesc("x", tensor_x);

    op.SetAttr("wrap", false);

    Runtime2TestParam param{{"wrap"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("x");
    std::vector<int64_t> expected_output_shape = {3, 7};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fill_diagonal_v2_test, fill_diagonal_v2_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FillDiagonalV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FillDiagonalV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_val = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"wrap", ge::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_ref, &input_val})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}