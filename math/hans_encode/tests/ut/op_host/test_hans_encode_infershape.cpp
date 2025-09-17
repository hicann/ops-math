/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "comp_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class HansEncode : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HansEncode SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HansEncode TearDown" << std::endl;
  }
};

TEST_F(HansEncode, HansEncode_infershape_case_0) {
  ge::op::HansEncode op;
  op.UpdateInputDesc("input_tensor", create_desc({4096, 1, 512}, ge::DT_FLOAT16));
  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(HansEncode, HansEncode_infer_dtype) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("HansEncode"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("HansEncode")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_tensor = ge::DT_FLOAT;
    ge::DataType pdf = ge::DT_INT32;
    ge::DataType mantissa = ge::DT_FLOAT;
    ge::DataType fixed = ge::DT_FLOAT;
    ge::DataType var = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
        .IrInputNum(2)
        .NodeIoNum(2,4)
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"statistic", ge::AnyValue::CreateFrom<bool>(false)},
            {"reshuff", ge::AnyValue::CreateFrom<bool>(false)}
        })
        .InputDataTypes({&input_tensor, &pdf})
        .OutputDataTypes({&pdf, &mantissa, &fixed, &var})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(1), mantissa);
    EXPECT_EQ(context->GetOutputDataType(2), fixed);
    EXPECT_EQ(context->GetOutputDataType(3), var);
  }
}