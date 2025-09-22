/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "comp_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class HansDecode : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HansDecode SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HansDecode TearDown" << std::endl;
  }
};

TEST_F(HansDecode, HansDecode_infershape_case_0) {
  ge::op::HansDecode op;
  op.UpdateInputDesc("mantissa", create_desc({4096, 1, 512}, ge::DT_FLOAT16));
  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(HansDecode, HansDecode_infer_dtype) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("HansDecode"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("HansDecode")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType mantissa = ge::DT_FLOAT;
    ge::DataType fixed = ge::DT_FLOAT;
    ge::DataType var = ge::DT_FLOAT;
    ge::DataType pdf = ge::DT_INT32;
    ge::DataType output = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
        .IrInputNum(4)
        .NodeIoNum(4,1)
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"reshuff", ge::AnyValue::CreateFrom<bool>(false)}
        })
        .InputDataTypes({&mantissa, &fixed, &var, &pdf})
        .OutputDataTypes({&output})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), output);
  }
}