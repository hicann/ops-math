/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "fusion_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class AddLora : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AddLora SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AddLora TearDown" << std::endl;
  }
};

TEST_F(AddLora, AddLora_infershape_case_0) {
  ge::op::AddLora op;
  op.UpdateInputDesc("y", create_desc({4,256}, ge::DT_FLOAT16));

  EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
  EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
}