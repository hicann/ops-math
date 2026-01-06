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
#include "base/registry/op_impl_space_registry_v2.h"

class PadV3InfershapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PadV3InfershapeTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PadV3InfershapeTest TearDown" << std::endl;
  }
};

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_0)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {
          {xShape, ge::DT_INT32, ge::FORMAT_ND}, 
          {padShape, ge::DT_INT32, ge::FORMAT_ND}, 
          {padShape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {{{{-2},{-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
} 