
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_broadcast_to_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/broadcast_to_tiling_arch35.h"

using namespace std;
using namespace ge;

class BroadcastToTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BroadcastToTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BroadcastToTiling TearDown" << std::endl;
  }
};

TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_failed) {
    optiling::BroadcastToCompileInfo compileInfo = {128};
    gert::StorageShape shape = {{1, 1, 5}, {1, 1, 5}};
    gert::StorageShape shape1 = {{3}, {3}};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo",
        {{ shape, ge::DT_FLOAT16, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND }},
        {{ shape, ge::DT_FLOAT16, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}
