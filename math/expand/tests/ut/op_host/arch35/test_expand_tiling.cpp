
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
#include "../../../../op_host/arch35/expand_tiling_arch35.h"

using namespace std;
using namespace ge;

class ExpandTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ExpandTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ExpandTiling TearDown" << std::endl;
  }
};

// TEST_F(ExpandTilingTest, Expand_tiling_test_failed) {
//     optiling::ExpandCompileInfo compileInfo;
//     compileInfo.coreNum = 72;
//     compileInfo.ubSize = 262144;
//     compileInfo.clSize = 256;
//     compileInfo.vRegSize = 256;
//     compileInfo.blockSize = 32;

//     gert::StorageShape shape = {{1, 1, 1}, {1, 1, 1}};
//     gert::StorageShape shape1 = {{1, 1, 313, 199}, {1, 1, 313, 199}};
//     gert::TilingContextPara tilingContextPara(
//         "Expand",
//         {{ shape, ge::DT_UINT8, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND }},
//         {{ shape1, ge::DT_UINT8, ge::FORMAT_ND }},
//         &compileInfo);
//     uint64_t expectedTilingKey = 232000000;
//     std::vector<size_t> expectedWorkspaces = { 16777216 };
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
// }
