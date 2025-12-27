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
#include "../../../../op_host/arch35/fills_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace std;

class FillsTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FillsTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FillsTilingTest TearDown" << std::endl;
  }
};

TEST_F(FillsTilingTest, fills_test_tiling_fp32_001)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Fills",
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
        },
        &compileInfo);

    uint64_t expectTilingKey = 7;
    string expectTilingData = "4096 4 32768 1024 4 1 1 1024 1024 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, expectTilingKey, true, expectTilingData, expectWorkspaces);
}
