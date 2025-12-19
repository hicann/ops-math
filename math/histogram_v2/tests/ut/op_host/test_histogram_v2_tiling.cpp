/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../op_host/histogram_v2_tiling.h"

using namespace ge;
using namespace std;
class HistogramV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HistogramV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HistogramV2Tiling TearDown" << std::endl;
    }
};

struct HistogramV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatform = 0;
    int64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND310P;
};

TEST_F(HistogramV2Tiling, ascend910B1_test_tiling__001)
{
    HistogramV2CompileInfo compileInfo = {64, 262144, 16 * 1024 * 1024};
    gert::TilingContextPara tilingContextPara(
        "HistogramV2",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("bins", Ops::Math::AnyValue::CreateFrom<int64_t>(100)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "100 16320 1 4 8 4 8 0 16000 4 8 0 16000 4 8 ";
    std::vector<size_t> expectWorkspaces = {16806912};
    ExecuteTestCase(tilingContextPara, 0, expectTilingKey, expectTilingData, expectWorkspaces);
}
