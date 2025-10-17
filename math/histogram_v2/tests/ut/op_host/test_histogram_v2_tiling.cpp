/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(HistogramV2Tiling, ascend910B1_test_tiling__001)
{
    HistogramV2CompileInfo compileInfo = {64, 262144, true};
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
    uint64_t expectTilingKey = 4294967295;
    string expectTilingData = "4294967360 68719476737 68719476752 2 1008981770 4096 0 4096 ";
    std::vector<size_t> expectWorkspaces = {281474942680800};
    ExecuteTestCase(tilingContextPara, 4294967295, expectTilingKey, expectTilingData, expectWorkspaces);
}
