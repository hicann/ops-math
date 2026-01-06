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
#include "../../../../op_host/arch35/pad_v3_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV3TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PadV3TilingTest TearDown" << std::endl;
    }
};

TEST_F(PadV3TilingTest, pad_v3_tiling_test_001)
{
    PadV3CompileInfo compileInfo = {0,0,0,0,0,false,"constant",0,0,0,false,"ascend910_95"};

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};
    int pad_value[2] = {3, 4};

    gert::TilingContextPara tilingContextPara(
        "PadV3",
        {
            {xShape, ge::DT_INT32, ge::FORMAT_ND}, 
            {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value},
            {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}
        },
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
            gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))
        },
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    string expectTilingData = "3 0 0 2 3 4 0 0 0 0 0 9 5 7 0 0 0 0 0 12 4 1 0 0 0 0 0 35 7 1 0 0 0 0 0 3 2 3 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}