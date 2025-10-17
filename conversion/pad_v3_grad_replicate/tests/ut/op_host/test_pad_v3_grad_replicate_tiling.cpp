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
#include "../../../op_host/pad_v3_grad_replicate_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class PadV3GradReplicateTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3GradReplicateTiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "PadV3GradReplicateTiling  TearDown" << std::endl;
    }
};

TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_test_success)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 0, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 64, 62}, {1, 1, 64, 62}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2101;
    std::string expectTilingData =
        "4294967297 274877907008 274877907008 266287972416 274877907008 0 4294967297 245191092994112 1 137438955573 "
        "128 ";
    std::vector<size_t> expectWorkspaces = {16785408};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PadV3GradReplicateTiling, pad_v3_grad_replicate_tiling_test_falied)
{
    optiling::Tiling4PadV3GradReplicateCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 100, 100, 100, 100};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplicate",
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 64, 62}, {1, 1, 64, 62}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("edge")),
         gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
