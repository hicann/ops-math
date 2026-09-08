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
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/cholesky_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

class CholeskyTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CholeskyTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CholeskyTiling TearDown" << std::endl; }
};

struct CholeskyCompileInfo {
    int32_t coreNum = 0;
};

TEST_F(CholeskyTiling, cholesky_test_tiling_case0)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{3, 6, 6}, {3, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 6, 6}, {3, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "6 3 4294967302 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_single_large_matrix)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "8192 1 137438953728 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_rank_nine)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_non_square_matrix)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_oversized_matrix)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{8193, 8193}, {8193, 8193}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8193, 8193}, {8193, 8193}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_unsupported_dtype)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_unsupported_format)
{
    CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
