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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/strided_slice_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

class StridedSliceGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceGradTiling TearDown" << std::endl;
    }
};

TEST_F(StridedSliceGradTiling, StridedSliceGrad_test_tiling_001)
{
    optiling::StridedSliceGradCompileInfo compileInfo = {64, 253952};
    vector<int32_t> shape = {1, 128};
    vector<int32_t> begin = {0, 0};
    vector<int32_t> end = {1, 46};
    vector<int32_t> strides = {1, 1};
    vector<int32_t> dy = {-1, -1};

    gert::TilingContextPara tilingContextPara(
        "StridedSliceGrad",
        {
            {{{1, 128}, {1, 128}}, ge::DT_INT32, ge::FORMAT_ND, true, shape.data()},
            {{{0, 0}, {0, 0}}, ge::DT_INT32, ge::FORMAT_ND, true, begin.data()},
            {{{1, 46}, {1, 46}}, ge::DT_INT32, ge::FORMAT_ND, true, end.data()},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, strides.data()},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND, true, dy.data()},
        },
        {
            {{{1, 128}, {1, 128}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::TilingContextPara::OpAttr("new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::TilingContextPara::OpAttr("shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 104;
    string expectTilingData =
        "1 0 0 94343321858568 140433105847881 0 0 0 1 94343321858568 64 106496 94343321868256 4 104 94343286440168 0 0 "
        "0 0 0 0 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}