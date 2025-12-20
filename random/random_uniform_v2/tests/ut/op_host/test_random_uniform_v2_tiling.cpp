/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/random_uniform_v2_tiling.h"
#include "../../../op_host/random_uniform_v2_tiling_arch35.h"

using namespace std;
using namespace ge;

class RandomUniformV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RandomUniformV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RandomUniformV2 TearDown" << std::endl;
  }
};

TEST_F(RandomUniformV2Tiling, random_uniform_v2_tiling_910_95_float_001)
{
    optiling::RandomUniformV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shape_shape = {{2}, {2}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);
    auto dtype = Ops::Math::AnyValue::CreateFrom<int64_t>(0);

    vector<int32_t> shape_value = {32, 512};
    vector<int64_t> offset_value = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomUniformV2", {{shape_shape, ge::DT_INT32, ge::FORMAT_ND, true, shape_value.data()}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
        {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dtype", dtype),
         gert::TilingContextPara::OpAttr("seed", seed),
         gert::TilingContextPara::OpAttr("seed2", seed2)},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "64 256 256 16384 10 5 16384 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
