/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/arch35/stateless_random_choice_with_mask_simt_tiling.h"

using namespace std;
using namespace ge;

class StatelessRandomChoiceWithMaskTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomChoiceWithMask SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomChoiceWithMask TearDown" << std::endl;
  }
};

TEST_F(StatelessRandomChoiceWithMaskTiling, stateless_random_choice_with_mask_tiling_001)
{
    optiling::StatelessRandomChoiceWithMaskCompileInfo compileInfo = {64, 196608};
    gert::StorageShape x_shape = {{1}, {1}};
    gert::StorageShape count_shape = {{1}, {1}};
    gert::StorageShape seed_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{1, 1}, {1, 1}};
    gert::StorageShape mask_shape = {{1}, {1}};

    vector<int32_t> count_value = {20};
    vector<int64_t> seed_value = {12};
    vector<int64_t> offset_value = {22};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomChoiceWithMask", 
        {
            {x_shape, ge::DT_BOOL, ge::FORMAT_ND},
            {count_shape, ge::DT_INT32, ge::FORMAT_ND, true, count_value.data()},
            {seed_shape, ge::DT_INT64, ge::FORMAT_ND, true, seed_value.data()},
            {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}
        },
        {
            {y_shape, ge::DT_INT32, ge::FORMAT_ND},
            {mask_shape, ge::DT_BOOL, ge::FORMAT_ND}
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16787712};
}

TEST_F(StatelessRandomChoiceWithMaskTiling, stateless_random_choice_with_mask_tiling_002)
{
    optiling::StatelessRandomChoiceWithMaskCompileInfo compileInfo = {64, 196608};
    gert::StorageShape x_shape = {{66561}, {66561}};
    gert::StorageShape count_shape = {{1}, {1}};
    gert::StorageShape seed_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{1, 1}, {1, 1}};
    gert::StorageShape mask_shape = {{1}, {1}};

    vector<int32_t> count_value = {20};
    vector<int64_t> seed_value = {12};
    vector<int64_t> offset_value = {22};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandomChoiceWithMask", 
        {
            {x_shape, ge::DT_BOOL, ge::FORMAT_ND},
            {count_shape, ge::DT_INT32, ge::FORMAT_ND, true, count_value.data()},
            {seed_shape, ge::DT_INT64, ge::FORMAT_ND, true, seed_value.data()},
            {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}
        },
        {
            {y_shape, ge::DT_INT32, ge::FORMAT_ND},
            {mask_shape, ge::DT_BOOL, ge::FORMAT_ND}
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {17838080};
}