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
 * \file test_pow_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/pow_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class PowTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PowTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PowTilingTest TearDown" << std::endl;
    }
};

// Note: pow tiling uses TilingRegistry which requires specific platform configuration.
// The infershape tests pass correctly, validating the operator logic.
// Tiling test is disabled pending further investigation of TilingRegistry requirements.

TEST_F(PowTilingTest, test_tiling_float32)
{
    optiling::PowCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;
    compileInfo.isRegBase = false;
    compileInfo.vectorLength = 128;
    compileInfo.blockSize = 32;

    gert::TilingContextPara tilingContextPara(
        "Pow",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 300000001000100;
    string expectTilingData =
        "1 640 13 512 1 1 0 13 640 8192 0 0 0 0 0 0 0 8192 0 0 0 0 0 0 0 8192 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 "
        "0 0 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
