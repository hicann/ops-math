/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/ceil_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class CeilTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CeilTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CeilTilingTest TearDown" << std::endl;
  }
};

TEST_F(CeilTilingTest, test_tiling_fp16_001) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_bf16_002) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 5;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}


TEST_F(CeilTilingTest, test_tiling_fp32_003) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 7;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_dtype_input_output_diff_005) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_shape_input_output_diff_007) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64},  {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_empty_tensor_008) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_unsupport_input_009) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(CeilTilingTest, test_tiling_failed_unsupport_output_010) {
    optiling::CeilCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Ceil",
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}