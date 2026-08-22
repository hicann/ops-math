/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class RaggedBinCountInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RaggedBinCountInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RaggedBinCountInfershapeTest TearDown" << std::endl; }
};

TEST_F(RaggedBinCountInfershapeTest, test_int32_static_shape)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_int64_vector_size_2d_values_and_empty_weights)
{
    int64_t sizeData[1] = {7};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 7}});
}

TEST_F(RaggedBinCountInfershapeTest, test_scalar_size_is_rejected)
{
    int64_t sizeData[1] = {7};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_dynamic_splits)
{
    int32_t sizeData[1] = {8};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 8}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_size)
{
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, -1}});
}

TEST_F(RaggedBinCountInfershapeTest, test_splits_too_short)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_length_mismatch)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_same_numel_but_different_shape_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_splits_scalar_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_splits_rank_two_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{1, 3}, {1, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_values_scalar_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_values_rank_three_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1, 2, 3}, {1, 2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_size_length_two_is_rejected)
{
    int32_t sizeData[2] = {5, 6};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_scalar_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_weights_rank_three_is_rejected)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{1, 1, 2}, {1, 1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(RaggedBinCountInfershapeTest, test_negative_size)
{
    int32_t sizeData[1] = {-1};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

// -2 is the UNKNOWN_RANK marker. def.cpp declares DynamicRankSupportFlag(true), so every
// IsUnknownRank branch in the infershape must be exercised rather than assumed (red line R4).
TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_splits_yields_unknown_row_dim)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    // The row count cannot be resolved from an unknown-rank splits, but the bin count still can.
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_values_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_weights_is_accepted)
{
    int32_t sizeData[1] = {5};
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeData},
                                            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 5}});
}

TEST_F(RaggedBinCountInfershapeTest, test_unknown_rank_size_yields_unknown_bin_dim)
{
    gert::InfershapeContextPara context("RaggedBinCount",
                                        {
                                            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    // No const size tensor is available, so the bin dimension stays unknown.
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, -1}});
}
