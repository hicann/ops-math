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
 * \file test_concat_infershpe.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ConcatTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(ConcatTest, concat_d_infer_shape_fp16)
{
    int64_t concatDim = -1;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_fp16_n1)
{
    int64_t concatDim = 1;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_fp16_shape)
{
    int64_t concatDim = 3;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 24}, {2, 100, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_fp16_errorshape)
{
    int64_t concatDim = 1;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 24}, {2, 100, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_fp16_errordim)
{
    int64_t concatDim = 5;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 24}, {2, 100, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_fp16_errorshapdim)
{
    int64_t concatDim = -1;
    gert::InfershapeContextPara infershapeContextPara(
        "Concat",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
            {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 2, 4}, {2, 100, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_fp16_scalar)
{
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_no_shape_range_fp16)
{
    int64_t concatDim = -1;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_no_shape_range_fp1612)
{
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                          {{{
                                                                -2,
                                                            },
                                                            {
                                                                -2,
                                                            }},
                                                           ge::DT_FLOAT16,
                                                           ge::FORMAT_ND},
                                                          {{{
                                                                -2,
                                                            },
                                                            {
                                                                -2,
                                                            }},
                                                           ge::DT_FLOAT16,
                                                           ge::FORMAT_ND},
                                                          {{{
                                                                -2,
                                                            },
                                                            {
                                                                -2,
                                                            }},
                                                           ge::DT_FLOAT16,
                                                           ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_no_shape_range_mix_fp16)
{
    int64_t concatDim = -1;
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &concatDim},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_multi_inputs)
{
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 6}, {2, 3, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, {3}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatTest, concat_d_infer_shape_dim_value_unavailable)
{
    // concat_dim 为数据依赖输入，编译期取值不可得（data-feed 场景），秩已知时输出应保秩、全维置 -1
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)}}, {1, 2},
                                                      {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_dim_value_unavailable_all_unknown_rank)
{
    // concat_dim 编译期不可得且所有输入均为未知秩时，输出应为未知秩
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)}}, {1, 2},
                                                      {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatTest, concat_d_infer_shape_dim_value_unavailable_rank_mismatch)
{
    // concat_dim 编译期不可得且输入秩不一致时，秩校验与轴无关，应在编译期直接报错
    gert::InfershapeContextPara infershapeContextPara("Concat",
                                                      {
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 100}, {2, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {{"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)}}, {1, 2},
                                                      {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
