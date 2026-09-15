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
 * \file test_stateless_drop_out_gen_mask_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class stateless_drop_out_gen_mask : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessDropOutGenMask SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessDropOutGenMask TearDown" << std::endl; }
};

TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test1)
{
    vector<int64_t> shapeValue = {32, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2048}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test2)
{
    vector<int32_t> shapeValue = {8, 512, 10};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT32,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT32,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5120}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// shape 输入 dtype 非法（DT_FLOAT），期望 infershape 返回 GRAPH_FAILED
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test3_unsupported_dtype)
{
    vector<float> shapeValue = {4.0, 4.0};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// shape 输入含负值维度，期望 infershape 返回 GRAPH_FAILED
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test4_negative_dim)
{
    vector<int64_t> shapeValue = {4, -2};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// shape 输入张量不可用（GetInputTensor 返回 nullptr，动态图/泛化场景），
// 期望输出 1-D 未知 (-1,)，实际输出大小由运行时 tiling 决定
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test5_null_shape_tensor)
{
    gert::InfershapeContextPara infershapeContextPara("StatelessDropOutGenMask",
                                                      {
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_FLOAT,
                                                              ge::FORMAT_ND,
                                                          },
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_INT64,
                                                              ge::FORMAT_ND,
                                                          },
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_INT64,
                                                              ge::FORMAT_ND,
                                                          },
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_INT64,
                                                              ge::FORMAT_ND,
                                                          },
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{},
                                                      std::vector<uint32_t>{}, std::vector<uint32_t>{},
                                                      std::vector<size_t>{0});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// shape 输入为非 const 张量（数据编译期不可用，动态图/泛化场景），
// 期望输出 1-D 未知 (-1,)，实际输出大小由运行时 tiling 决定
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test6_non_const_shape)
{
    gert::InfershapeContextPara infershapeContextPara("StatelessDropOutGenMask",
                                                      {
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, false, nullptr},
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_FLOAT,
                                                              ge::FORMAT_ND,
                                                          },
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_INT64,
                                                              ge::FORMAT_ND,
                                                          },
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_INT64,
                                                              ge::FORMAT_ND,
                                                          },
                                                          {
                                                              {{1}, {1}},
                                                              ge::DT_INT64,
                                                              ge::FORMAT_ND,
                                                          },
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// shape 输入 desc 含 -1（unknownshape）且 const 数据可用，
// 期望输出 1-D 未知 (-1,)，由运行时 tiling 决定（修复前会静默输出 (16,)）
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test7_desc_unknown_dim)
{
    vector<int64_t> shapeValue = {32, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// shape 输入 desc 为 -2（unknownrank）且 const 数据可用，
// 期望输出 1-D 未知 (-1,)，由运行时 tiling 决定（修复前会静默输出 (16,)）
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test8_desc_unknown_rank)
{
    vector<int64_t> shapeValue = {32, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// shape 输入 desc 为多个 -1（GetShapeSize 乘积为正 1 的陷阱场景）且 const 数据可用，
// 期望输出 1-D 未知 (-1,)，由运行时 tiling 决定（修复前只读 1 个维度算错）
TEST_F(stateless_drop_out_gen_mask, stateless_drop_out_gen_mask_infershape_test9_desc_multi_unknown_dim)
{
    vector<int64_t> shapeValue = {32, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessDropOutGenMask",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
