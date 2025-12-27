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
 * \file test_concat_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
class ConcatV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConcatV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ConcatV2Test TearDown" << std::endl;
    }
};

TEST_F(ConcatV2Test, infer_axis_type_test_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatV2",
        {
            {{{16, 4}, {16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 4}, {16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ConcatV2Test, infer_axis_type_test_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatV2",
        {
            {{{16, 4, 16}, {16, 4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 4, 16}, {16, 4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
