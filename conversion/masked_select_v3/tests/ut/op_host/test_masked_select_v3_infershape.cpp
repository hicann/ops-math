/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MaskedSelectv3InferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaskedSelectv3 Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaskedSelectv3 Proto Test TearDown" << std::endl; }
};

TEST_F(MaskedSelectv3InferTest, infershape_1d_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MaskedSelect", {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MaskedSelectv3InferTest, infershape_4d_fp32)
{
    gert::InfershapeContextPara infershapeContextPara("MaskedSelect",
                                                      {{{{2, 4, 6, 8}, {2, 4, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2, 4, 6, 8}, {2, 4, 6, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MaskedSelectv3InferTest, infershape_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MaskedSelect", {{{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND}, {{{16}, {16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
