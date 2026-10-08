/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "any_value.h"

class confusion_matrix : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "confusion_matrix SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "confusion_matrix TearDown" << std::endl; }
};

TEST_F(confusion_matrix, confusion_matrix_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConfusionMatrix",
        {
            {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("num_classes",
                                                Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(5))),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(confusion_matrix, confusion_matrix_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConfusionMatrix",
        {
            {{{1000}, {1000}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1000}, {1000}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("num_classes",
                                                Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(10))),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(confusion_matrix, confusion_matrix_infershape_test3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConfusionMatrix",
        {
            {{{5000}, {5000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5000}, {5000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5000}, {5000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("num_classes",
                                                Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(50))),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {50, 50},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case 4: labels/predictions unknown rank (-2)，无法校验 1D 约束 -> 输出透传 {-2}
TEST_F(confusion_matrix, confusion_matrix_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConfusionMatrix",
        {
            gert::InfershapeContextPara::TensorDescription({{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND),
            gert::InfershapeContextPara::TensorDescription({{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND),
            gert::InfershapeContextPara::TensorDescription({{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::InfershapeContextPara::TensorDescription({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::InfershapeContextPara::OpAttr("num_classes",
                                                Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(5))),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case 5: labels/predictions unknown shape (-1，rank 已知)，输出仍由 num_classes 决定 -> {5, 5}
TEST_F(confusion_matrix, confusion_matrix_infershape_unknown_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConfusionMatrix",
        {
            gert::InfershapeContextPara::TensorDescription({{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND),
            gert::InfershapeContextPara::TensorDescription({{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND),
            gert::InfershapeContextPara::TensorDescription({{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::InfershapeContextPara::TensorDescription({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND),
        },
        {
            gert::InfershapeContextPara::OpAttr("num_classes",
                                                Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(5))),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
