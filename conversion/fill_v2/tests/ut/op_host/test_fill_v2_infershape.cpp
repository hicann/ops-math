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

#include <cstdint>
#include <vector>

class FillV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "fill_v2 Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "fill_v2 Proto Test TearDown" << std::endl; }
};

TEST_F(FillV2Infershape, fill_v2_infershape_int32_dims_test)
{
    gert::StorageShape dimsShape = {{3}, {3}};
    gert::StorageShape yShape = {{6, 7, 2}, {6, 7, 2}};

    std::vector<int32_t> dims_values = {6, 7, 2};
    gert::InfershapeContextPara::TensorDescription dims(dimsShape, ge::DT_INT32, ge::FORMAT_ND, true,
                                                        dims_values.data());
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "FillV2", {dims}, {y},
        {gert::InfershapeContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.5))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 7, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FillV2Infershape, fill_v2_infershape_non_const_dims_test)
{
    gert::StorageShape dimsShape = {{3}, {3}};
    gert::StorageShape yShape = {{6, 7, 2}, {6, 7, 2}};

    gert::InfershapeContextPara::TensorDescription dims(dimsShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "FillV2", {dims}, {y},
        {gert::InfershapeContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.5))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FillV2Infershape, fill_v2_infershape_scalar_dims_test)
{
    gert::StorageShape dimsShape = {{}, {}};
    gert::StorageShape yShape = {{}, {}};

    gert::InfershapeContextPara::TensorDescription dims(dimsShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "FillV2", {dims}, {y},
        {gert::InfershapeContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.5))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FillV2Infershape, fill_v2_infershape_null_dims_test)
{
    gert::StorageShape dimsShape = {{1}, {1}};
    gert::StorageShape yShape = {{6}, {6}};

    gert::InfershapeContextPara::TensorDescription dims(dimsShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "FillV2", {dims}, {y},
        {gert::InfershapeContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.5))}, {}, {}, {0});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FillV2Infershape, fill_v2_infershape_invalid_dtype_test)
{
    gert::StorageShape dimsShape = {{3}, {3}};
    gert::StorageShape yShape = {{6, 7, 2}, {6, 7, 2}};

    std::vector<float> dims_values = {6.0f, 7.0f, 2.0f};
    gert::InfershapeContextPara::TensorDescription dims(dimsShape, ge::DT_FLOAT, ge::FORMAT_ND, true,
                                                        dims_values.data());
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara(
        "FillV2", {dims}, {y},
        {gert::InfershapeContextPara::OpAttr("value", Ops::Math::AnyValue::CreateFrom<float>(1.5))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 7, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
