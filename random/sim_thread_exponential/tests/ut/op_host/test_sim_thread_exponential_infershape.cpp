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
 * \file test_sim_random_uniform_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SimThreadExponentialInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SimThreadExponential Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SimThreadExponential Proto Test TearDown" << std::endl;
    }
};

TEST_F(SimThreadExponentialInferShapeTest, sim_exponential_test_success)
{   
    std::vector<int64_t> inputSizeValues = {2, 152064};
    gert::InfershapeContextPara infershapeContextPara("SimThreadExponential",
                                                      {{{{2, 152064}, {2, 152064}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                      {{{{2, 152064}, {2, 152064}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                      {gert::InfershapeContextPara::OpAttr("count", Ops::Math::AnyValue::CreateFrom<int64_t>(2*152064)),
                                                       gert::InfershapeContextPara::OpAttr("lambda", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
                                                       gert::InfershapeContextPara::OpAttr("seed", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
                                                       gert::InfershapeContextPara::OpAttr("offset", Ops::Math::AnyValue::CreateFrom<int64_t>(4))});
    
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 152064}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
