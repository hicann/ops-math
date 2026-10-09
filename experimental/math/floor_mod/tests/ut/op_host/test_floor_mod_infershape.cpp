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
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

TEST(FloorModInferShapeTest, infers_tensor_scalar_tensor_broadcast_shape)
{
    gert::InfershapeContextPara para("FloorMod",
                                     {
                                         {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expected = {{2, 3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(FloorModInferShapeTest, infers_broadcast_shape)
{
    gert::InfershapeContextPara para("FloorMod",
                                     {
                                         {{{2, 1, 4}, {2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1, 3, 1}, {1, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expected = {{2, 3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(FloorModInferShapeTest, rejects_non_broadcastable_shapes)
{
    gert::InfershapeContextPara para("FloorMod",
                                     {
                                         {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
