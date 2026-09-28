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
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class QueueDataTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QueueData SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QueueData TearDown" << std::endl; }
};

TEST_F(QueueDataTest, queue_data_single_output_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("QueueData"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("QueueData")->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("QueueData")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&dummy_input})
                      .OutputShapes({&output_shape})
                      .Attr("index", int64_t(0))
                      .Attr("queue_name", ge::AscendString(""))
                      .Attr("output_types", std::vector<int64_t>{ge::DT_INT8})
                      .Attr("output_shapes", std::vector<std::vector<int64_t>>{{224, 224, 3}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    int64_t type_size = 1;
    int64_t data_len = type_size * 224 * 224 * 3;
    int64_t dims_size = sizeof(int64_t) * 3;
    int64_t expected = 64 + dims_size + data_len;
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), expected);
}

TEST_F(QueueDataTest, queue_data_types_shapes_length_mismatch_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("QueueData"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("QueueData")->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("QueueData")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&dummy_input})
                      .OutputShapes({&output_shape})
                      .Attr("index", int64_t(0))
                      .Attr("queue_name", ge::AscendString(""))
                      .Attr("output_types", std::vector<int64_t>{ge::DT_INT8})
                      .Attr("output_shapes", std::vector<std::vector<int64_t>>{{224, 224, 3}, {1}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(QueueDataTest, queue_data_unsupported_dtype_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("QueueData"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("QueueData")->infer_shape;
    gert::StorageShape dummy_input = {{1}, {1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("QueueData")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&dummy_input})
                      .OutputShapes({&output_shape})
                      .Attr("index", int64_t(0))
                      .Attr("queue_name", ge::AscendString(""))
                      .Attr("output_types", std::vector<int64_t>{ge::DT_STRING})
                      .Attr("output_shapes", std::vector<std::vector<int64_t>>{{1}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
}
