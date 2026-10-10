/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "op_infer_shape_range_context_builder.h"

namespace {
ge::graphStatus InferShapeRange(int64_t numPartitions, gert::ContextHolder<gert::InferShapeRangeContext>& holder)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (spaceRegistry == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto opImpl = spaceRegistry->GetOpImpl("DynamicPartition");
    if (opImpl == nullptr || opImpl->infer_shape_range == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::StorageShape xShape{{4, 8}, {4, 8}};
    gert::StorageShape partitionsShape{{4}, {4}};
    gert::StorageFormat format{ge::FORMAT_ND, ge::FORMAT_ND, {}};
    gert::Tensor xMin(xShape, format, ge::DT_FLOAT);
    gert::Tensor xMax(xShape, format, ge::DT_FLOAT);
    gert::Tensor partitionsMin(partitionsShape, format, ge::DT_INT32);
    gert::Tensor partitionsMax(partitionsShape, format, ge::DT_INT32);
    gert::Range<gert::Tensor> xRange(&xMin, &xMax);
    gert::Range<gert::Tensor> partitionsRange(&partitionsMin, &partitionsMax);

    gert::OpInferShapeRangeContextBuilder builder;
    holder = builder.OpType("DynamicPartition")
                 .OpName("DynamicPartition")
                 .IONum(2, 1)
                 .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                 .AppendAttr(numPartitions)
                 .InputTensorsRange({&xRange, &partitionsRange})
                 .Build();
    auto* context = holder.GetContext();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return opImpl->infer_shape_range(context);
}
} // namespace

TEST(DynamicPartitionInferShapeRange, AcceptsOnePartition)
{
    gert::ContextHolder<gert::InferShapeRangeContext> holder;
    EXPECT_EQ(InferShapeRange(1, holder), ge::GRAPH_SUCCESS);
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_NE(holder.GetContext()->GetOutputShapeRange(0), nullptr);
    EXPECT_EQ(*holder.GetContext()->GetOutputShapeRange(0)->GetMin(), gert::Shape({0, 8}));
    EXPECT_EQ(*holder.GetContext()->GetOutputShapeRange(0)->GetMax(), gert::Shape({4, 8}));
}

TEST(DynamicPartitionInferShapeRange, RejectsZeroPartitions)
{
    gert::ContextHolder<gert::InferShapeRangeContext> holder;
    EXPECT_EQ(InferShapeRange(0, holder), ge::GRAPH_FAILED);
}

TEST(DynamicPartitionInferShapeRange, RejectsNegativePartitions)
{
    gert::ContextHolder<gert::InferShapeRangeContext> holder;
    EXPECT_EQ(InferShapeRange(-1, holder), ge::GRAPH_FAILED);
}
