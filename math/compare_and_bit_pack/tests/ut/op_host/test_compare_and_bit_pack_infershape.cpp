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

namespace {

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

void RunCompareAndBitpackInferShapeCase(const std::vector<int64_t>& xShape, const std::vector<int64_t>& thresholdShape,
                                        ge::graphStatus expectedStatus, const std::vector<int64_t>& expectedShape = {})
{
    gert::InfershapeContextPara contextPara("CompareAndBitpack",
                                            {
                                                {MakeStorageShape(xShape), ge::DT_FLOAT, ge::FORMAT_ND},
                                                {MakeStorageShape(thresholdShape), ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                            });

    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ExecuteTestCase(contextPara, expectedStatus, {expectedShape});
        return;
    }
    ExecuteTestCase(contextPara, expectedStatus);
}

} // namespace

TEST(CompareAndBitpackInferShape, UpdatesLastDimension)
{
    RunCompareAndBitpackInferShapeCase({2, 16}, {}, ge::GRAPH_SUCCESS, {2, 2});
}

TEST(CompareAndBitpackInferShape, PreservesUnknownLastDimension)
{
    RunCompareAndBitpackInferShapeCase({4, -1}, {}, ge::GRAPH_SUCCESS, {4, -1});
}

TEST(CompareAndBitpackInferShape, PreservesOtherUnknownDimensions)
{
    RunCompareAndBitpackInferShapeCase({-1, 24}, {}, ge::GRAPH_SUCCESS, {-1, 3});
}

TEST(CompareAndBitpackInferShape, RejectsScalarInput) { RunCompareAndBitpackInferShapeCase({}, {}, ge::GRAPH_FAILED); }

TEST(CompareAndBitpackInferShape, RejectsNonScalarThreshold)
{
    RunCompareAndBitpackInferShapeCase({16}, {1}, ge::GRAPH_FAILED);
}

TEST(CompareAndBitpackInferShape, AcceptsUnknownRankThresholdLikeRt1)
{
    RunCompareAndBitpackInferShapeCase({16}, {-2}, ge::GRAPH_SUCCESS, {2});
}

TEST(CompareAndBitpackInferShape, RejectsNonMultipleOfEight)
{
    RunCompareAndBitpackInferShapeCase({2, 15}, {}, ge::GRAPH_FAILED);
}

TEST(CompareAndBitpackInferShape, RejectsUnknownRankInputLikeRt1)
{
    RunCompareAndBitpackInferShapeCase({-2}, {}, ge::GRAPH_FAILED);
}
