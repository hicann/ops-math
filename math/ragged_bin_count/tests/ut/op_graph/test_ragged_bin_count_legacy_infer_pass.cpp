/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_graph/ragged_bin_count_proto.h"

namespace {
ge::TensorDesc MakeDesc(const std::vector<int64_t>& dims, ge::DataType dtype)
{
    ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, dtype);
    desc.SetOriginShape(ge::Shape(dims));
    desc.SetOriginFormat(ge::FORMAT_ND);
    return desc;
}

class SizeConstant : public ge::Operator {
public:
    explicit SizeConstant(const ge::Tensor& value) : ge::Operator("size_constant", "Const")
    {
        OutputRegister("y");
        UpdateOutputDesc("y", value.GetTensorDesc());
        SetAttr("value", value);
    }
};

ge::op::RaggedBinCount MakeOp(const std::vector<int64_t>& values, const std::vector<int64_t>& weights,
                              ge::DataType indexType = ge::DT_INT32, const std::vector<int64_t>& splits = {2})
{
    ge::op::RaggedBinCount op("ragged_bin_count_legacy_infer");
    EXPECT_EQ(op.update_input_desc_splits(MakeDesc(splits, ge::DT_INT64)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.update_input_desc_values(MakeDesc(values, indexType)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.update_input_desc_size(MakeDesc({1}, indexType)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.update_input_desc_weights(MakeDesc(weights, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    op.update_output_desc_output(ge::TensorDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_FLOAT));
    EXPECT_EQ(op.get_input_desc_splits().GetShape().GetDims(), splits);
    EXPECT_EQ(op.get_input_desc_values().GetShape().GetDims(), values);
    EXPECT_EQ(op.get_input_desc_size().GetShape().GetDims(), (std::vector<int64_t>{1}));
    EXPECT_EQ(op.get_input_desc_weights().GetShape().GetDims(), weights);
    return op;
}

void CheckPreservedShapes(const std::vector<int64_t>& values, const std::vector<int64_t>& weights)
{
    auto op = MakeOp(values, weights);
    ASSERT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_values().GetShape().GetDims(), values);
    EXPECT_EQ(op.get_input_desc_weights().GetShape().GetDims(), weights);
    EXPECT_EQ(op.get_output_desc_output().GetShape().GetDims(), (std::vector<int64_t>{1, -1}));
    EXPECT_EQ(op.get_output_desc_output().GetDataType(), ge::DT_FLOAT);
}

template <typename T>
void CheckConstantSize(ge::DataType indexType, T bins, ge::graphStatus expected)
{
    auto op = MakeOp({6}, {2, 3}, indexType);
    const ge::TensorDesc desc = MakeDesc({1}, indexType);
    const ge::Tensor tensor(desc, reinterpret_cast<const uint8_t*>(&bins), sizeof(bins));
    SizeConstant constant(tensor);
    op.set_input_size(constant, "y");
    ASSERT_EQ(op.InferShapeAndType(), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(op.get_output_desc_output().GetShape().GetDims(),
                  (std::vector<int64_t>{1, static_cast<int64_t>(bins)}));
        EXPECT_EQ(op.get_input_desc_weights().GetShape().GetDims(), (std::vector<int64_t>{2, 3}));
    }
}
} // namespace

TEST(RaggedBinCountLegacyInferTest, PreservesTwoDimensionalWeightsForVectorValues)
{
    CheckPreservedShapes({6}, {2, 3});
}

TEST(RaggedBinCountLegacyInferTest, PreservesVectorWeightsForTwoDimensionalValues)
{
    CheckPreservedShapes({2, 3}, {6});
}

TEST(RaggedBinCountLegacyInferTest, PreservesDifferentTwoDimensionalShapes) { CheckPreservedShapes({2, 3}, {3, 2}); }

TEST(RaggedBinCountLegacyInferTest, PreservesEmptyWeights)
{
    for (const auto& weights : std::vector<std::vector<int64_t>>{{0}, {0, 3}, {2, 0}}) {
        CheckPreservedShapes({2, 3}, weights);
    }
}

TEST(RaggedBinCountLegacyInferTest, PreservesScalarWeights) { CheckPreservedShapes({1}, {}); }

TEST(RaggedBinCountLegacyInferTest, ReadsInt32ConstantSize)
{
    CheckConstantSize<int32_t>(ge::DT_INT32, 7, ge::GRAPH_SUCCESS);
}

TEST(RaggedBinCountLegacyInferTest, ReadsInt64ConstantSizeWithoutTruncation)
{
    CheckConstantSize<int64_t>(ge::DT_INT64, (int64_t{1} << 32) + 7, ge::GRAPH_SUCCESS);
}

TEST(RaggedBinCountLegacyInferTest, AcceptsZeroBins) { CheckConstantSize<int64_t>(ge::DT_INT64, 0, ge::GRAPH_SUCCESS); }

TEST(RaggedBinCountLegacyInferTest, RejectsNegativeBins)
{
    CheckConstantSize<int32_t>(ge::DT_INT32, -1, ge::GRAPH_FAILED);
    CheckConstantSize<int64_t>(ge::DT_INT64, -1, ge::GRAPH_FAILED);
}

TEST(RaggedBinCountLegacyInferTest, RetainsKnownRowsWhenSizeIsUnknown)
{
    auto op = MakeOp({6}, {2, 3}, ge::DT_INT32, {4});
    ASSERT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_output_desc_output().GetShape().GetDims(), (std::vector<int64_t>{3, -1}));
    std::vector<std::pair<int64_t, int64_t>> ranges;
    ASSERT_EQ(op.get_output_desc_output().GetShapeRange(ranges), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ranges, (std::vector<std::pair<int64_t, int64_t>>{{3, 3}, {0, -1}}));
}

TEST(RaggedBinCountLegacyInferTest, PreservesUnknownInputDimensions)
{
    auto op = MakeOp({-1}, {-1, 3}, ge::DT_INT32, {-1});
    ASSERT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_weights().GetShape().GetDims(), (std::vector<int64_t>{-1, 3}));
    EXPECT_EQ(op.get_output_desc_output().GetShape().GetDims(), (std::vector<int64_t>{-1, -1}));
}

TEST(RaggedBinCountLegacyInferTest, PreservesUnknownInputRanks)
{
    auto op = MakeOp({-2}, {-2}, ge::DT_INT32, {-2});
    ASSERT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_weights().GetShape().GetDims(), (std::vector<int64_t>{-2}));
    EXPECT_EQ(op.get_output_desc_output().GetShape().GetDims(), (std::vector<int64_t>{-1, -1}));
}

TEST(RaggedBinCountLegacyInferTest, PreservesLegacyOutputDataTypes)
{
    for (const auto dtype : {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_DOUBLE}) {
        auto op = MakeOp({6}, {2, 3});
        ASSERT_EQ(op.update_input_desc_weights(MakeDesc({2, 3}, dtype)), ge::GRAPH_SUCCESS);
        ASSERT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
        EXPECT_EQ(op.get_output_desc_output().GetDataType(), dtype);
    }
}

TEST(RaggedBinCountLegacyInferTest, RejectsInvalidSplitsAndSizeShapes)
{
    for (const auto& splits : std::vector<std::vector<int64_t>>{{}, {1}, {2, 1}}) {
        auto op = MakeOp({6}, {2, 3}, ge::DT_INT32, splits);
        EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED);
    }
    auto op = MakeOp({6}, {2, 3});
    op.update_input_desc_size(ge::TensorDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32));
    EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_FAILED);
    for (const int64_t sizeElements : {int64_t{1} << 31, int64_t{1} << 32}) {
        auto oversizedSizeOp = MakeOp({6}, {2, 3});
        ASSERT_EQ(oversizedSizeOp.update_input_desc_size(MakeDesc({sizeElements}, ge::DT_INT32)), ge::GRAPH_SUCCESS);
        EXPECT_EQ(oversizedSizeOp.InferShapeAndType(), ge::GRAPH_FAILED);
    }
}
