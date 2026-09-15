/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "../../../op_host/hans_encode_tiling.h"
#include "hans_950_tiling_test_utils.h"

namespace {
using namespace Hans950Test;

gert::TilingContextPara EncodeParameters(ge::DataType dtype = ge::DT_FLOAT, int64_t count = 65536,
                                         bool statistic = false, bool reshuff = false)
{
    // The context builder requires non-null compile info, even for an empty type.
    static optiling::HansEncodeCompileInfo compileInfo;
    const int64_t bytes = ge::GetSizeByDataType(dtype);
    return {"HansEncode",
            {Tensor(count, dtype), Tensor(256, ge::DT_INT32)},
            {Tensor(256, ge::DT_INT32), Tensor(count * (bytes - 1) / bytes, dtype), Tensor(count, dtype),
             Tensor(count, dtype)},
            {BoolAttr("statistic", statistic), BoolAttr("reshuff", reshuff)},
            &compileInfo,
            64,
            192 * 1024};
}

class HansEncode950Modes : public testing::TestWithParam<std::tuple<ge::DataType, bool, bool>> {};

TEST_P(HansEncode950Modes, SerializesDtypeFlagsAndOutputCapacities)
{
    const auto [dtype, statistic, reshuff] = GetParam();
    const int64_t bytes = ge::GetSizeByDataType(dtype);
    const auto result = Hans950Test::Run(EncodeParameters(dtype, 65536, statistic, reshuff));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, bytes);
    EXPECT_EQ(result.blockDim, 2U);
    EXPECT_EQ(result.Int64(0), 2);
    EXPECT_EQ(result.Int64(1), 512);
    EXPECT_EQ(result.Int64(2), 512);
    // A reshuffled output uses the host upper bound as its staging capacity.
    const int64_t fixedBytes = reshuff ? 84992 : 65536 * bytes;
    EXPECT_EQ(result.Int64(3), (fixedBytes - 512) / 2);
    EXPECT_EQ(result.Int64(4), (fixedBytes - 512) / 2);
    EXPECT_EQ(result.Int64(5), 65536 * bytes);
    EXPECT_EQ(result.Bool(48), statistic);
    EXPECT_EQ(result.Bool(49), reshuff);
    const auto plain = Hans950Test::Run(EncodeParameters(dtype));
    ASSERT_EQ(plain.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.workspaces.size(), 1U);
    ASSERT_EQ(plain.workspaces.size(), 1U);
    EXPECT_EQ(result.workspaces[0], plain.workspaces[0] + (reshuff ? 84992U : 0U));
}

INSTANTIATE_TEST_SUITE_P(StorageAndAttributes, HansEncode950Modes,
                         testing::Combine(testing::Values(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT), testing::Bool(),
                                          testing::Bool()));

TEST(HansEncode950Tiling, MinimumInputUsesOneCore)
{
    const auto result = Hans950Test::Run(EncodeParameters(ge::DT_FLOAT, 32768));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.Int64(1), 512);
}

TEST(HansEncode950Tiling, HeaderLimitsLaunchTo56Cores)
{
    const auto result = Hans950Test::Run(EncodeParameters(ge::DT_FLOAT, 32768 * 64));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 56U);
    EXPECT_EQ(result.Int64(0), 56);
    EXPECT_EQ(result.Int64(1), 585);
    EXPECT_EQ(result.Int64(2), 593);
}

TEST(HansEncode950Tiling, LastCoreReceivesRemainderLoops)
{
    const auto result = Hans950Test::Run(EncodeParameters(ge::DT_FLOAT, 65536 + 64));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.Int64(1), 512);
    EXPECT_EQ(result.Int64(2), 513);
}

TEST(HansEncode950Tiling, RejectsInvalidInputSizes)
{
    for (const int64_t count : {int64_t{0}, int64_t{32704}, int64_t{32769}}) {
        SCOPED_TRACE(count);
        EXPECT_EQ(Hans950Test::Run(EncodeParameters(ge::DT_FLOAT, count)).status, ge::GRAPH_FAILED);
    }
}

TEST(HansEncode950Tiling, RejectsInvalidPdfOrMantissa)
{
    auto parameters = EncodeParameters();
    parameters.inputTensorDesc_[1] = Tensor(255, ge::DT_INT32);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters = EncodeParameters();
    parameters.outputTensorDesc_[1] = Tensor(49151, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, RejectsInsufficientOutputCapacity)
{
    auto parameters = EncodeParameters();
    parameters.outputTensorDesc_[2] = Tensor(127, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters.outputTensorDesc_[2] = Tensor(128, ge::DT_FLOAT);
    parameters.outputTensorDesc_[3] = Tensor(0, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters = EncodeParameters(ge::DT_FLOAT, 65536, false, true);
    parameters.outputTensorDesc_[2] = Tensor(21247, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, AcceptsExactReshuffCapacity)
{
    auto parameters = EncodeParameters(ge::DT_FLOAT, 65536, false, true);
    parameters.outputTensorDesc_[2] = Tensor(21248, ge::DT_FLOAT);
    parameters.outputTensorDesc_[3] = Tensor(0, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_SUCCESS);
}

TEST(HansEncode950Tiling, RejectsMissingCoresOrDcacheOnlyUb)
{
    auto parameters = EncodeParameters();
    parameters.coreNum_ = 0;
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters = EncodeParameters();
    parameters.ubSize_ = 32 * 1024;
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, LargeReshuffReservesEveryTileRecord)
{
    constexpr int64_t count = 56 * 129 * 4096;
    constexpr int64_t capacity = 30987776; // 512 + 56 * (528384 + 129 * 128 + 8448).
    for (const auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        SCOPED_TRACE(dtype);
        const int64_t bytes = ge::GetSizeByDataType(dtype);
        auto parameters = EncodeParameters(dtype, count, false, true);
        parameters.outputTensorDesc_[2] = Tensor(capacity / bytes, dtype);
        parameters.outputTensorDesc_[3] = Tensor(0, dtype);
        const auto result = Hans950Test::Run(parameters);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.blockDim, 56U);
        EXPECT_EQ(result.Int64(3), 553344);
        EXPECT_EQ(result.Int64(4), 553344);
        EXPECT_EQ(result.Int64(5), 0);
    }
}

TEST(HansEncode950Tiling, RejectsLegacyLargeCapacityWithoutVar)
{
    for (const bool reshuff : {false, true}) {
        auto parameters = EncodeParameters(ge::DT_FLOAT, 56 * 129 * 4096, false, reshuff);
        parameters.outputTensorDesc_[2] = Tensor(30525440 / 4, ge::DT_FLOAT);
        parameters.outputTensorDesc_[3] = Tensor(0, ge::DT_FLOAT);
        EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    }
}

TEST(HansEncode950Tiling, PartialTileReservesFullGroupMetadata)
{
    auto parameters = EncodeParameters(ge::DT_FLOAT, 65536 + 64, false, true);
    parameters.outputTensorDesc_[2] = Tensor(85376 / 4, ge::DT_FLOAT);
    parameters.outputTensorDesc_[3] = Tensor(0, ge::DT_FLOAT);
    const auto result = Hans950Test::Run(parameters);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.Int64(3), 42432);
    EXPECT_EQ(result.Int64(4), 42432);
    parameters.outputTensorDesc_[2] = Tensor(85376 / 4 - 1, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, EqualSlotsCoverTheLongestLastCore)
{
    for (const bool reshuff : {false, true}) {
        auto parameters = EncodeParameters(ge::DT_FLOAT, 32768 * 64, false, reshuff);
        parameters.outputTensorDesc_[2] = Tensor(2670592 / 4, ge::DT_FLOAT);
        parameters.outputTensorDesc_[3] = Tensor(0, ge::DT_FLOAT);
        const auto result = Hans950Test::Run(parameters);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.Int64(1), 585);
        EXPECT_EQ(result.Int64(2), 593);
        EXPECT_EQ(result.Int64(3), 47680);
        EXPECT_EQ(result.Int64(4), 47680);
    }
}

TEST(HansEncode950Tiling, VarOnlyRequiresOneBytePerInputValue)
{
    auto parameters = EncodeParameters();
    parameters.outputTensorDesc_[2] = Tensor(512 / 4, ge::DT_FLOAT);
    parameters.outputTensorDesc_[3] = Tensor(65536 / 4, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_SUCCESS);
    parameters.outputTensorDesc_[3] = Tensor(65536 / 4 - 1, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, MixedCapacityUsesWholeTilePrefixBudget)
{
    // Each slot guarantees one full tile: 8448 + 4096 + 128 bytes.
    // The shared var therefore reserves (32768 - 4096) * 2 bytes.
    auto parameters = EncodeParameters();
    parameters.outputTensorDesc_[2] = Tensor(25856 / 4, ge::DT_FLOAT);
    parameters.outputTensorDesc_[3] = Tensor(57344 / 4, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_SUCCESS);
    parameters.outputTensorDesc_[3] = Tensor(57344 / 4 - 1, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, Legacy910RetainsItsOriginalCapacityBudget)
{
    auto parameters = EncodeParameters(ge::DT_FLOAT, 65536, false, true);
    parameters.outputTensorDesc_[2] = Tensor(83968 / 4, ge::DT_FLOAT);
    parameters.outputTensorDesc_[3] = Tensor(0, ge::DT_FLOAT);
    const auto result = Hans950Test::Run(parameters, false);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.Int64(3), 41728);
    EXPECT_EQ(result.Int64(4), 41728);
    parameters.outputTensorDesc_[2] = Tensor(83968 / 4 - 1, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters, false).status, ge::GRAPH_FAILED);
}

TEST(HansEncode950Tiling, RejectsCapacitiesOutsideSignedHeaderFields)
{
    auto parameters = EncodeParameters(ge::DT_FLOAT, int64_t{2147483648} * 64);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters = EncodeParameters(ge::DT_FLOAT, int64_t{2147483648});
    parameters.coreNum_ = 1;
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

} // namespace
