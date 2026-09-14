/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include "../../../op_host/hans_decode_tiling.h"
#include "../../../../hans_encode/tests/ut/op_host/hans_950_tiling_test_utils.h"

namespace {
using namespace Hans950Test;

gert::TilingContextPara DecodeParameters(ge::DataType dtype = ge::DT_FLOAT, bool reshuff = false)
{
    // The context builder requires non-null compile info, even for an empty type.
    static optiling::HansDecodeCompileInfo compileInfo;
    const int64_t bytes = ge::GetSizeByDataType(dtype);
    return {"HansDecode",
            {Tensor(65536 * (bytes - 1) / bytes, dtype), Tensor(32768, dtype), Tensor(16384, dtype),
             Tensor(256, ge::DT_INT32)},
            {Tensor(65536, dtype)},
            {BoolAttr("reshuff", reshuff)},
            &compileInfo,
            64,
            192 * 1024};
}

class HansDecode950Modes : public testing::TestWithParam<std::tuple<ge::DataType, bool>> {};

TEST_P(HansDecode950Modes, SerializesOutputCountAndDecodeBounds)
{
    const auto [dtype, reshuff] = GetParam();
    const int64_t bytes = ge::GetSizeByDataType(dtype);
    const auto result = Hans950Test::Run(DecodeParameters(dtype, reshuff));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, bytes);
    EXPECT_EQ(result.blockDim, 56U);
    EXPECT_EQ(result.Int64(0), 65536 * (bytes - 1));
    EXPECT_EQ(result.Int64(1), 32768 * bytes);
    // These legacy fields retain their existing byte-size convention.
    EXPECT_EQ(result.Int64(2), 65536 * bytes);
    EXPECT_EQ(result.Int64(3), 65536 * bytes * bytes);
    EXPECT_EQ(result.Int64(4), 16384 * bytes);
    EXPECT_EQ(result.Int64(5), 65536);
    EXPECT_EQ(result.Int64(6), 56);
    EXPECT_EQ(result.Bool(56), reshuff);
    const auto plain = Hans950Test::Run(DecodeParameters(dtype));
    ASSERT_EQ(plain.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.workspaces.size(), 1U);
    EXPECT_EQ(result.workspaces, plain.workspaces);
}

INSTANTIATE_TEST_SUITE_P(StorageAndAttributes, HansDecode950Modes,
                         testing::Combine(testing::Values(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT), testing::Bool()));

TEST(HansDecode950Tiling, LaunchRespectsAvailableCores)
{
    for (const uint64_t cores : {uint64_t{1}, uint64_t{48}, uint64_t{56}, uint64_t{64}}) {
        SCOPED_TRACE(cores);
        auto parameters = DecodeParameters();
        parameters.coreNum_ = cores;
        const auto result = Hans950Test::Run(parameters);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        const auto expected = std::min(cores, uint64_t{56});
        EXPECT_EQ(result.blockDim, expected);
        EXPECT_EQ(result.Int64(6), expected);
    }
}

TEST(HansDecode950Tiling, RejectsInvalidPdfLength)
{
    auto parameters = DecodeParameters();
    parameters.inputTensorDesc_[3] = Tensor(255, ge::DT_INT32);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansDecode950Tiling, RejectsTruncatedFixedHeader)
{
    for (const auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const int64_t bytes = ge::GetSizeByDataType(dtype);
        for (const int64_t count : {int64_t{0}, 512 / bytes - 1}) {
            SCOPED_TRACE(count);
            auto parameters = DecodeParameters(dtype);
            parameters.inputTensorDesc_[1] = Tensor(count, dtype);
            EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
        }
    }
}

TEST(HansDecode950Tiling, AcceptsHeaderOnlyFixedCapacity)
{
    for (const auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const int64_t bytes = ge::GetSizeByDataType(dtype);
        auto parameters = DecodeParameters(dtype);
        // A header-only fixed buffer is valid when every value is stored in var.
        parameters.inputTensorDesc_[1] = Tensor(512 / bytes, dtype);
        parameters.inputTensorDesc_[2] = Tensor(65536 / bytes, dtype);
        const auto result = Hans950Test::Run(parameters);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.Int64(1), 512);
    }
}

TEST(HansDecode950Tiling, RejectsMismatchedMantissaOrOutput)
{
    auto parameters = DecodeParameters();
    parameters.inputTensorDesc_[0] = Tensor(49151, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters = DecodeParameters();
    parameters.outputTensorDesc_[0] = Tensor(32768, ge::DT_FLOAT);
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

TEST(HansDecode950Tiling, RejectsMissingCoresOrDcacheOnlyUb)
{
    auto parameters = DecodeParameters();
    parameters.coreNum_ = 0;
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
    parameters = DecodeParameters();
    parameters.ubSize_ = 32 * 1024;
    EXPECT_EQ(Hans950Test::Run(parameters).status, ge::GRAPH_FAILED);
}

} // namespace
