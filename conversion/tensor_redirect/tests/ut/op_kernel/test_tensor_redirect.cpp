/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tensor_redirect.cpp
 * \brief TensorRedirect op_kernel UT
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "../../../op_kernel/tensor_redirect.cpp"

namespace {

constexpr uint32_t NUM_BLOCKS = 1;
constexpr size_t ALIGN_BYTES = 32;
constexpr size_t WORKSPACE_BYTES = 16 * 1024 * 1024;

size_t Align32(size_t size) { return (size + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES; }

class TensorRedirectKernelTest : public testing::Test {};

TEST_F(TensorRedirectKernelTest, empty_tensor_returns_before_zero_length_init_buffer)
{
    const size_t dataBytes = ALIGN_BYTES;
    const size_t tilingBytes = Align32(sizeof(TensorRedirectTilingData));
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_BYTES));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(tilingBytes));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memset(x, 0xA5, dataBytes);
    std::memset(y, 0x5A, dataBytes);
    std::memset(tiling, 0, tilingBytes); // usedCoreNum=0, ubFactor=0: host 的空 Tensor tiling

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((tensor_redirect<0>), NUM_BLOCKS, x, y, workspace, tiling);

    for (size_t i = 0; i < dataBytes; ++i) {
        EXPECT_EQ(y[i], 0x5A) << "empty tensor must not touch output, byte=" << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Ascend950PR_9599 的 CANN 9.2 tikicpulib 尚不能正确回写设备侧 DataCopyPad 结果；
// 此处只验证正常搬运路径可执行且不破坏输入。位级精度由真机 TTK binary 用例覆盖。
TEST_F(TensorRedirectKernelTest, normal_one_block_executes_without_crash)
{
    constexpr int64_t elementCount = 16;
    const size_t dataBytes = Align32(elementCount * sizeof(int32_t));
    const size_t tilingBytes = Align32(sizeof(TensorRedirectTilingData));
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(dataBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_BYTES));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(tilingBytes));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    for (size_t i = 0; i < dataBytes; ++i) {
        x[i] = static_cast<uint8_t>((i * 37U + 11U) & 0xFFU);
    }
    std::memset(y, 0, dataBytes);
    std::memset(tiling, 0, tilingBytes);
    auto* tilingData = reinterpret_cast<TensorRedirectTilingData*>(tiling);
    tilingData->usedCoreNum = 1;
    tilingData->blockFactor = 1;
    tilingData->tailBlockFactor = 1;
    tilingData->ubFactor = elementCount;
    tilingData->tailBlockTailUbFactor = elementCount;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((tensor_redirect<0>), NUM_BLOCKS, x, y, workspace, tiling);

    for (size_t i = 0; i < dataBytes; ++i) {
        EXPECT_EQ(x[i], static_cast<uint8_t>((i * 37U + 11U) & 0xFFU));
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace
