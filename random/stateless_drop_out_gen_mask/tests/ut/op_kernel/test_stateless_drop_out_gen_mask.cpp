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
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void stateless_drop_out_gen_mask(
    GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR seed1, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100;
constexpr int64_t kElementCount = 256;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class StatelessDropOutGenMaskKernelTest : public testing::Test {
};

TEST_F(StatelessDropOutGenMaskKernelTest, smoke_float)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* prob = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(float))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* seed1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(uint8_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(uint8_t));
    std::memset(tiling, 0, sizeof(RandomUnifiedTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    *reinterpret_cast<float*>(prob) = 0.5f;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(seed1) = 0;
    *reinterpret_cast<int64_t*>(offset) = 0;

    auto* tilingData = reinterpret_cast<RandomUnifiedTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->normalCoreProNum = kElementCount;
    tilingData->tailCoreProNum = kElementCount;
    tilingData->singleBufferSize = kElementCount;
    tilingData->key[0] = 42;
    tilingData->key[1] = 0;
    tilingData->counter[0] = 0;
    tilingData->counter[1] = 0;
    tilingData->counter[2] = 0;
    tilingData->counter[3] = 0;
    tilingData->outputSize = kElementCount;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_drop_out_gen_mask, kNumBlocks, shape, prob, seed, seed1, offset, y, workspace, tiling);

    AscendC::GmFree(shape);
    AscendC::GmFree(prob);
    AscendC::GmFree(seed);
    AscendC::GmFree(seed1);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
