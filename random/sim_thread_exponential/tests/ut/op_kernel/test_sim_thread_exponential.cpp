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
 * \file test_sim_thread_exponential.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/sim_thread_exponential_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void sim_thread_exponential(
    GM_ADDR self, GM_ADDR self_ref, GM_ADDR workspace, GM_ADDR tiling);

class sim_thread_exponential_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "sim_thread_exponential_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "sim_thread_exponential_test TearDown\n" << std::endl;
    }
};

TEST_F(sim_thread_exponential_test, test_case_float_outputshape_8_2)
{
    size_t selfSize = 100 * sizeof(float);
    size_t tilingSize = sizeof(SimThreadExponentialTilingData);

    uint8_t* self = (uint8_t*)AscendC::GmAlloc(selfSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    SimThreadExponentialTilingData* tilingData = reinterpret_cast<SimThreadExponentialTilingData*>(tiling);
    tilingData->key0 = 5;
    tilingData->key1 = 0;
    tilingData->offset_t_low = 0;
    tilingData->offset_t_high = 0;
    tilingData->useCoreNum = 1;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->batchNumTotal = 1;
    tilingData->numel = 200;
    tilingData->stepBlock = 4;
    tilingData->roundedSizeBlock = 4;
    tilingData->range = 1;
    tilingData->handleNumLoop = 0;
    tilingData->handleNumTail = 1;
    tilingData->state = 0;
    tilingData->start = 0;
    tilingData->end = 1;
    tilingData->lambda = 1.0;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(sim_thread_exponential, numBlocks, self, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}