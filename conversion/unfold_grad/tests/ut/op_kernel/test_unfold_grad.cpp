/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_unfold_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/unfold_grad_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void unfold_grad(
    GM_ADDR grad_out, GM_ADDR input_sizes, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling);

class unfold_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "unfold_grad_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "unfold_grad_test TearDown\n" << std::endl;
    }
};

TEST_F(unfold_grad_test, test_case_float32_outputshape_8_2_dim_0_size_3_step_2)
{
    size_t gradOutSize = 3 * 2 * 8 * sizeof(float);
    size_t inputSizeSize = 1 * 3 * sizeof(int64_t);
    size_t outputSize = 8 * 4 * sizeof(float);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 0;
    tilingData->size = 3;
    tilingData->step = 2;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->inputSizeLastDim = 2;
    tilingData->gradOutSizeDim = 3;
    tilingData->typeSizeT1 = 4;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 1;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = blockDim;
    tilingData->ubSizeT1 = 0;
    tilingData->ubSizeT2 = 97280;
    tilingData->outputNumPerCore = 16;
    tilingData->inputNumPerCore = 18;
    tilingData->iterationNumPerCore = 3;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 2944;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 8;
    tilingData->lowestCommonMultiple = 48;
    tilingData->colOnceMaxPerUB = 184;
    tilingData->tailColLength = 1;

    ICPU_SET_TILING_KEY(212);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, blockDim, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}