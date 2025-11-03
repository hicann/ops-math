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

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../../../op_host/stft_tiling_base.h"
#include "stft_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void stft(GM_ADDR x, GM_ADDR window, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class stft_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "stft_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "stft_test TearDown\n" << endl;
    }
};

TEST_F(stft_test, test_case_float32)
{
    size_t inputByteSize = 2 * 30000 * sizeof(float);
    size_t windowByteSize = 2 * 201 * 400 * sizeof(float);
    size_t outputByteSize = 2 * 201 * 188 * sizeof(float);
    size_t tiling_data_size = sizeof(STFTTilingData);
    size_t windowWokrspaceSize = ((2 * 188 * 400 * sizeof(float) + 511) / 512) * 512;
    size_t matmulWokrspaceSize = ((2 * 188 * 201 * sizeof(float) + 511) / 512) * 512;
    uint32_t blockDim = 24;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* window = (uint8_t*)AscendC::GmAlloc(windowByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16 + matmulWokrspaceSize + windowWokrspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    STFTTilingData* tilingDatafromBin = reinterpret_cast<STFTTilingData*>(tiling);

    TCubeTiling mmTilingData;

    tilingDatafromBin->tilingKey = 0;
    tilingDatafromBin->batch = 2;
    tilingDatafromBin->inputSize = 30000;
    tilingDatafromBin->nfft = 400;
    tilingDatafromBin->hop = 160;
    tilingDatafromBin->frameCount = 188;
    tilingDatafromBin->blkFrame = 32;
    tilingDatafromBin->matmulM = 201;
    tilingDatafromBin->sizePerRepeat = 10240;

    mmTilingData.usedCoreNum = 1;
    mmTilingData.M = 402;
    mmTilingData.N = 188;
    mmTilingData.Ka = 400;
    mmTilingData.Kb = 400;
    mmTilingData.singleCoreM = 402;
    mmTilingData.singleCoreN = 188;
    mmTilingData.singleCoreK = 400;
    mmTilingData.baseM = 208;
    mmTilingData.baseN = 128;
    mmTilingData.baseK = 32;
    mmTilingData.depthA1 = 13;
    mmTilingData.depthB1 = 10;
    mmTilingData.stepM = 1;
    mmTilingData.stepN = 1;
    mmTilingData.isBias = 0;
    mmTilingData.transLength = 98304;
    mmTilingData.iterateOrder = 1;
    mmTilingData.shareMode = 0;
    mmTilingData.shareL1Size = 389120;
    mmTilingData.shareL0CSize = 98304;
    mmTilingData.shareUbSize = 0;
    mmTilingData.batchM = 1;
    mmTilingData.batchN = 1;
    mmTilingData.singleBatchM = 1;
    mmTilingData.singleBatchN = 1;
    mmTilingData.stepKa = 2;
    mmTilingData.stepKb = 10;
    mmTilingData.dbL0A = 2;
    mmTilingData.dbL0B = 2;
    mmTilingData.dbL0C = 1;
    mmTilingData.ALayoutInfoB = 0;
    mmTilingData.ALayoutInfoS = 0;
    mmTilingData.ALayoutInfoN = 0;
    mmTilingData.ALayoutInfoG = 0;
    mmTilingData.ALayoutInfoD = 0;
    mmTilingData.BLayoutInfoB = 0;
    mmTilingData.BLayoutInfoS = 0;
    mmTilingData.BLayoutInfoN = 0;
    mmTilingData.BLayoutInfoG = 0;
    mmTilingData.BLayoutInfoD = 0;
    mmTilingData.CLayoutInfoB = 0;
    mmTilingData.CLayoutInfoS1 = 0;
    mmTilingData.CLayoutInfoN = 0;
    mmTilingData.CLayoutInfoG = 0;
    mmTilingData.CLayoutInfoS2 = 0;
    mmTilingData.BatchNum = 0;
    tilingDatafromBin->mmTilingData = mmTilingData;

    ICPU_SET_TILING_KEY(1);

    AscendC::GmFree(x);
    AscendC::GmFree(window);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
