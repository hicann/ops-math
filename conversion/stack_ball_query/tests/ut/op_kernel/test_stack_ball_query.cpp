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

#include <iostream>
#include <string>
#include <cstdint>
#include <sstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/stack_ball_query_tiling.h"

using namespace std;
extern "C" __global__ __aicore__ void stack_ball_query(
    GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt, GM_ADDR idx,
    GM_ADDR workspace, GM_ADDR tiling);

class stack_ball_query_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "stack_ball_query_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "stack_ball_query_test TearDown\n" << endl;
    }
};

TEST_F(stack_ball_query_test, test_case_fp32)
{
    uint32_t batch_size = 2;
    uint32_t len_center_xyz = 10;
    uint32_t len_xyz = 20;
    uint32_t sample_num = 5;

    stringstream shape_raw_info;
    shape_raw_info << batch_size << " ";
    shape_raw_info << len_center_xyz << " ";
    shape_raw_info << len_xyz << " ";
    shape_raw_info << sample_num << " ";

    string shape_info = shape_raw_info.str();

    size_t center_xyz_bytes_size = 30 * sizeof(float);
    size_t xyz_bytes_size = 60 * sizeof(float);
    size_t center_xyz_batch_cnt_bytes_size = 10 * sizeof(int);
    size_t xyz_batch_cnt_bytes_size = 20 * sizeof(int);
    size_t idx_bytes_size = 50 * sizeof(int);

    uint8_t* center_xyz = (uint8_t*)AscendC::GmAlloc(center_xyz_bytes_size);
    uint8_t* xyz = (uint8_t*)AscendC::GmAlloc(xyz_bytes_size);
    uint8_t* center_xyz_batch_cnt = (uint8_t*)AscendC::GmAlloc(center_xyz_batch_cnt_bytes_size);
    uint8_t* xyz_batch_cnt = (uint8_t*)AscendC::GmAlloc(xyz_batch_cnt_bytes_size);
    uint8_t* outputIdx = (uint8_t*)AscendC::GmAlloc(idx_bytes_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);

    size_t tiling_bytes = sizeof(StackBallQueryTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_bytes);
    auto tilingData = reinterpret_cast<StackBallQueryTilingData*>(tiling);
    tilingData->batchSize = 2;
    tilingData->totalLengthCenterXyz = 10;
    tilingData->totalLengthXyz = 20;
    tilingData->totalIdxLength = 50;
    tilingData->coreNum = 2;
    tilingData->centerXyzPerCore = 5;
    tilingData->tailCenterXyzPerCore = 0;
    tilingData->maxRadius = 0.2;
    tilingData->sampleNum = 5;
    uint32_t block_dim = 1;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        stack_ball_query, block_dim, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, outputIdx, workspace,
        tiling);

    AscendC::GmFree((void*)xyz);
    AscendC::GmFree((void*)center_xyz);
    AscendC::GmFree((void*)xyz_batch_cnt);
    AscendC::GmFree((void*)center_xyz_batch_cnt);
    AscendC::GmFree((void*)outputIdx);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)workspace);
}