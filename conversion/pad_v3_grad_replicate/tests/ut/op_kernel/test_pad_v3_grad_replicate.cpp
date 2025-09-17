/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

#include <vector>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_data_def.h"

extern "C" __global__ __aicore__ void pad_v3_grad_replicate(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class pad_v3_grad_replicate_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "pad_v3_grad_replicate SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "pad_v3_grad_replicate TearDown\n" << std::endl;
    }
};

TEST_F(pad_v3_grad_replicate_test, test_float_case)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/pad_v3_grad_replicate/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data.py");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 64;
    uint32_t W = 64;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 1;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(float);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    struct InputParamsInfo params = {N, C, H, W, H, W, H, W - padLeft - padRight, 64, 64, 0, 0, 1, 1, 1, 1};

    ReadFile("./gen_data/tiling.bin", tilingSize, tiling, tilingSize);
    ReadFile("./gen_data/x.bin", x_size, x, x_size);

    WriteFile("./gen_data/dx.bin", dx, dx_size);

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 4>(
        reinterpret_cast<PadV3GradReplicateTilingData*>(tiling), params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1101);
    ICPU_RUN_KF(pad_v3_grad_replicate, blockDim, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 64;
    uint32_t W = 64;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 8;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    struct InputParamsInfo params = {N, C, H, W, H, W, H, W - padLeft - padRight, 64, 64, 0, 0, 1, 1, 1, 3};
    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 64;
    tilingData->width = 64;
    tilingData->alignHeight = 64;
    tilingData->alignWidth = 64;
    tilingData->outHeight = 64;
    tilingData->outWidth = 62;
    tilingData->alignOutHeight = 64;
    tilingData->alignOutWidth = 64;
    tilingData->padTop = 0;
    tilingData->padBottom = 0;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 13440;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3101;
    tilingData->workspacePerCore = 128;
    tilingData->wCalCount = 32;

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(3101);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case2)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 16;
    uint32_t W = 530;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 16;
    tilingData->width = 530;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 544;
    tilingData->outHeight = 16;
    tilingData->outWidth = 528;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 528;
    tilingData->padTop = 0;
    tilingData->padBottom = 0;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 16;
    tilingData->ubFactorElement = 13440;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3101;
    tilingData->workspacePerCore = 128;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        0,
        0,
        1,
        1,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(3101);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case3)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 200;
    uint32_t W = 200;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 200;
    tilingData->width = 200;
    tilingData->alignHeight = 208;
    tilingData->alignWidth = 208;
    tilingData->outHeight = 198;
    tilingData->outWidth = 200;
    tilingData->alignOutHeight = 208;
    tilingData->alignOutWidth = 208;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 0;
    tilingData->padRight = 0;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 6656;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3110;
    tilingData->workspacePerCore = 0;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case4)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 150;
    uint32_t W = 30;
    uint32_t padTop = 3;
    uint32_t padBottom = 5;
    uint32_t padLeft = 3;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 150;
    tilingData->width = 30;
    tilingData->alignHeight = 160;
    tilingData->alignWidth = 32;
    tilingData->outHeight = 142;
    tilingData->outWidth = 26;
    tilingData->alignOutHeight = 144;
    tilingData->alignOutWidth = 32;
    tilingData->padTop = 3;
    tilingData->padBottom = 5;
    tilingData->padLeft = 3;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 208;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3010;
    tilingData->workspacePerCore = 20480;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case5)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 16;
    uint32_t W = 530;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 16;
    tilingData->width = 530;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 544;
    tilingData->outHeight = 14;
    tilingData->outWidth = 528;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 528;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 208;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3100;
    tilingData->workspacePerCore = 69632;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case6)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 3;
    uint32_t W = 64;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 3;
    tilingData->width = 64;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 64;
    tilingData->outHeight = 1;
    tilingData->outWidth = 62;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 64;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 512;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3000;
    tilingData->workspacePerCore = 8192;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case7)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t padTop = 2;
    uint32_t padBottom = 2;
    uint32_t padLeft = 2;
    uint32_t padRight = 2;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 8;
    tilingData->width = 8;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 16;
    tilingData->outHeight = 4;
    tilingData->outWidth = 4;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 16;
    tilingData->padTop = 2;
    tilingData->padBottom = 2;
    tilingData->padLeft = 2;
    tilingData->padRight = 2;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 512;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3000;
    tilingData->workspacePerCore = 2048;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_bfloat16_case8)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 3;
    uint32_t W = 65;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(bfloat16_t);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 3;
    tilingData->width = 65;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 80;
    tilingData->outHeight = 1;
    tilingData->outWidth = 63;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 64;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 208;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3100;
    tilingData->workspacePerCore = 10240;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        3};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_float32_case1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 6;
    uint32_t W = 6;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(float);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 6;
    tilingData->width = 6;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 16;
    tilingData->outHeight = 4;
    tilingData->outWidth = 4;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 16;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 128;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 1000;
    tilingData->workspacePerCore = 20480;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        1};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 4>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_float32_case2)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 3;
    uint32_t W = 65;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(float);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 3;
    tilingData->width = 65;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 80;
    tilingData->outHeight = 1;
    tilingData->outWidth = 63;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 64;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 208;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 1100;
    tilingData->workspacePerCore = 20480;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        1};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_float16_case1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 6;
    uint32_t W = 6;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(half);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 6;
    tilingData->width = 6;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 16;
    tilingData->outHeight = 4;
    tilingData->outWidth = 4;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 16;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 256;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 2000;
    tilingData->workspacePerCore = 10240;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        2};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v3_grad_replicate_test, test_float16_case2)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 3;
    uint32_t W = 65;
    uint32_t padTop = 1;
    uint32_t padBottom = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t blockDim = 48;
    uint32_t ubSize = 192 * 1024 - 32 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV3GradReplicateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(half);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - padLeft - padRight) * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV3GradReplicateTilingData* tilingData = reinterpret_cast<PadV3GradReplicateTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 3;
    tilingData->width = 65;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 80;
    tilingData->outHeight = 1;
    tilingData->outWidth = 63;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 64;
    tilingData->padTop = 1;
    tilingData->padBottom = 1;
    tilingData->padLeft = 1;
    tilingData->padRight = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 416;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 2100;
    tilingData->workspacePerCore = 10240;
    tilingData->wCalCount = 32;
    struct InputParamsInfo params = {
        N,
        C,
        H,
        W,
        tilingData->alignHeight,
        tilingData->alignWidth,
        tilingData->outHeight,
        tilingData->outWidth,
        tilingData->alignOutHeight,
        tilingData->alignOutWidth,
        tilingData->padTop,
        tilingData->padBottom,
        tilingData->padLeft,
        tilingData->padRight,
        1,
        2};

    optiling::GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, 2>(tilingData, params, blockDim, ubSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(
        pad_v3_grad_replicate, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}