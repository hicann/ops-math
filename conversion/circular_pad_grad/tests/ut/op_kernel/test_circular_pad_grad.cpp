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
#include "gtest/gtest.h"
#include "test_circular_pad_grad.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void circular_pad_grad(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class circualr_pad_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "circualr_pad_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "circualr_pad_grad_test TearDown\n" << endl;
    }
};

TEST_F(circualr_pad_grad_test, test_case_fp32_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 5 * 5;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 3 * 3;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(144 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 5;
    tilingData->inputW = 5;
    tilingData->outputH = 3;
    tilingData->outputW = 3;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 144;

    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_fp32_bigshape_2d)
{
    uint32_t x_shape = 1 * 1 * 500 * 500;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 300 * 300;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(156000 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 500;
    tilingData->inputW = 500;
    tilingData->outputH = 300;
    tilingData->outputW = 300;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 156000;

    ICPU_SET_TILING_KEY(212);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_fp32_smallshape_3d)
{
    uint32_t x_shape = 1 * 3 * 5 * 5;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 1 * 3 * 3;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(144 * 4 * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 5;
    tilingData->inputW = 5;
    tilingData->outputH = 3;
    tilingData->outputW = 3;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 3;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 144 * 3;

    ICPU_SET_TILING_KEY(311);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_fp32_bigshape_3d)
{
    uint32_t x_shape = 1 * 3 * 500 * 500;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 1 * 300 * 300;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(156000 * 4 * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 500;
    tilingData->inputW = 500;
    tilingData->outputH = 300;
    tilingData->outputW = 300;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 3;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 156000 * 3;

    ICPU_SET_TILING_KEY(312);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_fp16_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 5 * 5;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 3 * 3;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(144 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 5;
    tilingData->inputW = 5;
    tilingData->outputH = 3;
    tilingData->outputW = 3;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 144;

    ICPU_SET_TILING_KEY(221);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_fp16_smallshape_3d)
{
    uint32_t x_shape = 1 * 3 * 5 * 5;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 1 * 3 * 3;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(144 * 4 * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 5;
    tilingData->inputW = 5;
    tilingData->outputH = 3;
    tilingData->outputW = 3;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 3;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 144 * 3;

    ICPU_SET_TILING_KEY(321);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_fp16_bigshape_3d)
{
    uint32_t x_shape = 1 * 3 * 500 * 500;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 1 * 300 * 300;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(156000 * 4 * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 500;
    tilingData->inputW = 500;
    tilingData->outputH = 300;
    tilingData->outputW = 300;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 3;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 156000 * 3;

    ICPU_SET_TILING_KEY(322);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_bf16_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 5 * 5;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 3 * 3;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(144 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 5;
    tilingData->inputW = 5;
    tilingData->outputH = 3;
    tilingData->outputW = 3;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 144;

    ICPU_SET_TILING_KEY(231);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_bf16_smallshape_3d)
{
    uint32_t x_shape = 1 * 3 * 5 * 5;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 1 * 3 * 3;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(144 * 4 * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 5;
    tilingData->inputW = 5;
    tilingData->outputH = 3;
    tilingData->outputW = 3;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 3;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 144 * 3;

    ICPU_SET_TILING_KEY(331);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_grad_test, test_case_bfp16_bigshape_3d)
{
    uint32_t x_shape = 1 * 3 * 500 * 500;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 1 * 300 * 300;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(156000 * 4 * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 500;
    tilingData->inputW = 500;
    tilingData->outputH = 300;
    tilingData->outputW = 300;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 3;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 156000;

    ICPU_SET_TILING_KEY(332);
    ICPU_RUN_KF(circular_pad_grad, blockDim, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
