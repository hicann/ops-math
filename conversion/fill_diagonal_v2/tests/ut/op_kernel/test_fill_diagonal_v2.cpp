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
#include "../../../op_host/fill_diagonal_v2_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void fill_diagonal_v2(
    GM_ADDR x, GM_ADDR fill_value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class fill_diagonal_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "fill_diagonal_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "fill_diagonal_v2_test TearDown\n" << endl;
    }
};

TEST_F(fill_diagonal_v2_test, test_case_float)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(float);
    size_t value_size = sizeof(float);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);
    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_flaot16)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(short);
    size_t value_size = sizeof(short);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_int32)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(int32_t);
    size_t value_size = sizeof(int32_t);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_int8)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(char);
    size_t value_size = sizeof(char);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_int16)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(short);
    size_t value_size = sizeof(short);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_bfloat16)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(short);
    size_t value_size = sizeof(short);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_uint8)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(char);
    size_t value_size = sizeof(char);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(sizeof(char));
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_bool)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(char);
    size_t value_size = sizeof(char);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_int64_dense)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(int64_t);
    size_t value_size = sizeof(char);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}

TEST_F(fill_diagonal_v2_test, test_case_int64_sparse)
{
    int64_t inputLength = 16;

    // inputs
    size_t self_size = inputLength * sizeof(int64_t);
    size_t value_size = sizeof(char);
    size_t tiling_data_size = sizeof(FillDiagonalV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *fill_value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint8_t *workspace = nullptr;
    uint32_t blockDim = 1; //cpu模拟使用单核
    FillDiagonalV2TilingData* tilingData = reinterpret_cast<FillDiagonalV2TilingData*>(tiling);

    // 设置tilingData
    tilingData->totalLength = inputLength;
    tilingData->step = 3;
    tilingData->end = 9;
    tilingData->ubSize = 192 * 1024; // 256KB
    tilingData->blockLength = 0;
    tilingData->lastBlockLength = 16;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fill_diagonal_v2, blockDim, self, fill_value, self, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(fill_value);
    AscendC::GmFree(tiling);
}