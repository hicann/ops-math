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
#include "../../../op_host/feeds_repeat_tiling.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void feeds_repeat(
    GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class feeds_repeat_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "feeds_repeat_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "feeds_repeat_test TearDown\n" << endl;
    }
};

TEST_F(feeds_repeat_test, test_case_fp32_int32)
{
    system(
        "cp -rf "
        "../../../../conversion/feeds_repeat/tests/ut/op_kernel/feeds_repeat_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape
    system("cd ./gen_data/ && python3 gen_data.py [1, 2, 3, 4] 15 float32 int32 [5, 6, 7]");
    system("cd ./feeds_repeat_data/ && python3 gen_tiling.py case_fp32_int32");
    size_t feeds_size = 4 * 5 * 6 * 7 * sizeof(float);
    size_t feeds_repeat_times_size = 4 * sizeof(int32_t);
    size_t y_size = 15 * 5 * 6 * 7 * sizeof(float);
    size_t tiling_data_size = sizeof(FeedsRepeatTilingData);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size + 32);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    FeedsRepeatTilingData* tilingDatafromBin = reinterpret_cast<FeedsRepeatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(feeds_repeat_test, test_case_fp16_int32)
{
    system(
        "cp -rf "
        "../../../../conversion/feeds_repeat/tests/ut/op_kernel/feeds_repeat_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape
    system("cd ./gen_data/ && python3 gen_data.py [128] 128 float16 int32 [50, 65]");
    system("cd ./feeds_repeat_data/ && python3 gen_tiling.py case_fp16_int32");
    size_t feeds_size = 1 * 50 * 65 * sizeof(half);
    size_t feeds_repeat_times_size = 128 * sizeof(int32_t);
    size_t y_size = 128 * 50 * 65 * sizeof(half);
    size_t tiling_data_size = sizeof(FeedsRepeatTilingData);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size + 32);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    FeedsRepeatTilingData* tilingDatafromBin = reinterpret_cast<FeedsRepeatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(feeds_repeat_test, test_case_bf16_int32)
{
    system(
        "cp -rf "
        "../../../../conversion/feeds_repeat/tests/ut/op_kernel/feeds_repeat_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape
    system("cd ./gen_data/ && python3 gen_data.py [1, 2, 3, 4] 15 bfp16 int32 [50, 60, 7]");
    system("cd ./feeds_repeat_data/ && python3 gen_tiling.py case_bf16_int32");
    size_t feeds_size = 4 * 50 * 60 * 7 * sizeof(bfloat16_t);
    size_t feeds_repeat_times_size = 4 * sizeof(int32_t);
    size_t y_size = 15 * 50 * 60 * 7 * sizeof(bfloat16_t);
    size_t tiling_data_size = sizeof(FeedsRepeatTilingData);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size + 32);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    FeedsRepeatTilingData* tilingDatafromBin = reinterpret_cast<FeedsRepeatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(feeds_repeat_test, test_case_fp32_int64)
{
    system(
        "cp -rf "
        "../../../../conversion/feeds_repeat/tests/ut/op_kernel/feeds_repeat_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape
    system("cd ./gen_data/ && python3 gen_data.py [2]*48 100 float32 int64 [5, 6, 7]");
    system("cd ./feeds_repeat_data/ && python3 gen_tiling.py case_fp32_int64");
    size_t feeds_size = 48 * 5 * 6 * 7 * sizeof(float);
    size_t feeds_repeat_times_size = 48 * sizeof(int64_t);
    size_t y_size = 100 * 5 * 6 * 7 * sizeof(float);
    size_t tiling_data_size = sizeof(FeedsRepeatTilingData);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size + 32);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    FeedsRepeatTilingData* tilingDatafromBin = reinterpret_cast<FeedsRepeatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(feeds_repeat_test, test_case_fp16_int64)
{
    system(
        "cp -rf "
        "../../../../conversion/feeds_repeat/tests/ut/op_kernel/feeds_repeat_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape
    system("cd ./gen_data/ && python3 gen_data.py [1]*50 50 float16 int64 [5, 6, 7]");
    system("cd ./feeds_repeat_data/ && python3 gen_tiling.py case_fp16_int64");
    size_t feeds_size = 50 * 5 * 6 * 7 * sizeof(half);
    size_t feeds_repeat_times_size = 50 * sizeof(int64_t);
    size_t y_size = 50 * 5 * 6 * 7 * sizeof(half);
    size_t tiling_data_size = sizeof(FeedsRepeatTilingData);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size + 32);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    FeedsRepeatTilingData* tilingDatafromBin = reinterpret_cast<FeedsRepeatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(feeds_repeat_test, test_case_bf16_int64)
{
    system(
        "cp -rf "
        "../../../../conversion/feeds_repeat/tests/ut/op_kernel/feeds_repeat_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // feeds_repeat_times, output_feeds_size, feeds_dtype, repeat_times_dtype, row_shape
    system("cd ./gen_data/ && python3 gen_data.py [1]*100 101 bfp16 int64 [5, 6, 7]");
    system("cd ./feeds_repeat_data/ && python3 gen_tiling.py case_bf16_int64");
    size_t feeds_size = 100 * 5 * 6 * 7 * sizeof(bfloat16_t);
    size_t feeds_repeat_times_size = 100 * sizeof(int32_t);
    size_t y_size = 101 * 5 * 6 * 7 * sizeof(bfloat16_t);
    size_t tiling_data_size = sizeof(FeedsRepeatTilingData);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size + 32);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    FeedsRepeatTilingData* tilingDatafromBin = reinterpret_cast<FeedsRepeatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(103);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}