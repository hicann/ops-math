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
#include <vector>
#include "gtest/gtest.h"
#include "test_angle_v2.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling_data);
class angle_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "angle_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "angle_v2_test TearDown\n" << endl;
    }
};

TEST_F(angle_v2_test, test_case_complex64) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(float) * 2;
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 complex64");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 complex64");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;
    
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_fp32) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(float);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 float32");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 float32");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_fp16) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(half);
    size_t y_size = totalLength * sizeof(half);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 float32");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 float32");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 16;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 128;
    tilingDatafromBin->dataPerRepeat = 128;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_bool) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(bool);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 bool");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 bool");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_uint8) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(uint8_t);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 uint8");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 uint8");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_int8) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(int8_t);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 int8");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 int8");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 32;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_int16) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(int16_t);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 int16");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 int16");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 16;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_int32) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(int32_t);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 int32");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 int32");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(angle_v2_test, test_case_int64) {
    uint32_t totalLength = 32;

    // inputs
    size_t x_size = totalLength * sizeof(int64_t);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../../../../ops/math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 32 int64");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 32 int64");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 32;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 32;
    tilingDatafromBin->tailLength = 32;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 32;
    tilingDatafromBin->tileLength = 64;
    tilingDatafromBin->dataPerRepeat = 64;

    ICPU_SET_TILING_KEY(9);
    ICPU_RUN_KF(angle_v2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
