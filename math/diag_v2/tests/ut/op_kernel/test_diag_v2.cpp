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

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;
//using namespace AscendC;

extern "C" __global__ __aicore__ void diag_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);


class diag_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "diag_v2_test SetUp\n " << endl;
    }
    static void TearDownTestCase() {
        cout << "diag_v2_test TearDown\n" << endl;
    }
};

TEST_F(diag_v2_test, test_case_0) {
    // x
    size_t inputByteSize = 326 * 326 * sizeof(int8_t);
    // y
    size_t outputByteSize = 326 * sizeof(int8_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 3;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 326 326 int8");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case0");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ICPU_SET_TILING_KEY(2101);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_1) {
    // x
    size_t inputByteSize = 326 * 326 * sizeof(int16_t);
    // y
    size_t outputByteSize = 326 * sizeof(int16_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 3;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 326 326 int16");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case0");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2102);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_3) {
    // x
    size_t inputByteSize = 326 * 326 * sizeof(int32_t);
    // y
    size_t outputByteSize = 326 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 3;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 326 326 int32");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case1");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2103);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_4) {
    // x
    size_t inputByteSize = 128 * 128 * sizeof(int64_t);
    // y
    size_t outputByteSize = 128 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 128 128 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case3");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2104);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_5) {
    // x
    size_t inputByteSize = 64 * 64 * 2 * sizeof(int64_t);
    // y
    size_t outputByteSize = 64 * 2  * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case2");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    
    ICPU_SET_TILING_KEY(2105);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_6) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int8_t);
    // y
    size_t outputByteSize = 64 * sizeof(int8_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int8");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2401);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_7) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int16_t);
    // y
    size_t outputByteSize = 64 * sizeof(int16_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int16");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2402);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_8) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int32_t);
    // y
    size_t outputByteSize = 64 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int32");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    
    ICPU_SET_TILING_KEY(2403);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_9) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int64_t);
    // y
    size_t outputByteSize = 64 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2404);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_10) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int32_t);
    // y
    size_t outputByteSize = 64 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int32");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_11) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int16_t);
    // y
    size_t outputByteSize = 64 * sizeof(int16_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int16");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_12) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int64_t);
    // y
    size_t outputByteSize = 64 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(103);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_13) {
    // x
    size_t inputByteSize = 64 * 64 * sizeof(int64_t);
    // y
    size_t outputByteSize = 64 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(104);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_14) {
    // x
    size_t inputByteSize = 326 * 326 * sizeof(int8_t);
    // y
    size_t outputByteSize = 326 * sizeof(int8_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 0;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 326 326 int8");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case0");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ICPU_SET_TILING_KEY(2101);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_15) {
    // x
    size_t inputByteSize = 326 * 326 * sizeof(int16_t);
    // y
    size_t outputByteSize = 326 * sizeof(int16_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 0;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 326 326 int16");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case0");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2102);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_16) {
    // x
    size_t inputByteSize = 326 * 326 * sizeof(int32_t);
    // y
    size_t outputByteSize = 326 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 0;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 326 326 int32");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case1");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2103);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_17) {
    // x
    size_t inputByteSize = 128 * 128 * sizeof(int64_t);
    // y
    size_t outputByteSize = 128 * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 0;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 128 128 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case3");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);
    
    ICPU_SET_TILING_KEY(2104);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(diag_v2_test, test_case_18) {
    // x
    size_t inputByteSize = 64 * 64 * 2 * sizeof(int64_t);
    // y
    size_t outputByteSize = 64 * 2  * sizeof(int64_t);

    size_t tiling_data_size = sizeof(DiagV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 0;
    system("cp -r ../../../../../../../ops/math/diag_v2/tests/ut/op_kernel/diag_v2_data ./");
    system("chmod -R 755 ./diag_v2_data/");
    system("cd ./diag_v2_data/ && rm -rf ./*bin");
    system("cd ./diag_v2_data/ && python3 gen_data.py 64 64 int64");
    system("cd ./diag_v2_data/ && python3 gen_tiling.py case2");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/diag_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/diag_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    
    ICPU_SET_TILING_KEY(2105);
    ICPU_RUN_KF(diag_v2, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}