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
#include "../../../op_host/mul_addn_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void mul_addn(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class mul_addn_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mul_addn SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mul_addn TearDown\n" << endl;
    }
};

TEST_F(mul_addn_test, test_case_mean_fp32_01)
{
    std::cout << "get_current_dir_name:" << get_current_dir_name() << std::endl;
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/mul_addn/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch_size, num_classes, dtype, reduction, flag
    system("cd ./gen_data/ && python3 gen_data.py 30 1024 float32 mean True");

    size_t x1ByteSize = 30 * 1024 * sizeof(float);
    size_t x2ByteSize = 30 * 1024 * sizeof(float);
    // output
    size_t yByteSize = 30 * 1024 * sizeof(float);
    size_t tilingDataSize = sizeof(MulAddnTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    auto read_file_0 = ReadFile("./gen_data/input_x1.bin", x1ByteSize, x1, x1ByteSize);
    auto read_file_1 = ReadFile("./gen_data/input_target.bin", x2ByteSize, x2, x2ByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    MulAddnTilingData* tilingDatafromBin = reinterpret_cast<MulAddnTilingData*>(tiling);

    tilingDatafromBin->N = 8;
    tilingDatafromBin->shapeNAlign = 0;
    tilingDatafromBin->coreTaskNum = 0;
    tilingDatafromBin->useCoreNum = 3840;
    tilingDatafromBin->lastCoreTaskNum = 16336;
    tilingDatafromBin->mNum = 3840;
    tilingDatafromBin->mLoopNum = 3840;
    tilingDatafromBin->mNumTail = 3840;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(mul_addn, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingDatafromBin));

    WriteFile(path + "/gen_data/out.bin", y, yByteSize);

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
    system("rm -rf gen_data");
}

TEST_F(mul_addn_test, test_case_mean_fp16_01)
{
    std::cout << "get_current_dir_name:" << get_current_dir_name() << std::endl;
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/mul_addn/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch_size, num_classes, dtype, reduction, flag
    system("cd ./gen_data/ && python3 gen_data.py 30 1024 float16 mean True");

    size_t x1ByteSize = 30 * 1024 * sizeof(float);
    size_t x2ByteSize = 30 * 1024 * sizeof(float);
    // output
    size_t yByteSize = 30 * 1024 * sizeof(float);
    size_t tilingDataSize = sizeof(MulAddnTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    auto read_file_0 = ReadFile("./gen_data/input_x1.bin", x1ByteSize, x1, x1ByteSize);
    auto read_file_1 = ReadFile("./gen_data/input_target.bin", x2ByteSize, x2, x2ByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    MulAddnTilingData* tilingDatafromBin = reinterpret_cast<MulAddnTilingData*>(tiling);

    tilingDatafromBin->N = 8;
    tilingDatafromBin->shapeNAlign = 0;
    tilingDatafromBin->coreTaskNum = 0;
    tilingDatafromBin->useCoreNum = 3840;
    tilingDatafromBin->lastCoreTaskNum = 16336;
    tilingDatafromBin->mNum = 3840;
    tilingDatafromBin->mLoopNum = 3840;
    tilingDatafromBin->mNumTail = 3840;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(mul_addn, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingDatafromBin));

    WriteFile(path + "/gen_data/out.bin", y, yByteSize);

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
    system("rm -rf gen_data");
}

TEST_F(mul_addn_test, test_case_mean_bf16_01)
{
    std::cout << "get_current_dir_name:" << get_current_dir_name() << std::endl;
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/mul_addn/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch_size, num_classes, dtype, reduction, flag
    system("cd ./gen_data/ && python3 gen_data.py 30 1024 bf16 mean True");

    size_t x1ByteSize = 30 * 1024 * sizeof(float);
    size_t x2ByteSize = 30 * 1024 * sizeof(float);
    // output
    size_t yByteSize = 30 * 1024 * sizeof(float);
    size_t tilingDataSize = sizeof(MulAddnTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    auto read_file_0 = ReadFile("./gen_data/input_x1.bin", x1ByteSize, x1, x1ByteSize);
    auto read_file_1 = ReadFile("./gen_data/input_target.bin", x2ByteSize, x2, x2ByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    MulAddnTilingData* tilingDatafromBin = reinterpret_cast<MulAddnTilingData*>(tiling);

    tilingDatafromBin->N = 8;
    tilingDatafromBin->shapeNAlign = 0;
    tilingDatafromBin->coreTaskNum = 0;
    tilingDatafromBin->useCoreNum = 3840;
    tilingDatafromBin->lastCoreTaskNum = 16336;
    tilingDatafromBin->mNum = 3840;
    tilingDatafromBin->mLoopNum = 3840;
    tilingDatafromBin->mNumTail = 3840;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(mul_addn, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingDatafromBin));

    WriteFile(path + "/gen_data/out.bin", y, yByteSize);

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
    system("rm -rf gen_data");
}