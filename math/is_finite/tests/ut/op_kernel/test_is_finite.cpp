/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_is_finite.cpp
 * \brief
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
#include "../../../op_host/is_finite_tiling.h"
#include "../../../op_kernel/is_finite_struct.h"
#include "../../../../../math/is_finite/op_kernel/is_finite.cpp"

using namespace std;
using namespace IsFiniteNs;

extern "C" __global__ __aicore__ void is_finite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class IsFiniteTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "is_finite_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./is_finite_data/");
    }
    static void TearDownTestCase() {
        std::cout << "is_finite_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string IsFiniteTest::rootPath = "../../../../";
const std::string IsFiniteTest::dataPath = rootPath + "math/is_finite/tests/ut/op_kernel/is_finite_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

TEST_F(IsFiniteTest, test_case_float16_1) {
    size_t tiling_data_size = sizeof(IsFiniteTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    IsFiniteTilingData* tilingDatafromBin = reinterpret_cast<IsFiniteTilingData*>(tiling);

    tilingDatafromBin->usableUbSize = 1;
    tilingDatafromBin->needCoreNum = 64;
    tilingDatafromBin->totalDataCount = 64;
    tilingDatafromBin->perCoreDataCount = 1;
    tilingDatafromBin->tailDataCoreNum = 6400;
    tilingDatafromBin->lastCoreDataCount = 1;

    ICPU_SET_TILING_KEY(3UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cd ./is_finite_data/ && python3 gen_data.py '(128, 64)' 'float16'");	
    uint32_t dataCount = 128 * 64;	
    size_t inputByteSize = dataCount * sizeof(half);
    std::string fileName = "./is_finite_data/float16_input_t_is_finite.bin";;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(bool);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    // ICPU_RUN_KF(is_finite, tilingInfo.blockNum, x, y, workspace, tiling);
    auto is_finite_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::is_finite<20, 10>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(is_finite_kernel, blockDim, x, y, workspace, tiling);

    fileName = "./is_finite_data/float16_output_t_is_finite.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_finite_data/ && python3 compare_data.py 'float16'");
}

TEST_F(IsFiniteTest, test_case_float32_1) {
    size_t tiling_data_size = sizeof(IsFiniteTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    IsFiniteTilingData* tilingDatafromBin = reinterpret_cast<IsFiniteTilingData*>(tiling);

    tilingDatafromBin->usableUbSize = 1;
    tilingDatafromBin->needCoreNum = 64;
    tilingDatafromBin->totalDataCount = 64;
    tilingDatafromBin->perCoreDataCount = 1;
    tilingDatafromBin->tailDataCoreNum = 6400;
    tilingDatafromBin->lastCoreDataCount = 1;

    ICPU_SET_TILING_KEY(3UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cd ./is_finite_data/ && python3 gen_data.py '(256, 33)' 'float32'");	
    uint32_t dataCount = 256 * 33;	
    size_t inputByteSize = dataCount * sizeof(float);
    std::string fileName = "./is_finite_data/float32_input_t_is_finite.bin";;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(bool);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    // ICPU_RUN_KF(is_finite, tilingInfo.blockNum, x, y, workspace, tiling);
    auto is_finite_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::is_finite<30, 10>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(is_finite_kernel, blockDim, x, y, workspace, tiling);

    fileName = "./is_finite_data/float32_output_t_is_finite.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_finite_data/ && python3 compare_data.py 'float32'");
}

// TEST_F(IsFiniteTest, test_case_bfloat16_3) {
//     system("cd ./is_finite_data/ && python3 gen_data.py '(128, 64)' 'bfloat16'");
//     uint32_t dataCount = 128 * 64;
//     size_t inputByteSize = dataCount * sizeof(half);
//     size_t outputByteSize = dataCount * sizeof(bool);
//     size_t tiling_data_size = sizeof(IsFiniteTilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 1;

//     std::string fileName = "./is_finite_data/bfloat16_input_t_is_finite.bin";
//     ReadFile(fileName, inputByteSize, x, inputByteSize);
//     IsFiniteTilingData* tilingDatafromBin = reinterpret_cast<IsFiniteTilingData*>(tiling);


//     tilingDatafromBin->totalDataCount = dataCount;
//     tilingDatafromBin->usableUbSize = 32 * 1024;
//     tilingDatafromBin->needCoreNum = 1;
//     tilingDatafromBin->perCoreDataCount = dataCount;
//     tilingDatafromBin->tailDataCoreNum = 0;
//     tilingDatafromBin->lastCoreDataCount = dataCount;

//     ICPU_SET_TILING_KEY(3); // bfloat16
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(is_finite, blockDim, x, y, workspace, tiling);


//     fileName = "./is_finite_data/bfloat16_output_t_is_finite.bin";
//     WriteFile(fileName, y, outputByteSize);

//     AscendC::GmFree((void*)(x));
//     AscendC::GmFree((void*)(y));
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);

//     system("cd ./is_finite_data/ && python3 compare_data.py 'bfloat16'");
// }

