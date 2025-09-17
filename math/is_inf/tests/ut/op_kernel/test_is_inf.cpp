/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_is_inf.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_is_inf.h"
#include "../data_utils.h"

extern "C" __global__ __aicore__ void is_inf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class is_inf_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "is_inf_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "is_inf_test TearDown\n" << std::endl;
    }
};

template <typename T1, typename T2>
inline T1 CeilA2B(T1 a, T2 b) {
    if(b == 0){
        return a;
    }
    return (a + b - 1) / b;
}

TEST_F(is_inf_test, test_float16) {
    system(
        "cp -rf "
        "../../../../../../../ops/math/is_inf/tests/ut/op_kernel/is_inf_data ./");
    system("chmod -R 755 ./is_inf_data/");
    system("cd ./is_inf_data/ && python3 gen_data.py '(512, 1024)' 'float16'");

    uint32_t dataCount = 512 * 1024;
    size_t inputByteSize = dataCount * sizeof(half);
    size_t outputByteSize = dataCount * sizeof(bool);
    size_t tiling_data_size = sizeof(IsInfTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilA2B(inputByteSize, 32) * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilA2B(outputByteSize, 32) * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string fileName = "./is_inf_data/float16_input.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    IsInfTilingData* tilingDatafromBin = reinterpret_cast<IsInfTilingData*>(tiling);


    tilingDatafromBin->totalDataCount = dataCount;
    tilingDatafromBin->usableUbSize = 32 * 1024;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->perCoreDataCount = dataCount;
    tilingDatafromBin->tailDataCoreNum = 0;
    tilingDatafromBin->lastCoreDataCount = dataCount;

    ICPU_SET_TILING_KEY(1); // float16
    ICPU_RUN_KF(is_inf, blockDim, x, y, workspace, tiling);

    
    fileName = "./is_inf_data/float16_output_rel.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_inf_data/ && python3 compare_data.py 'float16'");
}

TEST_F(is_inf_test, test_float32) {
    system(
        "cp -rf "
        "../../../../../../../ops/math/is_inf/tests/ut/op_kernel/is_inf_data ./");
    system("chmod -R 755 ./is_inf_data/");
    system("cd ./is_inf_data/ && python3 gen_data.py '(512, 1024)' 'float32'");

    uint32_t dataCount = 512 * 1024;
    size_t inputByteSize = dataCount * sizeof(float);
    size_t outputByteSize = dataCount * sizeof(bool);
    size_t tiling_data_size = sizeof(IsInfTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilA2B(inputByteSize, 32) * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilA2B(outputByteSize, 32) * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string fileName = "./is_inf_data/float32_input.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    IsInfTilingData* tilingDatafromBin = reinterpret_cast<IsInfTilingData*>(tiling);


    tilingDatafromBin->totalDataCount = dataCount;
    tilingDatafromBin->usableUbSize = 32 * 1024;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->perCoreDataCount = dataCount;
    tilingDatafromBin->tailDataCoreNum = 0;
    tilingDatafromBin->lastCoreDataCount = dataCount;

    ICPU_SET_TILING_KEY(2); // float
    ICPU_RUN_KF(is_inf, blockDim, x, y, workspace, tiling);

    
    fileName = "./is_inf_data/float32_output_rel.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_inf_data/ && python3 compare_data.py 'float32'");
}

TEST_F(is_inf_test, test_bfloat16) {
    system(
        "cp -rf "
        "../../../../../../../ops/math/is_inf/tests/ut/op_kernel/is_inf_data ./");
    system("chmod -R 755 ./is_inf_data/");
    system("cd ./is_inf_data/ && python3 gen_data.py '(512, 1024)' 'bfloat16'");

    uint32_t dataCount = 512 * 1024;
    size_t inputByteSize = dataCount * sizeof(bfloat16_t);
    size_t outputByteSize = dataCount * sizeof(bool);
    size_t tiling_data_size = sizeof(IsInfTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilA2B(inputByteSize, 32) * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilA2B(outputByteSize, 32) * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string fileName = "./is_inf_data/bfloat16_input.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);
   
    IsInfTilingData* tilingDatafromBin = reinterpret_cast<IsInfTilingData*>(tiling);


    tilingDatafromBin->totalDataCount = dataCount;
    tilingDatafromBin->usableUbSize = 32 * 1024;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->perCoreDataCount = dataCount;
    tilingDatafromBin->tailDataCoreNum = 0;
    tilingDatafromBin->lastCoreDataCount = dataCount;

    ICPU_SET_TILING_KEY(3); // bfloat16
    ICPU_RUN_KF(is_inf, blockDim, x, y, workspace, tiling);

    
    fileName = "./is_inf_data/bfloat16_output_rel.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_inf_data/ && python3 compare_data.py 'bfloat16'");
}

