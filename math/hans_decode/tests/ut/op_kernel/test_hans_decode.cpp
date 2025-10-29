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
#include "../../../op_host/hans_decode_tiling.h"
#include "data_utils.h"

using namespace std;
// using namespace AscendC;

extern "C" __global__ __aicore__ void hans_decode(
    GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf, GM_ADDR recover, GM_ADDR workspace, GM_ADDR tiling);

class hans_decode_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "hans_decode_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "hans_decode_test TearDown\n" << endl;
    }
};

// test case 0
TEST_F(hans_decode_test, test_case_0)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t testNumel = 32768;
    float fixedRatio = 1.0;
    bool statistic = false;
    bool reshuff = false;
    string dtypeName = "float32";

    // allocate memory
    size_t outputMaxSize = testNumel + testNumel / 64 + blockDim * 8448;
    size_t inputByteSize = testNumel * sizeof(float);
    size_t mantissaByteSize = testNumel * (sizeof(float) - 1);
    size_t outputFixedByteSize = size_t(testNumel * fixedRatio);
    size_t outputVarByteSize = outputMaxSize - outputFixedByteSize;
    size_t pdfByteSize = 256 * sizeof(int32_t);
    size_t tilingDecodeByteSize = sizeof(HansDecodeTilingData);
    size_t workSpaceSize = reshuff ? outputMaxSize : 0;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* mantissa = (uint8_t*)AscendC::GmAlloc(mantissaByteSize);
    uint8_t* outputFixed = (uint8_t*)AscendC::GmAlloc(outputFixedByteSize);
    uint8_t* outputVar = (uint8_t*)AscendC::GmAlloc(outputVarByteSize);
    uint8_t* pdf = (uint8_t*)AscendC::GmAlloc(pdfByteSize);
    uint8_t* recover = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    system("cp -r ../../../../math/hans_decode/tests/ut/op_kernel/hans_decode_data ./");
    system("chmod -R 755 ./hans_decode_data/");
    system("cd ./hans_decode_data/ && rm -rf ./*bin");
    system("cd ./hans_decode_data/ && python3 gen_data.py '(1, 32768)' 'float32'");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/hans_decode_data/float32_input.bin", inputByteSize, input, inputByteSize);
    ReadFile(path + "/hans_decode_data/golden_pdf.bin", pdfByteSize, pdf, pdfByteSize);
    ReadFile(path + "/hans_decode_data/fixed.bin", outputFixedByteSize, outputFixed, outputFixedByteSize);
    ReadFile(path + "/hans_decode_data/mantissa.bin", mantissaByteSize, mantissa, mantissaByteSize);

    // decode
    uint8_t* tilingDecode = (uint8_t*)AscendC::GmAlloc(tilingDecodeByteSize);
    HansDecodeTilingData* decodeTiling4TestCase = reinterpret_cast<HansDecodeTilingData*>(tilingDecode);
    decodeTiling4TestCase->fixedByteSize = outputFixedByteSize;
    decodeTiling4TestCase->mantissaByteSize = mantissaByteSize;
    decodeTiling4TestCase->recoverExpByteSize = testNumel;
    decodeTiling4TestCase->recoverByteSize = inputByteSize;
    decodeTiling4TestCase->reshuff = reshuff;
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(
        hans_decode, blockDim, outputFixed, outputVar, mantissa, pdf, recover, workspace,
        (uint8_t*)decodeTiling4TestCase);
    WriteFile("./hans_decode_data/output_recover.bin", recover, inputByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(pdf);
    AscendC::GmFree(mantissa);
    AscendC::GmFree(outputFixed);
    AscendC::GmFree(outputVar);
    AscendC::GmFree(recover);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingDecode);
    free(path_);
}

// test case 1
TEST_F(hans_decode_test, test_case_1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t testNumel = 32768;
    float fixedRatio = 1.0;
    bool statistic = false;
    bool reshuff = false;
    string dtypeName = "float16";

    // allocate memory
    size_t outputMaxSize = testNumel + testNumel / 64 + blockDim * 8448;
    size_t inputByteSize = testNumel * sizeof(half);
    size_t mantissaByteSize = testNumel * (sizeof(half) - 1);
    size_t outputFixedByteSize = size_t(testNumel * fixedRatio);
    size_t outputVarByteSize = outputMaxSize - outputFixedByteSize;
    size_t pdfByteSize = 256 * sizeof(int32_t);
    size_t tilingDecodeByteSize = sizeof(HansDecodeTilingData);
    size_t workSpaceSize = reshuff ? outputMaxSize : 0;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* mantissa = (uint8_t*)AscendC::GmAlloc(mantissaByteSize);
    uint8_t* outputFixed = (uint8_t*)AscendC::GmAlloc(outputFixedByteSize);
    uint8_t* outputVar = (uint8_t*)AscendC::GmAlloc(outputVarByteSize);
    uint8_t* pdf = (uint8_t*)AscendC::GmAlloc(pdfByteSize);
    uint8_t* recover = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    system("cp -r ../../../../math/hans_decode/tests/ut/op_kernel/hans_decode_data ./");
    system("chmod -R 755 ./hans_decode_data/");
    system("cd ./hans_decode_data/ && rm -rf ./*bin");
    system("cd ./hans_decode_data/ && python3 gen_data.py '(1, 32768)' 'float16'");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/hans_decode_data/float16_input.bin", inputByteSize, input, inputByteSize);
    ReadFile(path + "/hans_decode_data/golden_pdf.bin", pdfByteSize, pdf, pdfByteSize);
    ReadFile(path + "/hans_decode_data/fixed.bin", outputFixedByteSize, outputFixed, outputFixedByteSize);
    ReadFile(path + "/hans_decode_data/mantissa.bin", mantissaByteSize, mantissa, mantissaByteSize);

    uint8_t* tilingDecode = (uint8_t*)AscendC::GmAlloc(tilingDecodeByteSize);
    HansDecodeTilingData* decodeTiling4TestCase = reinterpret_cast<HansDecodeTilingData*>(tilingDecode);
    decodeTiling4TestCase->fixedByteSize = outputFixedByteSize;
    decodeTiling4TestCase->mantissaByteSize = mantissaByteSize;
    decodeTiling4TestCase->recoverExpByteSize = testNumel;
    decodeTiling4TestCase->recoverByteSize = inputByteSize;
    decodeTiling4TestCase->reshuff = reshuff;
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        hans_decode, blockDim, outputFixed, outputVar, mantissa, pdf, recover, workspace,
        (uint8_t*)decodeTiling4TestCase);
    WriteFile("./hans_decode_data/output_recover.bin", recover, inputByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(pdf);
    AscendC::GmFree(mantissa);
    AscendC::GmFree(outputFixed);
    AscendC::GmFree(outputVar);
    AscendC::GmFree(recover);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingDecode);
    free(path_);
}
