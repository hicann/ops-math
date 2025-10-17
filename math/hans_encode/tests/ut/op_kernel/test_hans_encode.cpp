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
#include "../../../op_host/hans_encode_tiling.h"
#include "data_utils.h"

#include <cstdint>
using namespace std;
// using namespace AscendC;

extern "C" __global__ __aicore__ void hans_encode(
    GM_ADDR input_gm, GM_ADDR pdf_gm, GM_ADDR pdf_ref, GM_ADDR output_mantissa_gm, GM_ADDR fixed_gm, GM_ADDR var_gm,
    GM_ADDR workspace, GM_ADDR tiling);

class hans_encode_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "hans_encode_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "hans_encode_test TearDown\n" << endl;
    }
};

// test case 0
TEST_F(hans_encode_test, test_case_0)
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
    size_t tilingEncodeByteSize = sizeof(HansEncodeTilingData);
    size_t workSpaceSize = reshuff ? outputMaxSize : 0;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* mantissa = (uint8_t*)AscendC::GmAlloc(mantissaByteSize);
    uint8_t* outputFixed = (uint8_t*)AscendC::GmAlloc(outputFixedByteSize);
    uint8_t* outputVar = (uint8_t*)AscendC::GmAlloc(outputVarByteSize);
    uint8_t* pdf = (uint8_t*)AscendC::GmAlloc(pdfByteSize);
    uint8_t* tilingEncode = (uint8_t*)AscendC::GmAlloc(tilingEncodeByteSize);

    HansEncodeTilingData* encodeTiling4TestCase = reinterpret_cast<HansEncodeTilingData*>(tilingEncode);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    system("cp -r ../../../../../../../ops/math/hans_encode/tests/ut/op_kernel/hans_encode_data ./");
    system("chmod -R 755 ./hans_encode_data/");
    system("cd ./hans_encode_data/ && rm -rf ./*bin");
    system("cd ./hans_encode_data/ && python3 gen_data.py '(1, 32768)' 'float32'");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/hans_encode_data/float32_input.bin", inputByteSize, input, inputByteSize);
    ReadFile(path + "/hans_encode_data/golden_pdf.bin", pdfByteSize, pdf, pdfByteSize);

    // encode
    int64_t processCoreDim = testNumel / 32768 > blockDim ? blockDim : testNumel / 32768;
    int64_t processBlockLoopNum = testNumel / 64;
    int64_t processLoopPerCore = processBlockLoopNum / processCoreDim;
    int64_t processLoopLastCore = processLoopPerCore + (processBlockLoopNum % processCoreDim);
    int64_t fixedLengthPerCore = outputFixedByteSize / processCoreDim;
    int64_t fixedLengthLastCore = fixedLengthPerCore + outputFixedByteSize % processCoreDim;
    int64_t varLength = testNumel;
    encodeTiling4TestCase->processCoreDim = processCoreDim;
    encodeTiling4TestCase->processLoopPerCore = processLoopPerCore;
    encodeTiling4TestCase->processLoopLastCore = processLoopLastCore;
    encodeTiling4TestCase->fixedLengthPerCore = fixedLengthPerCore;
    encodeTiling4TestCase->fixedLengthLastCore = fixedLengthLastCore;
    encodeTiling4TestCase->varLength = varLength;
    encodeTiling4TestCase->statistic = statistic;
    encodeTiling4TestCase->reshuff = reshuff;
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(
        hans_encode, blockDim, input, pdf, pdf, mantissa, outputFixed, outputVar, workspace,
        (uint8_t*)encodeTiling4TestCase);
    WriteFile("./hans_encode_data/output_pdf.bin", pdf, pdfByteSize);
    WriteFile("./hans_encode_data/mantissa.bin", mantissa, mantissaByteSize);
    WriteFile("./hans_encode_data/fixed.bin", outputFixed, outputFixedByteSize);
    WriteFile("./hans_encode_data/var.bin", outputVar, outputVarByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(pdf);
    AscendC::GmFree(mantissa);
    AscendC::GmFree(outputFixed);
    AscendC::GmFree(outputVar);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingEncode);
    free(path_);
}

// test case 1
TEST_F(hans_encode_test, test_case_1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t testNumel = 32768;
    float fixedRatio = 1.0;
    bool statistic = true;
    bool reshuff = false;
    string dtypeName = "float32";

    // allocate memory
    size_t outputMaxSize = testNumel + testNumel / 64 + blockDim * 8448;
    size_t inputByteSize = testNumel * sizeof(float);
    size_t mantissaByteSize = testNumel * (sizeof(float) - 1);
    size_t outputFixedByteSize = size_t(testNumel * fixedRatio);
    size_t outputVarByteSize = outputMaxSize - outputFixedByteSize;
    size_t pdfByteSize = 256 * sizeof(int32_t);
    size_t tilingEncodeByteSize = sizeof(HansEncodeTilingData);
    size_t workSpaceSize = reshuff ? outputMaxSize : 0;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* mantissa = (uint8_t*)AscendC::GmAlloc(mantissaByteSize);
    uint8_t* outputFixed = (uint8_t*)AscendC::GmAlloc(outputFixedByteSize);
    uint8_t* outputVar = (uint8_t*)AscendC::GmAlloc(outputVarByteSize);
    uint8_t* pdf = (uint8_t*)AscendC::GmAlloc(pdfByteSize);
    uint8_t* tilingEncode = (uint8_t*)AscendC::GmAlloc(tilingEncodeByteSize);

    HansEncodeTilingData* encodeTiling4TestCase = reinterpret_cast<HansEncodeTilingData*>(tilingEncode);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    system("cp -r ../../../../../../../ops/math/hans_encode/tests/ut/op_kernel/hans_encode_data ./");
    system("chmod -R 755 ./hans_encode_data/");
    system("cd ./hans_encode_data/ && rm -rf ./*bin");
    system("cd ./hans_encode_data/ && python3 gen_data.py '(1, 32768)' 'float32'");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/hans_encode_data/float32_input.bin", inputByteSize, input, inputByteSize);
    ReadFile(path + "/hans_encode_data/golden_pdf.bin", pdfByteSize, pdf, pdfByteSize);

    // encode
    int64_t processCoreDim = testNumel / 32768 > blockDim ? blockDim : testNumel / 32768;
    int64_t processBlockLoopNum = testNumel / 64;
    int64_t processLoopPerCore = processBlockLoopNum / processCoreDim;
    int64_t processLoopLastCore = processLoopPerCore + (processBlockLoopNum % processCoreDim);
    int64_t fixedLengthPerCore = outputFixedByteSize / processCoreDim;
    int64_t fixedLengthLastCore = fixedLengthPerCore + outputFixedByteSize % processCoreDim;
    int64_t varLength = testNumel;
    encodeTiling4TestCase->processCoreDim = processCoreDim;
    encodeTiling4TestCase->processLoopPerCore = processLoopPerCore;
    encodeTiling4TestCase->processLoopLastCore = processLoopLastCore;
    encodeTiling4TestCase->fixedLengthPerCore = fixedLengthPerCore;
    encodeTiling4TestCase->fixedLengthLastCore = fixedLengthLastCore;
    encodeTiling4TestCase->varLength = varLength;
    encodeTiling4TestCase->statistic = statistic;
    encodeTiling4TestCase->reshuff = reshuff;
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(
        hans_encode, blockDim, input, pdf, pdf, mantissa, outputFixed, outputVar, workspace,
        (uint8_t*)encodeTiling4TestCase);
    WriteFile("./hans_encode_data/output_pdf.bin", pdf, pdfByteSize);
    WriteFile("./hans_encode_data/mantissa.bin", mantissa, mantissaByteSize);
    WriteFile("./hans_encode_data/fixed.bin", outputFixed, outputFixedByteSize);
    WriteFile("./hans_encode_data/var.bin", outputVar, outputVarByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(pdf);
    AscendC::GmFree(mantissa);
    AscendC::GmFree(outputFixed);
    AscendC::GmFree(outputVar);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingEncode);
    free(path_);
}

// test case 2
TEST_F(hans_encode_test, test_case_2)
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
    size_t tilingEncodeByteSize = sizeof(HansEncodeTilingData);
    size_t workSpaceSize = reshuff ? outputMaxSize : 0;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* mantissa = (uint8_t*)AscendC::GmAlloc(mantissaByteSize);
    uint8_t* outputFixed = (uint8_t*)AscendC::GmAlloc(outputFixedByteSize);
    uint8_t* outputVar = (uint8_t*)AscendC::GmAlloc(outputVarByteSize);
    uint8_t* pdf = (uint8_t*)AscendC::GmAlloc(pdfByteSize);
    uint8_t* tilingEncode = (uint8_t*)AscendC::GmAlloc(tilingEncodeByteSize);

    HansEncodeTilingData* encodeTiling4TestCase = reinterpret_cast<HansEncodeTilingData*>(tilingEncode);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    system("cp -r ../../../../../../../ops/math/hans_encode/tests/ut/op_kernel/hans_encode_data ./");
    system("chmod -R 755 ./hans_encode_data/");
    system("cd ./hans_encode_data/ && rm -rf ./*bin");
    system("cd ./hans_encode_data/ && python3 gen_data.py '(1, 32768)' 'float16'");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/hans_encode_data/float16_input.bin", inputByteSize, input, inputByteSize);
    ReadFile(path + "/hans_encode_data/golden_pdf.bin", pdfByteSize, pdf, pdfByteSize);

    // encode
    int64_t processCoreDim = testNumel / 32768 > blockDim ? blockDim : testNumel / 32768;
    int64_t processBlockLoopNum = testNumel / 64;
    int64_t processLoopPerCore = processBlockLoopNum / processCoreDim;
    int64_t processLoopLastCore = processLoopPerCore + (processBlockLoopNum % processCoreDim);
    int64_t fixedLengthPerCore = outputFixedByteSize / processCoreDim;
    int64_t fixedLengthLastCore = fixedLengthPerCore + outputFixedByteSize % processCoreDim;
    int64_t varLength = testNumel;
    encodeTiling4TestCase->processCoreDim = processCoreDim;
    encodeTiling4TestCase->processLoopPerCore = processLoopPerCore;
    encodeTiling4TestCase->processLoopLastCore = processLoopLastCore;
    encodeTiling4TestCase->fixedLengthPerCore = fixedLengthPerCore;
    encodeTiling4TestCase->fixedLengthLastCore = fixedLengthLastCore;
    encodeTiling4TestCase->varLength = varLength;
    encodeTiling4TestCase->statistic = statistic;
    encodeTiling4TestCase->reshuff = reshuff;
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        hans_encode, blockDim, input, pdf, pdf, mantissa, outputFixed, outputVar, workspace,
        (uint8_t*)encodeTiling4TestCase);
    WriteFile("./hans_encode_data/output_pdf.bin", pdf, pdfByteSize);
    WriteFile("./hans_encode_data/mantissa.bin", mantissa, mantissaByteSize);
    WriteFile("./hans_encode_data/fixed.bin", outputFixed, outputFixedByteSize);
    WriteFile("./hans_encode_data/var.bin", outputVar, outputVarByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(pdf);
    AscendC::GmFree(mantissa);
    AscendC::GmFree(outputFixed);
    AscendC::GmFree(outputVar);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingEncode);
    free(path_);
}

// test case 3
TEST_F(hans_encode_test, test_case_3)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t testNumel = 32768;
    float fixedRatio = 1.0;
    bool statistic = true;
    bool reshuff = false;
    string dtypeName = "float16";

    // allocate memory
    size_t outputMaxSize = testNumel + testNumel / 64 + blockDim * 8448;
    size_t inputByteSize = testNumel * sizeof(half);
    size_t mantissaByteSize = testNumel * (sizeof(half) - 1);
    size_t outputFixedByteSize = size_t(testNumel * fixedRatio);
    size_t outputVarByteSize = outputMaxSize - outputFixedByteSize;
    size_t pdfByteSize = 256 * sizeof(int32_t);
    size_t tilingEncodeByteSize = sizeof(HansEncodeTilingData);
    size_t workSpaceSize = reshuff ? outputMaxSize : 0;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* mantissa = (uint8_t*)AscendC::GmAlloc(mantissaByteSize);
    uint8_t* outputFixed = (uint8_t*)AscendC::GmAlloc(outputFixedByteSize);
    uint8_t* outputVar = (uint8_t*)AscendC::GmAlloc(outputVarByteSize);
    uint8_t* pdf = (uint8_t*)AscendC::GmAlloc(pdfByteSize);
    uint8_t* tilingEncode = (uint8_t*)AscendC::GmAlloc(tilingEncodeByteSize);

    HansEncodeTilingData* encodeTiling4TestCase = reinterpret_cast<HansEncodeTilingData*>(tilingEncode);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    system("cp -r ../../../../../../../ops/math/hans_encode/tests/ut/op_kernel/hans_encode_data ./");
    system("chmod -R 755 ./hans_encode_data/");
    system("cd ./hans_encode_data/ && rm -rf ./*bin");
    system("cd ./hans_encode_data/ && python3 gen_data.py '(1, 32768)' 'float16'");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/hans_encode_data/float16_input.bin", inputByteSize, input, inputByteSize);
    ReadFile(path + "/hans_encode_data/golden_pdf.bin", pdfByteSize, pdf, pdfByteSize);

    // encode
    int64_t processCoreDim = testNumel / 32768 > blockDim ? blockDim : testNumel / 32768;
    int64_t processBlockLoopNum = testNumel / 64;
    int64_t processLoopPerCore = processBlockLoopNum / processCoreDim;
    int64_t processLoopLastCore = processLoopPerCore + (processBlockLoopNum % processCoreDim);
    int64_t fixedLengthPerCore = outputFixedByteSize / processCoreDim;
    int64_t fixedLengthLastCore = fixedLengthPerCore + outputFixedByteSize % processCoreDim;
    int64_t varLength = testNumel;
    encodeTiling4TestCase->processCoreDim = processCoreDim;
    encodeTiling4TestCase->processLoopPerCore = processLoopPerCore;
    encodeTiling4TestCase->processLoopLastCore = processLoopLastCore;
    encodeTiling4TestCase->fixedLengthPerCore = fixedLengthPerCore;
    encodeTiling4TestCase->fixedLengthLastCore = fixedLengthLastCore;
    encodeTiling4TestCase->varLength = varLength;
    encodeTiling4TestCase->statistic = statistic;
    encodeTiling4TestCase->reshuff = reshuff;
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        hans_encode, blockDim, input, pdf, pdf, mantissa, outputFixed, outputVar, workspace,
        (uint8_t*)encodeTiling4TestCase);
    WriteFile("./hans_encode_data/output_pdf.bin", pdf, pdfByteSize);
    WriteFile("./hans_encode_data/mantissa.bin", mantissa, mantissaByteSize);
    WriteFile("./hans_encode_data/fixed.bin", outputFixed, outputFixedByteSize);
    WriteFile("./hans_encode_data/var.bin", outputVar, outputVarByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(pdf);
    AscendC::GmFree(mantissa);
    AscendC::GmFree(outputFixed);
    AscendC::GmFree(outputVar);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingEncode);
    free(path_);
}