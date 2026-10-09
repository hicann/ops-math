/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_floor_mod.cpp
 * \brief
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/floor_mod.cpp"

using namespace std;
using namespace FloorModNs;

constexpr uint32_t USABLE_UB_SIZE = 2048;
constexpr uint32_t ALIGNMENT = 32;

class FloorModTest : public testing::Test {
public:
    static std::string srcDir;

protected:
    static void SetUpTestCase()
    {
        std::cout << "floor_mod_test SetUp" << std::endl;

        std::string fullPath = __FILE__;
        size_t lastSlash = fullPath.find_last_of("/");
        if (lastSlash != std::string::npos) {
            srcDir = fullPath.substr(0, lastSlash);
        } else {
            srcDir = ".";
        }
    }

    static void TearDownTestCase() { std::cout << "floor_mod_test TearDown" << std::endl; }
};

std::string FloorModTest::srcDir = "";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

struct KernelBuffers {
    uint8_t* x1 = nullptr;
    uint8_t* x2 = nullptr;
    uint8_t* y = nullptr;
    uint8_t* tiling = nullptr;
};

static KernelBuffers AllocateKernelBuffers(size_t x1Bytes, size_t x2Bytes, size_t outputBytes)
{
    KernelBuffers buffers;
    buffers.x1 = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(x1Bytes, ALIGNMENT)));
    buffers.x2 = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(x2Bytes, ALIGNMENT)));
    buffers.y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputBytes, ALIGNMENT)));
    buffers.tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(FloorModTilingData)));
    return buffers;
}

static void ReleaseKernelBuffers(const KernelBuffers& buffers)
{
    AscendC::GmFree(buffers.x1);
    AscendC::GmFree(buffers.x2);
    AscendC::GmFree(buffers.y);
    AscendC::GmFree(buffers.tiling);
}

static FloorModTilingData MakeDenseDoubleTiling(uint32_t totalElements, uint32_t x1Elements, uint32_t x2Elements,
                                                uint32_t tileElements)
{
    FloorModTilingData tiling = {};
    tiling.mode = FLOOR_MOD_MODE_DENSE;
    tiling.coreNum = 1;
    tiling.dtypeKey = FLOOR_MOD_TPL_DOUBLE;
    tiling.dtypeBytes = sizeof(double);
    tiling.totalElements = totalElements;
    tiling.x1Elements = x1Elements;
    tiling.x2Elements = x2Elements;
    tiling.denseTile = tileElements;
    tiling.maxTRowElems = CeilAlign(tileElements, ALIGNMENT / sizeof(double));
    tiling.maxFp32RowElems = tiling.maxTRowElems;
    tiling.ubUsedBytes = 256U;
    return tiling;
}

static void CheckDoubleOutput(const double* output, const double* expected, uint32_t count, bool specialAware)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (!specialAware) {
            EXPECT_DOUBLE_EQ(output[i], expected[i]) << "mismatch at output index " << i;
        } else if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(output[i])) << "NaN mismatch at " << i;
        } else if (std::isinf(expected[i]) || expected[i] == 0.0) {
            EXPECT_EQ(output[i], expected[i]) << "special value mismatch at " << i;
            EXPECT_EQ(std::signbit(output[i]), std::signbit(expected[i])) << "sign mismatch at " << i;
        } else {
            EXPECT_NEAR(output[i], expected[i], 1.0e-4) << "mismatch at output index " << i;
        }
    }
}

static void RunDoubleDenseCase(const double* x1Host, uint32_t x1Count, const double* x2Host, uint32_t x2Count,
                               const double* expected, uint32_t outputCount, const FloorModTilingData& tilingData,
                               bool specialAware)
{
    const auto buffers = AllocateKernelBuffers(x1Count * sizeof(double), x2Count * sizeof(double),
                                               outputCount * sizeof(double));
    memcpy(buffers.x1, x1Host, x1Count * sizeof(double));
    memcpy(buffers.x2, x2Host, x2Count * sizeof(double));
    memset(buffers.y, 0, CeilAlign(outputCount * sizeof(double), ALIGNMENT));
    memcpy(buffers.tiling, &tilingData, sizeof(tilingData));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = floor_mod<FLOOR_MOD_TPL_DOUBLE, FLOOR_MOD_TPL_DOUBLE, FLOOR_MOD_TPL_DOUBLE,
                          FLOOR_MOD_TPL_PATH_FP64_STORAGE>;
    ICPU_RUN_KF(func, 1, buffers.x1, buffers.x2, buffers.y, nullptr, buffers.tiling);
    CheckDoubleOutput(reinterpret_cast<const double*>(buffers.y), expected, outputCount, specialAware);
    ReleaseKernelBuffers(buffers);
}

static bool ApplyFiniteSpecialDoubleValue(uint32_t selector, double& x1, double& x2)
{
    switch (selector) {
        case 0:
            x1 = std::nextafter(6.0, 0.0);
            x2 = 3.0;
            return true;
        case 1:
            x1 = std::nextafter(-6.0, -7.0);
            x2 = 3.0;
            return true;
        case 2:
            x1 = 1.0e200;
            x2 = 3.0;
            return true;
        case 3:
            x1 = -1.0e200;
            x2 = 3.0;
            return true;
        case 4:
            x1 = std::numeric_limits<double>::denorm_min();
            x2 = -3.0;
            return true;
        case 5:
            x1 = -0.0;
            x2 = -3.0;
            return true;
        default:
            return false;
    }
}

static void ApplyExceptionalDoubleValue(uint32_t selector, double& x1, double& x2)
{
    switch (selector) {
        case 6:
            x1 = 1.0;
            x2 = std::numeric_limits<double>::infinity();
            break;
        case 7:
            x1 = -1.0;
            x2 = std::numeric_limits<double>::infinity();
            break;
        case 8:
            x1 = std::numeric_limits<double>::infinity();
            x2 = 3.0;
            break;
        case 9:
            x1 = 1.0;
            x2 = 0.0;
            break;
        case 10:
            x1 = std::numeric_limits<double>::quiet_NaN();
            break;
        case 11:
            x2 = std::numeric_limits<double>::quiet_NaN();
            break;
        default:
            break;
    }
}

static void ApplySpecialDoubleValue(uint32_t selector, double& x1, double& x2)
{
    if (!ApplyFiniteSpecialDoubleValue(selector, x1, x2)) {
        ApplyExceptionalDoubleValue(selector, x1, x2);
    }
}

static double ReferenceDoubleFloorMod(double x1, double x2)
{
    double remainder = std::fmod(x1, x2);
    if (remainder == 0.0) {
        return std::copysign(0.0, x2);
    }
    if ((remainder < 0.0) != (x2 < 0.0)) {
        remainder += x2;
    }
    return remainder;
}

static void PrepareLargeDoubleCase(double* x1, double* x2, double* expected, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        x1[i] = static_cast<double>((i * 79U) % 255U) - 127.1250000001;
        x2[i] = static_cast<double>((i * 17U) % 126U) + 1.00000003;
        if ((i & 1U) != 0U) {
            x2[i] = -x2[i];
        }
        ApplySpecialDoubleValue(i % 31U, x1[i], x2[i]);
        expected[i] = ReferenceDoubleFloorMod(x1[i], x2[i]);
    }
}

static FloorModTilingData MakeFp16DenseTiling(uint32_t dataCount)
{
    FloorModTilingData tiling = {};
    tiling.mode = FLOOR_MOD_MODE_DENSE;
    tiling.coreNum = 1;
    tiling.dtypeKey = FLOOR_MOD_TPL_FP16;
    tiling.dtypeBytes = sizeof(half);
    tiling.rank = 2;
    tiling.totalElements = dataCount;
    tiling.x1Elements = dataCount;
    tiling.x2Elements = dataCount;
    tiling.denseTile = dataCount;
    tiling.maxTRowElems = CeilAlign(dataCount, ALIGNMENT / sizeof(half));
    tiling.maxFp32RowElems = tiling.maxTRowElems;
    tiling.floorTmpBytes = 32;
    tiling.ubUsedBytes = USABLE_UB_SIZE;
    tiling.outShape[0] = 32;
    tiling.outShape[1] = 32;
    tiling.x1Stride[0] = 32;
    tiling.x1Stride[1] = 1;
    tiling.x2Stride[0] = 32;
    tiling.x2Stride[1] = 1;
    return tiling;
}

static void RunFp16DenseCase(uint32_t dataCount)
{
    size_t byteSize = dataCount * sizeof(half);
    const auto buffers = AllocateKernelBuffers(byteSize, byteSize, byteSize);
    ReadFile("float16_input_x1.bin", byteSize, buffers.x1, byteSize);
    ReadFile("float16_input_x2.bin", byteSize, buffers.x2, byteSize);
    const auto tilingData = MakeFp16DenseTiling(dataCount);
    memcpy(buffers.tiling, &tilingData, sizeof(tilingData));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = floor_mod<FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_FP16, FLOOR_MOD_TPL_PATH_GENERAL>;
    ICPU_RUN_KF(func, 1, buffers.x1, buffers.x2, buffers.y, nullptr, buffers.tiling);
    WriteFile("float16_output_y.bin", buffers.y, byteSize);
    ReleaseKernelBuffers(buffers);
}

static FloorModTilingData MakeInt32CrossTiling(uint32_t x1Count, uint32_t x2Count, uint32_t outputCount)
{
    FloorModTilingData tiling = {};
    tiling.mode = FLOOR_MOD_MODE_CROSSED;
    tiling.coreNum = 1;
    tiling.dtypeKey = FLOOR_MOD_TPL_INT32;
    tiling.dtypeBytes = sizeof(int32_t);
    tiling.totalElements = outputCount;
    tiling.x1Elements = x1Count;
    tiling.x2Elements = x2Count;
    tiling.seedIsX1 = 1;
    tiling.swapped = 1;
    tiling.crossOuter = 1;
    tiling.crossA = x1Count;
    tiling.crossM = 1;
    tiling.crossB = x2Count;
    tiling.crossD = 1;
    tiling.crossDAligned = 8;
    tiling.crossOuterTile = 1;
    tiling.crossATile = x1Count;
    tiling.crossBTile = x2Count;
    tiling.crossUnitBTAligned = 8;
    tiling.crossUnitBAligned = 8;
    tiling.crossOuterTileCount = 1;
    tiling.crossATileCount = 1;
    tiling.crossBTileCount = 1;
    tiling.crossTotalTasks = 1;
    tiling.floorTmpBytes = 32;
    tiling.ubUsedBytes = USABLE_UB_SIZE;
    tiling.rank = 2;
    tiling.outShape[0] = 2;
    tiling.outShape[1] = 3;
    tiling.x1Stride[0] = 1;
    tiling.x2Stride[1] = 1;
    return tiling;
}

static void RunInt32CrossCase(const int32_t* x1Host, uint32_t x1Count, const int32_t* x2Host, uint32_t x2Count,
                              const int32_t* expected, uint32_t outputCount)
{
    const auto buffers = AllocateKernelBuffers(x1Count * sizeof(int32_t), x2Count * sizeof(int32_t),
                                               outputCount * sizeof(int32_t));
    memcpy(buffers.x1, x1Host, x1Count * sizeof(int32_t));
    memcpy(buffers.x2, x2Host, x2Count * sizeof(int32_t));
    memset(buffers.y, 0, CeilAlign(outputCount * sizeof(int32_t), ALIGNMENT));
    const auto tilingData = MakeInt32CrossTiling(x1Count, x2Count, outputCount);
    memcpy(buffers.tiling, &tilingData, sizeof(tilingData));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = floor_mod<FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_INT32, FLOOR_MOD_TPL_PATH_GENERAL>;
    ICPU_RUN_KF(func, 1, buffers.x1, buffers.x2, buffers.y, nullptr, buffers.tiling);
    const auto* output = reinterpret_cast<const int32_t*>(buffers.y);
    for (uint32_t i = 0; i < outputCount; ++i) {
        EXPECT_EQ(output[i], expected[i]) << "mismatch at output index " << i;
    }
    ReleaseKernelBuffers(buffers);
}

TEST_F(FloorModTest, test_case_double_exact_remainder)
{
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t dataCount = 8;
    const double x1Host[dataCount] = {16777217.0, -16777217.0, 16777217.0, 1.0, 0.1, std::numeric_limits<double>::min(),
                                      -3.0,       -0.0};
    const double x2Host[dataCount] = {3.0, 3.0, -3.0, -32768.0, 0.3, std::numeric_limits<double>::denorm_min(),
                                      2.0, -3.0};
    const double expected[dataCount] = {2.0, 1.0, -1.0, -32767.0, 0.1, 0.0, 1.0, -0.0};

    auto* x1 = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(x1Host), ALIGNMENT)));
    auto* x2 = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(x2Host), ALIGNMENT)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(expected), ALIGNMENT)));
    memcpy(x1, x1Host, sizeof(x1Host));
    memcpy(x2, x2Host, sizeof(x2Host));
    memset(y, 0, CeilAlign(sizeof(expected), ALIGNMENT));

    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(FloorModTilingData)));
    auto* tilingData = reinterpret_cast<FloorModTilingData*>(tiling);
    memset(tilingData, 0, sizeof(FloorModTilingData));
    tilingData->mode = FLOOR_MOD_MODE_DENSE;
    tilingData->coreNum = blockDim;
    tilingData->dtypeKey = FLOOR_MOD_TPL_DOUBLE;
    tilingData->dtypeBytes = sizeof(double);
    tilingData->totalElements = dataCount;
    tilingData->x1Elements = dataCount;
    tilingData->x2Elements = dataCount;
    tilingData->denseTile = dataCount;
    tilingData->maxTRowElems = CeilAlign(dataCount, ALIGNMENT / sizeof(double));
    tilingData->maxFp32RowElems = tilingData->maxTRowElems;
    tilingData->floorTmpBytes = 0U;
    tilingData->ubUsedBytes = 256U;
    tilingData->rank = 1;
    tilingData->outShape[0] = dataCount;
    tilingData->x1Stride[0] = 1;
    tilingData->x2Stride[0] = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = floor_mod<FLOOR_MOD_TPL_DOUBLE, FLOOR_MOD_TPL_DOUBLE, FLOOR_MOD_TPL_DOUBLE,
                          FLOOR_MOD_TPL_PATH_FP64_STORAGE>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, nullptr, tiling);

    const auto* output = reinterpret_cast<const double*>(y);
    for (uint32_t i = 0; i < dataCount; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "mismatch at output index " << i;
    }

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
}

TEST_F(FloorModTest, test_case_double_large_vectorized)
{
    constexpr uint32_t dataCount = 257;
    constexpr uint32_t tileCount = 64;
    double x1Host[dataCount];
    double x2Host[dataCount];
    double expected[dataCount];
    PrepareLargeDoubleCase(x1Host, x2Host, expected, dataCount);

    auto tilingData = MakeDenseDoubleTiling(dataCount, dataCount, dataCount, tileCount);
    tilingData.rank = 1;
    tilingData.outShape[0] = dataCount;
    tilingData.x1Stride[0] = 1;
    tilingData.x2Stride[0] = 1;
    RunDoubleDenseCase(x1Host, dataCount, x2Host, dataCount, expected, dataCount, tilingData, true);
}

TEST_F(FloorModTest, test_case_double_trailing_broadcast)
{
    constexpr uint32_t x1Count = 2;
    constexpr uint32_t x2Count = 18;
    constexpr uint32_t outputCount = 18;
    const double x1Host[x1Count] = {16777217.0, 7.0};
    const double x2Host[x2Count] = {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                                    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    const double expected[outputCount] = {2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 7.0, 6.0,
                                          1.0, 3.0, 2.0, 1.0, 0.0, 7.0, 7.0, 7.0, 7.0};

    auto tilingData = MakeDenseDoubleTiling(outputCount, x1Count, x2Count, outputCount);
    tilingData.rank = 2;
    tilingData.outShape[0] = 2;
    tilingData.outShape[1] = 9;
    tilingData.x1Stride[0] = 1;
    tilingData.x2Stride[0] = 9;
    tilingData.x2Stride[1] = 1;
    RunDoubleDenseCase(x1Host, x1Count, x2Host, x2Count, expected, outputCount, tilingData, false);
}

TEST_F(FloorModTest, test_case_float16_1)
{
    constexpr uint32_t dataCount = 32 * 32;
    const std::string scriptPath = srcDir + "/floor_mod_data/gen_data.py";
    const std::string generateCommand = "python3 " + scriptPath + " '(32, 32)' 'float16'";
    EXPECT_EQ(system(generateCommand.c_str()), 0) << "Failed to generate data using script: " << scriptPath;
    RunFp16DenseCase(dataCount);
    const std::string comparePath = srcDir + "/floor_mod_data/compare_data.py";
    const std::string compareCommand = "python3 " + comparePath + " 'float16'";
    EXPECT_EQ(system(compareCommand.c_str()), 0) << "Data comparison failed using script: " << comparePath;
}

TEST_F(FloorModTest, test_case_int32_general_broadcast)
{
    constexpr uint32_t x1Count = 2;
    constexpr uint32_t x2Count = 3;
    constexpr uint32_t outputCount = 6;
    const int32_t x1Host[x1Count] = {5, 7};
    const int32_t x2Host[x2Count] = {2, 3, 4};
    const int32_t expected[outputCount] = {1, 2, 1, 1, 1, 3};

    RunInt32CrossCase(x1Host, x1Count, x2Host, x2Count, expected, outputCount);
}

TEST_F(FloorModTest, test_case_int64_scalar_floor_mod)
{
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t dataCount = 6;
    const int64_t x1Host[dataCount] = {-7, -7, 7, 7, std::numeric_limits<int64_t>::min(), 0};
    const int64_t x2Host[dataCount] = {3, -3, 3, -3, -1, 5};
    const int64_t expected[dataCount] = {2, -1, 1, -2, 0, 0};

    auto* x1 = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(x1Host), ALIGNMENT)));
    auto* x2 = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(x2Host), ALIGNMENT)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(expected), ALIGNMENT)));
    memcpy(x1, x1Host, sizeof(x1Host));
    memcpy(x2, x2Host, sizeof(x2Host));
    memset(y, 0, CeilAlign(sizeof(expected), ALIGNMENT));

    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(FloorModTilingData)));
    auto* tilingData = reinterpret_cast<FloorModTilingData*>(tiling);
    memset(tilingData, 0, sizeof(FloorModTilingData));
    tilingData->mode = FLOOR_MOD_MODE_DENSE;
    tilingData->coreNum = blockDim;
    tilingData->dtypeKey = FLOOR_MOD_TPL_INT64;
    tilingData->dtypeBytes = sizeof(int64_t);
    tilingData->totalElements = dataCount;
    tilingData->x1Elements = dataCount;
    tilingData->x2Elements = dataCount;
    tilingData->denseTile = dataCount;
    tilingData->maxTRowElems = CeilAlign(dataCount, ALIGNMENT / sizeof(int64_t));
    tilingData->maxFp32RowElems = tilingData->maxTRowElems;
    tilingData->floorTmpBytes = 32;
    tilingData->ubUsedBytes = USABLE_UB_SIZE;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = floor_mod<FLOOR_MOD_TPL_INT64, FLOOR_MOD_TPL_INT64, FLOOR_MOD_TPL_INT64, FLOOR_MOD_TPL_PATH_GENERAL>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, nullptr, tiling);

    const auto* output = reinterpret_cast<const int64_t*>(y);
    for (uint32_t i = 0; i < dataCount; ++i) {
        EXPECT_EQ(output[i], expected[i]) << "mismatch at output index " << i;
    }

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
}
