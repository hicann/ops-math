/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include "securec.h"

using namespace std;
using namespace aicpu;

class TEST_MUL_AICPU_UT : public testing::Test {};

auto CreateMulNodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                      const vector<void*>& datas) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "Mul", "Mul")
        .Input({"x1", data_types[0], shapes[0], datas[0]})
        .Input({"x2", data_types[1], shapes[1], datas[1]})
        .Output({"y", data_types[2], shapes[2], datas[2]});
    return node_def;
}

template <typename T>
void RunMulKernel(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types, const vector<T>& input1,
                  const vector<T>& input2, const vector<T>& expect_output, uint32_t expect_status = KERNEL_STATUS_OK)
{
    auto calc_size = [](const vector<int64_t>& shape) -> uint64_t {
        return shape.empty() ? 1 : accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
    };

    const uint64_t in0_size = calc_size(shapes[0]);
    const uint64_t in1_size = calc_size(shapes[1]);
    const uint64_t out_size = calc_size(shapes[2]);

    auto x1_data = make_unique<T[]>(in0_size);
    auto x2_data = make_unique<T[]>(in1_size);
    auto output_data = make_unique<T[]>(out_size);

    for (uint64_t i = 0; i < in0_size; ++i) {
        x1_data[i] = input1[i];
    }
    for (uint64_t i = 0; i < in1_size; ++i) {
        x2_data[i] = input2[i];
    }
    for (uint64_t i = 0; i < out_size; ++i) {
        output_data[i] = T();
    }

    vector<void*> datas = {static_cast<void*>(x1_data.get()), static_cast<void*>(x2_data.get()),
                           static_cast<void*>(output_data.get())};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, expect_status);

    if (expect_status == KERNEL_STATUS_OK) {
        auto expect = make_unique<T[]>(out_size);
        for (uint64_t i = 0; i < out_size; ++i) {
            expect[i] = expect_output[i];
        }
        EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), out_size));
    }
}

template <typename T>
void RunMulKernelInplace(const vector<vector<int64_t>>& shapes, DataType data_type, const vector<T>& input1,
                         const vector<T>& input2, const vector<T>& expect_output, bool output_aliases_first)
{
    vector<T> left = input1;
    vector<T> right = input2;
    vector<T> expected = expect_output;
    T* output = output_aliases_first ? left.data() : right.data();
    vector<void*> datas = {static_cast<void*>(left.data()), static_cast<void*>(right.data()),
                           static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, {data_type, data_type, data_type}, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expected.data(), expected.size()));
}

void RunDiffTypeInvalidOutputShape(const vector<vector<int64_t>>& shapes)
{
    const auto element_count = [](const vector<int64_t>& shape) {
        return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    };
    vector<int8_t> left(element_count(shapes[0]), 2);
    vector<uint8_t> right(element_count(shapes[1]), 3U);
    vector<int16_t> output(element_count(shapes[2]), 0);
    vector<void*> datas = {static_cast<void*>(left.data()), static_cast<void*>(right.data()),
                           static_cast<void*>(output.data())};
    auto node_def = CreateMulNodeDef(shapes, {DT_INT8, DT_UINT8, DT_INT16}, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

constexpr uint16_t kUint16SafeSquareOperand = 46340;
constexpr uint16_t kUint16OverflowSquareOperand = kUint16SafeSquareOperand + 1;
constexpr int64_t kUint16NeonLanes = 8;
constexpr int64_t kMulSmallBlockElements = 16;
constexpr int64_t kMulLargeBlockElements = 64;
constexpr int64_t kMulTileBudgetBytes = 2048;
constexpr int64_t kMulTileRunLimitBytes = 64;
constexpr int64_t kMulInplaceFlatElements = kMulLargeBlockElements + 1;
constexpr int64_t kMulInplaceShortRows = 4;
constexpr int64_t kMulInplaceShortRun = 3;
constexpr int64_t kMulInplaceStrideOuter = 2;
constexpr int64_t kMulInplaceStrideMiddle = 4;
constexpr int64_t kMulInplaceStrideInner = 3;
constexpr size_t kMulSupportedRank = 8;
constexpr int64_t kComplexPayloadVectorElements = kMulLargeBlockElements + 1;
constexpr uint32_t kPositiveQuietNanBits = 0x7FC00000U;
constexpr uint32_t kNegativeQuietNanBits = 0xFFC00000U;
constexpr int64_t kComplexTileRows = 5;
constexpr int64_t kComplexTileRun = 3;

float FloatFromBits(uint32_t bits)
{
    float value = 0.0F;
    EXPECT_EQ(memcpy_s(&value, sizeof(value), &bits, sizeof(bits)), EOK);
    return value;
}

void RunUint16SquareBoundary(const std::vector<std::vector<int64_t>>& shapes)
{
    const auto elementCount = [](const std::vector<int64_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    };
    const std::vector<DataType> dataTypes = {DT_UINT16, DT_UINT16, DT_UINT16};
    const std::vector<uint16_t> operands = {0, 1, kUint16SafeSquareOperand, kUint16OverflowSquareOperand,
                                            std::numeric_limits<uint16_t>::max()};
    for (const uint16_t operand : operands) {
        SCOPED_TRACE(operand);
        const std::vector<uint16_t> left(elementCount(shapes[0]), operand);
        const std::vector<uint16_t> right(elementCount(shapes[1]), operand);
        const auto expectedValue = static_cast<uint16_t>(static_cast<uint64_t>(operand) * operand);
        const std::vector<uint16_t> expected(elementCount(shapes.back()), expectedValue);
        RunMulKernel(shapes, dataTypes, left, right, expected);
    }
}

TEST_F(TEST_MUL_AICPU_UT, UINT16_SQUARE_BOUNDARY_BLOCKS_AND_TAIL)
{
    RunUint16SquareBoundary({{}, {}, {}});
    for (const int64_t block : {kUint16NeonLanes, kMulSmallBlockElements, kMulLargeBlockElements}) {
        for (const int64_t length : {int64_t{1}, block - 1, block, block + 1}) {
            SCOPED_TRACE(length);
            RunUint16SquareBoundary({{length}, {length}, {length}});
        }
    }
}

TEST_F(TEST_MUL_AICPU_UT, UINT16_SQUARE_BOUNDARY_SCALAR_BROADCAST)
{
    for (const int64_t block : {kUint16NeonLanes, kMulSmallBlockElements, kMulLargeBlockElements}) {
        for (const int64_t length : {block - 1, block, block + 1}) {
            SCOPED_TRACE(length);
            RunUint16SquareBoundary({{}, {length}, {length}});
            RunUint16SquareBoundary({{length}, {}, {length}});
        }
    }
}

TEST_F(TEST_MUL_AICPU_UT, UINT16_SQUARE_BOUNDARY_TILE_AND_STRIDE)
{
    constexpr int64_t tileElements = kMulTileBudgetBytes / sizeof(uint16_t);
    constexpr int64_t tileRunLimit = kMulTileRunLimitBytes / sizeof(uint16_t);
    constexpr int64_t shortRun = 3;
    const int64_t rows = tileElements + 1;
    for (const int64_t inner : {shortRun, tileRunLimit - 1, tileRunLimit, tileRunLimit + 1}) {
        SCOPED_TRACE(inner);
        RunUint16SquareBoundary({{rows, 1}, {rows, inner}, {rows, inner}});
        RunUint16SquareBoundary({{rows, inner}, {rows, 1}, {rows, inner}});
    }
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> x2 = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    vector<float> expect = {6.0f, 10.0f, 12.0f, 12.0f, 10.0f, 6.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, INT32_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<int32_t> x1 = {1, 2, 3, 4, 5, 6};
    vector<int32_t> x2 = {6, 5, 4, 3, 2, 1};
    vector<int32_t> expect = {6, 10, 12, 12, 10, 6};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, DOUBLE_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<double> x1 = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
    vector<double> x2 = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    vector<double> expect = {3.0, 5.0, 7.0, 9.0, 11.0, 13.0};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT16_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<Eigen::half> x1 = {Eigen::half(1.0), Eigen::half(2.0), Eigen::half(3.0),
                              Eigen::half(4.0), Eigen::half(5.0), Eigen::half(6.0)};
    vector<Eigen::half> x2 = {Eigen::half(6.0), Eigen::half(5.0), Eigen::half(4.0),
                              Eigen::half(3.0), Eigen::half(2.0), Eigen::half(1.0)};
    vector<Eigen::half> expect = {Eigen::half(6.0),  Eigen::half(10.0), Eigen::half(12.0),
                                  Eigen::half(12.0), Eigen::half(10.0), Eigen::half(6.0)};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, COMPLEX64_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<complex<float>> x1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    vector<complex<float>> x2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    vector<complex<float>> expect = {{1.0f * 5.0f - 2.0f * 6.0f, 1.0f * 6.0f + 2.0f * 5.0f},
                                     {3.0f * 7.0f - 4.0f * 8.0f, 3.0f * 8.0f + 4.0f * 7.0f}};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<float> x1 = {3.0f};
    vector<float> x2 = {4.0f};
    vector<float> expect = {12.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_BROADCAST_X_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {2, 3}, {2, 3}};
    vector<float> x1 = {3.0f};
    vector<float> x2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> expect = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_BROADCAST_Y_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {}, {2, 3}};
    vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> x2 = {3.0f};
    vector<float> expect = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_BROADCAST_BOTH_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2}, {2, 1}, {2, 2}};
    vector<float> x1 = {1.0f, 2.0f};
    vector<float> x2 = {3.0f, 4.0f};
    vector<float> expect = {3.0f, 6.0f, 4.0f, 8.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, INT64_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{4, 2048}, {4, 2048}, {4, 2048}};
    vector<int64_t> x1(4 * 2048);
    vector<int64_t> x2(4 * 2048);
    vector<int64_t> expect(4 * 2048);
    for (int i = 0; i < 4 * 2048; ++i) {
        x1[i] = static_cast<int64_t>(i % 100);
        x2[i] = static_cast<int64_t>(2);
        expect[i] = x1[i] * x2[i];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_INT8_UINT8_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_UINT8, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    int8_t x1[6] = {1, 2, 3, -1, -2, -3};
    uint8_t x2[6] = {1, 2, 3, 1, 2, 3};
    int16_t output[6] = {0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int16_t expect[6] = {1, 4, 9, static_cast<int16_t>(-1), static_cast<int16_t>(-4), static_cast<int16_t>(-9)};
    EXPECT_TRUE(CompareResult(output, expect, static_cast<uint64_t>(6)));
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_FLOAT_DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    float x1[2] = {1.5f, 2.5f};
    double x2[2] = {2.0, 4.0};
    double output[2] = {0.0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    double expect[2] = {3.0, 10.0};
    EXPECT_TRUE(CompareResult(output, expect, static_cast<uint64_t>(2)));
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_UINT8_INT32_SUCC)
{
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
    uint8_t x1[4] = {0, 100, 200, 255};
    int32_t x2[4] = {3, 3, 3, 3};
    int32_t output[4] = {0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect[4] = {0, 300, 600, 765};
    EXPECT_TRUE(CompareResult(output, expect, static_cast<uint64_t>(4)));
}

TEST_F(TEST_MUL_AICPU_UT, INPUT_DTYPE_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    bool x1[6] = {true};
    bool x2[6] = {true};
    bool output[6] = {false};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MUL_AICPU_UT, BCAST_SHAPE_MISMATCH)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 3}, {1, 2}, {1, 3}};
    float x1[3] = {1.0f, 2.0f, 3.0f};
    float x2[2] = {4.0f, 5.0f};
    float output[3] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MUL_AICPU_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    float output[6] = {0.0f};
    vector<void*> datas = {static_cast<void*>(nullptr), static_cast<void*>(nullptr), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MUL_AICPU_UT, INT64_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    vector<int64_t> x1 = {1, 2, 3, 4, 5, 6};
    vector<int64_t> x2 = {6, 5, 4, 3, 2, 1};
    vector<int64_t> expect = {6, 10, 12, 12, 10, 6};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, INT64_OVERFLOW_WRAPS)
{
    const int64_t signed_max = std::numeric_limits<int64_t>::max();
    const int64_t signed_min = std::numeric_limits<int64_t>::min();
    std::vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    std::vector<std::vector<int64_t>> shapes = {{3}, {3}, {3}};
    std::vector<int64_t> x1 = {signed_max, signed_min, signed_max};
    std::vector<int64_t> x2 = {2, -1, 2};
    std::vector<int64_t> expect = {-2, signed_min, -2};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, INT32_BLOCK_AND_TAIL_OVERFLOW_WRAP)
{
    constexpr int64_t kBlockAndTailElements = 65;
    const int32_t signed_max = std::numeric_limits<int32_t>::max();
    std::vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    std::vector<std::vector<int64_t>> shapes = {
        {kBlockAndTailElements}, {kBlockAndTailElements}, {kBlockAndTailElements}};
    std::vector<int32_t> x1(static_cast<size_t>(kBlockAndTailElements), signed_max);
    std::vector<int32_t> x2(static_cast<size_t>(kBlockAndTailElements), 2);
    std::vector<int32_t> expect(static_cast<size_t>(kBlockAndTailElements), -2);
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, MIXED_UINT8_INT64_OVERFLOW_WRAPS)
{
    const int64_t signed_max = std::numeric_limits<int64_t>::max();
    uint8_t x1[1] = {2U};
    int64_t x2[1] = {signed_max};
    int64_t output[1] = {0};
    const int64_t expect = -2;
    std::vector<DataType> data_types = {DT_UINT8, DT_INT64, DT_INT64};
    std::vector<std::vector<int64_t>> shapes = {{1}, {1}, {1}};
    std::vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], expect);
}

TEST_F(TEST_MUL_AICPU_UT, SHAPE_PRODUCT_OVERFLOW_FAILS)
{
    const int64_t signed_max = std::numeric_limits<int64_t>::max();
    float x1[1] = {1.0f};
    float x2[2] = {1.0f, 2.0f};
    float output[2] = {0.0f, 0.0f};
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{1, 1}, {1, 2}, {1, 2}};
    std::vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    node_def->MutableInputs(0)->GetTensorShape()->SetDimSizes({signed_max, 1});
    node_def->MutableInputs(1)->GetTensorShape()->SetDimSizes({1, 2});
    node_def->MutableOutputs(0)->GetTensorShape()->SetDimSizes({signed_max, 2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MUL_AICPU_UT, DIFF_TYPE_OUTPUT_SHAPE_MISMATCH_FAILS)
{
    RunDiffTypeInvalidOutputShape({{2, 3}, {2, 3}, {3, 2}});
    RunDiffTypeInvalidOutputShape({{2}, {2}, {3}});
}

TEST_F(TEST_MUL_AICPU_UT, NEGATIVE_DIMENSION_FLAT_FAILS)
{
    const std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    const std::vector<std::vector<std::vector<int64_t>>> shape_cases = {
        {{-1}, {-1}, {-1}}, {{-1}, {1}, {1}}, {{1}, {-1}, {1}}, {{1}, {1}, {-1}}};
    for (const auto& shapes : shape_cases) {
        float x1[1] = {1.0F};
        float x2[1] = {2.0F};
        float output[1] = {0.0F};
        std::vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
        auto node_def = CreateMulNodeDef({{1}, {1}, {1}}, data_types, datas);
        node_def->MutableInputs(0)->GetTensorShape()->SetDimSizes(shapes[0]);
        node_def->MutableInputs(1)->GetTensorShape()->SetDimSizes(shapes[1]);
        node_def->MutableOutputs(0)->GetTensorShape()->SetDimSizes(shapes[2]);
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    }
}

TEST_F(TEST_MUL_AICPU_UT, COMPLEX128_SAME_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<complex<double>> x1 = {{1.0, 2.0}, {3.0, 4.0}};
    vector<complex<double>> x2 = {{5.0, 6.0}, {7.0, 8.0}};
    vector<complex<double>> expect = {{1.0 * 5.0 - 2.0 * 6.0, 1.0 * 6.0 + 2.0 * 5.0},
                                      {3.0 * 7.0 - 4.0 * 8.0, 3.0 * 8.0 + 4.0 * 7.0}};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, BOTH_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<float> x1 = {3.0f};
    vector<float> x2 = {4.0f};
    vector<float> expect = {12.0f};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

void RunFlatInplaceCases()
{
    const vector<float> sequence_65 = [] {
        vector<float> values(kMulInplaceFlatElements);
        std::iota(values.begin(), values.end(), 1.0F);
        return values;
    }();
    vector<float> expected_65(kMulInplaceFlatElements);
    std::transform(sequence_65.begin(), sequence_65.end(), sequence_65.begin(), expected_65.begin(),
                   [](float left, float right) { return left * right; });
    RunMulKernelInplace<float>({{kMulInplaceFlatElements}, {kMulInplaceFlatElements}, {kMulInplaceFlatElements}},
                               DT_FLOAT, sequence_65, sequence_65, expected_65, false);
    RunMulKernelInplace<float>({{kMulInplaceFlatElements}, {kMulInplaceFlatElements}, {kMulInplaceFlatElements}},
                               DT_FLOAT, sequence_65, sequence_65, expected_65, true);

    vector<float> scalar_expected(kMulInplaceFlatElements);
    std::transform(sequence_65.begin(), sequence_65.end(), scalar_expected.begin(),
                   [](float value) { return 3.0F * value; });
    RunMulKernelInplace<float>({{}, {kMulInplaceFlatElements}, {kMulInplaceFlatElements}}, DT_FLOAT, {3.0F},
                               sequence_65, scalar_expected, false);
}

void RunTileInplaceCases()
{
    const vector<float> short_broadcast = {2.0F, 3.0F, 4.0F, 5.0F};
    vector<float> dense_12(kMulInplaceShortRows * kMulInplaceShortRun);
    std::iota(dense_12.begin(), dense_12.end(), 1.0F);
    vector<float> tile_expected(kMulInplaceShortRows * kMulInplaceShortRun);
    for (size_t i = 0; i < tile_expected.size(); ++i) {
        tile_expected[i] = short_broadcast[i / kMulInplaceShortRun] * dense_12[i];
    }
    RunMulKernelInplace<float>({{kMulInplaceShortRows, 1},
                                {kMulInplaceShortRows, kMulInplaceShortRun},
                                {kMulInplaceShortRows, kMulInplaceShortRun}},
                               DT_FLOAT, short_broadcast, dense_12, tile_expected, false);
    RunMulKernelInplace<float>({{kMulInplaceShortRows, kMulInplaceShortRun},
                                {kMulInplaceShortRows, 1},
                                {kMulInplaceShortRows, kMulInplaceShortRun}},
                               DT_FLOAT, dense_12, short_broadcast, tile_expected, true);
}

void RunStrideInplaceCases()
{
    const vector<float> middle_broadcast = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    vector<float> dense_24(kMulInplaceStrideOuter * kMulInplaceStrideMiddle * kMulInplaceStrideInner);
    std::iota(dense_24.begin(), dense_24.end(), 1.0F);
    vector<float> stride_expected(kMulInplaceStrideOuter * kMulInplaceStrideMiddle * kMulInplaceStrideInner);
    for (size_t i = 0; i < stride_expected.size(); ++i) {
        const size_t outer_stride = kMulInplaceStrideMiddle * kMulInplaceStrideInner;
        stride_expected[i] = middle_broadcast[(i / outer_stride) * kMulInplaceStrideInner +
                                              (i % kMulInplaceStrideInner)] *
                             dense_24[i];
    }
    RunMulKernelInplace<float>({{kMulInplaceStrideOuter, 1, kMulInplaceStrideInner},
                                {kMulInplaceStrideOuter, kMulInplaceStrideMiddle, kMulInplaceStrideInner},
                                {kMulInplaceStrideOuter, kMulInplaceStrideMiddle, kMulInplaceStrideInner}},
                               DT_FLOAT, middle_broadcast, dense_24, stride_expected, false);
    RunMulKernelInplace<float>({{kMulInplaceStrideOuter, kMulInplaceStrideMiddle, kMulInplaceStrideInner},
                                {kMulInplaceStrideOuter, 1, kMulInplaceStrideInner},
                                {kMulInplaceStrideOuter, kMulInplaceStrideMiddle, kMulInplaceStrideInner}},
                               DT_FLOAT, dense_24, middle_broadcast, stride_expected, true);
}

TEST_F(TEST_MUL_AICPU_UT, INPLACE_ALIAS_SUCC)
{
    RunFlatInplaceCases();
    RunTileInplaceCases();
    RunStrideInplaceCases();
}

// Innermost dimension broadcast with a short run: 3 floats is 12 bytes, below the byte
// threshold, so this takes the short-inner tile engine.
TEST_F(TEST_MUL_AICPU_UT, FLOAT_INNER_BROADCAST_SHORT_RUN_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4, 1}, {4, 3}, {4, 3}};
    vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> x2(12);
    vector<float> expect(12);
    for (int i = 0; i < 12; ++i) {
        x2[i] = static_cast<float>(i + 1);
        expect[i] = x1[i / 3] * x2[i];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

// Same shape family but 32 int64 in the innermost run is 256 bytes, above the threshold,
// so this one takes the stride path. The two cases together cover both engines.
TEST_F(TEST_MUL_AICPU_UT, INT64_INNER_BROADCAST_LONG_RUN_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 1}, {2, 32}, {2, 32}};
    vector<int64_t> x1 = {3, 5};
    vector<int64_t> x2(64);
    vector<int64_t> expect(64);
    for (int i = 0; i < 64; ++i) {
        x2[i] = i + 1;
        expect[i] = x1[i / 32] * x2[i];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_MIDDLE_BROADCAST_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 3}, {2, 4, 3}, {2, 4, 3}};
    vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<float> x2(24);
    vector<float> expect(24);
    for (int i = 0; i < 24; ++i) {
        x2[i] = static_cast<float>(i + 1);
        expect[i] = x1[(i / 12) * 3 + (i % 3)] * x2[i];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, UINT64_BROADCAST_X_SUCC)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{1, 3}, {2, 3}, {2, 3}};
    vector<uint64_t> x1 = {2, 3, 4};
    vector<uint64_t> x2 = {1, 2, 3, 4, 5, 6};
    vector<uint64_t> expect = {2, 6, 12, 8, 15, 24};
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, FLOAT_RANK8_BROADCAST_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 1, 1, 2}, {2, 2, 2, 2, 2, 1, 1, 2}};
    vector<float> x1 = {3.0f};
    vector<float> x2(64);
    vector<float> expect(64);
    for (int i = 0; i < 64; ++i) {
        x2[i] = static_cast<float>(i + 1);
        expect[i] = 3.0f * x2[i];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

// Alternating rank-8 broadcast: nothing collapses, so the two innermost levels only cover
// four elements. The stride engine handles these short runs through its scalar tail.
TEST_F(TEST_MUL_AICPU_UT, FLOAT_RANK8_ALTERNATING_BROADCAST_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2, 1, 2, 1, 2, 1, 2}, {2, 1, 2, 1, 2, 1, 2, 1}, {2, 2, 2, 2, 2, 2, 2, 2}};
    vector<float> x1(16);
    vector<float> x2(16);
    vector<float> expect(256);
    for (int i = 0; i < 16; ++i) {
        x1[i] = static_cast<float>(i + 1);
        x2[i] = static_cast<float>(i + 2);
    }
    for (int i = 0; i < 256; ++i) {
        int xi = 0;
        int yi = 0;
        for (int d = 0; d < 8; ++d) {
            const int coord = (i >> (7 - d)) & 1;
            if (d % 2 == 1) {
                xi = (xi << 1) | coord;
            } else {
                yi = (yi << 1) | coord;
            }
        }
        expect[i] = x1[xi] * x2[yi];
    }
    RunMulKernel(shapes, data_types, x1, x2, expect);
}

TEST_F(TEST_MUL_AICPU_UT, EMPTY_TENSOR_SUCC)
{
    const auto run_same_type = [](const vector<vector<int64_t>>& shapes) {
        float x1[1] = {0.0f};
        float x2[1] = {0.0f};
        float output[1] = {0.0f};
        vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
        auto node_def = CreateMulNodeDef(shapes, {DT_FLOAT, DT_FLOAT, DT_FLOAT}, datas);
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    };
    run_same_type({{2, 0, 3}, {2, 0, 3}, {2, 0, 3}});
    run_same_type({{0, 3}, {1, 3}, {0, 3}});
    run_same_type({{1, 3}, {0, 3}, {0, 3}});
    run_same_type({{}, {0, 3}, {0, 3}});
    run_same_type({{0, 3}, {}, {0, 3}});

    int8_t x1[1] = {0};
    uint8_t x2[1] = {0U};
    int16_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef({{0, 3}, {1, 3}, {0, 3}}, {DT_INT8, DT_UINT8, DT_INT16}, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// Rank above the ceiling is rejected, matching what the previous implementation did.
TEST_F(TEST_MUL_AICPU_UT, RANK_OVER_LIMIT_FAILS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {1, 1, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 2, 2, 2, 1, 1, 1, 2}};
    float x1[16] = {0.0f};
    float x2[32] = {0.0f};
    float output[32] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

void RunFlatRankBoundary(const std::vector<int64_t>& leftShape, const std::vector<int64_t>& rightShape,
                         const std::vector<int64_t>& outputShape, uint32_t expectedStatus)
{
    const auto count = [](const std::vector<int64_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    };
    const std::vector<float> left(count(leftShape), 2.0F);
    const std::vector<float> right(count(rightShape), 3.0F);
    const std::vector<float> expected(count(outputShape), 6.0F);
    RunMulKernel({leftShape, rightShape, outputShape}, {DT_FLOAT, DT_FLOAT, DT_FLOAT}, left, right, expected,
                 expectedStatus);
}

TEST_F(TEST_MUL_AICPU_UT, FLAT_RANK_BOUNDARY_SAME_SHAPE)
{
    for (const size_t rank : {kMulSupportedRank, kMulSupportedRank + 1}) {
        const uint32_t status = rank == kMulSupportedRank ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
        std::vector<int64_t> shape(rank, 1);
        RunFlatRankBoundary(shape, shape, shape, status);
        shape.back() = 2;
        RunFlatRankBoundary(shape, shape, shape, status);
    }
}

TEST_F(TEST_MUL_AICPU_UT, FLAT_RANK_BOUNDARY_SCALAR_BROADCAST)
{
    for (const size_t rank : {kMulSupportedRank, kMulSupportedRank + 1}) {
        const uint32_t status = rank == kMulSupportedRank ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
        std::vector<int64_t> shape(rank, 1);
        shape.back() = 2;
        RunFlatRankBoundary({}, shape, shape, status);
        RunFlatRankBoundary(shape, {}, shape, status);
    }
}

TEST_F(TEST_MUL_AICPU_UT, FLAT_RANK_BOUNDARY_SINGLE_ELEMENT_BROADCAST)
{
    for (const size_t rank : {kMulSupportedRank, kMulSupportedRank + 1}) {
        const uint32_t status = rank == kMulSupportedRank ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
        const std::vector<int64_t> singleElement(rank, 1);
        std::vector<int64_t> outputShape(rank, 1);
        outputShape.back() = 2;
        RunFlatRankBoundary(singleElement, {2}, outputShape, status);
        RunFlatRankBoundary({2}, singleElement, outputShape, status);
    }
}

// An output tensor smaller than the broadcast result is rejected instead of being written
// past its end, which is what assigning through the Eigen expression used to do.
TEST_F(TEST_MUL_AICPU_UT, OUTPUT_TOO_SMALL_FAILS)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 3}, {4, 3}, {2, 3}};
    float x1[3] = {1.0f, 2.0f, 3.0f};
    float x2[12] = {0.0f};
    float output[6] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// The uint8 x int32 dispatch entry must read the first operand as uint8. A value of 200
// read as int8 would become -56, so this pins the entry against that regression.
TEST_F(TEST_MUL_AICPU_UT, UINT8_INT32_KEEPS_UNSIGNED_RANGE)
{
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
    uint8_t x1[3] = {200U, 255U, 1U};
    int32_t x2[3] = {3, 2, 7};
    int32_t output[3] = {0, 0, 0};
    int32_t expect_out[3] = {600, 510, 7};
    vector<void*> datas = {static_cast<void*>(x1), static_cast<void*>(x2), static_cast<void*>(output)};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expect_out, static_cast<uint64_t>(3)));
}

namespace {
// NaN-aware complex comparison: the special-value probes below legitimately produce NaN,
// which plain operator== would never match.
template <typename T>
bool SameComplexValue(const std::complex<T>& a, const std::complex<T>& b)
{
    const auto same = [](T x, T y) {
        return (std::isnan(x) && std::isnan(y)) || (!std::isnan(x) && !std::isnan(y) && x == y);
    };
    return same(a.real(), b.real()) && same(a.imag(), b.imag());
}

// Repeated identical operands must give identical results no matter where the 16/64-element
// blocked loops stop: (inf, inf) * (0, 1) is the probe, because the plain four-multiply
// form yields (NaN, NaN) while libstdc++ operator* recovers to (-inf, inf) through the
// C99 Annex G fix-up. Any block/tail disagreement shows up as a mixed row.
template <typename T>
void RunComplexSpecialUniformCase(DataType dt, const vector<vector<int64_t>>& shapes)
{
    const uint64_t out_num = static_cast<uint64_t>(
        accumulate(shapes[2].begin(), shapes[2].end(), 1LL, multiplies<int64_t>()));
    const uint64_t in0_num = static_cast<uint64_t>(
        accumulate(shapes[0].begin(), shapes[0].end(), 1LL, multiplies<int64_t>()));
    const uint64_t in1_num = static_cast<uint64_t>(
        accumulate(shapes[1].begin(), shapes[1].end(), 1LL, multiplies<int64_t>()));

    const std::complex<T> a(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
    const std::complex<T> b(T(0), T(1));
    const std::complex<T> expect(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());

    auto x1 = std::make_unique<std::complex<T>[]>(in0_num);
    auto x2 = std::make_unique<std::complex<T>[]>(in1_num);
    auto output = std::make_unique<std::complex<T>[]>(out_num);
    for (uint64_t i = 0; i < in0_num; ++i) {
        x1[i] = a;
    }
    for (uint64_t i = 0; i < in1_num; ++i) {
        x2[i] = b;
    }
    for (uint64_t i = 0; i < out_num; ++i) {
        output[i] = std::complex<T>(T(-1), T(-1));
    }

    vector<DataType> data_types = {dt, dt, dt};
    vector<void*> datas = {static_cast<void*>(x1.get()), static_cast<void*>(x2.get()),
                           static_cast<void*>(output.get())};
    auto node_def = CreateMulNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    for (uint64_t i = 0; i < out_num; ++i) {
        EXPECT_TRUE(SameComplexValue(output[i], expect)) << "index=" << i;
    }
}
} // namespace

// Covers the 16- and 64-element block boundaries plus the 1-element pure-tail case for
// both complex dtypes: identical operands must not change result across a block boundary.
TEST_F(TEST_MUL_AICPU_UT, COMPLEX_SPECIAL_VALUES_UNIFORM_ACROSS_BLOCK_BOUNDARY)
{
    for (int64_t n : {1, 16, 17, 64, 65}) {
        RunComplexSpecialUniformCase<float>(DT_COMPLEX64, {{n}, {n}, {n}});
        RunComplexSpecialUniformCase<double>(DT_COMPLEX128, {{n}, {n}, {n}});
    }
}

// Same probe through both scalar-broadcast orientations.
TEST_F(TEST_MUL_AICPU_UT, COMPLEX_SPECIAL_VALUES_SPLAT_UNIFORM)
{
    for (int64_t n : {17, 65}) {
        RunComplexSpecialUniformCase<float>(DT_COMPLEX64, {{1}, {n}, {n}});
        RunComplexSpecialUniformCase<float>(DT_COMPLEX64, {{n}, {1}, {n}});
        RunComplexSpecialUniformCase<double>(DT_COMPLEX128, {{1}, {n}, {n}});
        RunComplexSpecialUniformCase<double>(DT_COMPLEX128, {{n}, {1}, {n}});
    }
}

// A complex multiply is not bitwise commutative for NaN payloads and signs. The left-scalar
// vector path must keep the same operand order as its scalar tail.
TEST_F(TEST_MUL_AICPU_UT, COMPLEX64_LEFT_SCALAR_PRESERVES_OPERAND_ORDER)
{
    const float positive_nan = FloatFromBits(kPositiveQuietNanBits);
    const float negative_nan = FloatFromBits(kNegativeQuietNanBits);
    std::vector<std::complex<float>> left = {{positive_nan, positive_nan}};
    std::vector<std::complex<float>> right(static_cast<size_t>(kComplexPayloadVectorElements),
                                           {positive_nan, negative_nan});
    std::vector<std::complex<float>> output(static_cast<size_t>(kComplexPayloadVectorElements));
    std::vector<void*> datas = {static_cast<void*>(left.data()), static_cast<void*>(right.data()),
                                static_cast<void*>(output.data())};
    auto node_def = CreateMulNodeDef({{}, {kComplexPayloadVectorElements}, {kComplexPayloadVectorElements}},
                                     {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64}, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    for (size_t i = 1; i < output.size(); ++i) {
        EXPECT_EQ(std::memcmp(output.data(), output.data() + i, sizeof(output[0])), 0) << "index=" << i;
    }
}

// The right-broadcast tile path must produce the same raw bits as the equivalent flat
// multiply. Swapping the dense left operand with the replicated right operand changes NaN
// payload/sign propagation even though ordinary complex multiplication is commutative.
TEST_F(TEST_MUL_AICPU_UT, COMPLEX64_RIGHT_TILE_PRESERVES_OPERAND_ORDER)
{
    const float positive_nan = FloatFromBits(kPositiveQuietNanBits);
    const float negative_nan = FloatFromBits(kNegativeQuietNanBits);
    const size_t output_elements = static_cast<size_t>(kComplexTileRows * kComplexTileRun);
    std::vector<std::complex<float>> left(output_elements, {positive_nan, positive_nan});
    std::vector<std::complex<float>> right_bcast(static_cast<size_t>(kComplexTileRows), {positive_nan, negative_nan});
    std::vector<std::complex<float>> right_flat(output_elements);
    for (size_t i = 0; i < output_elements; ++i) {
        right_flat[i] = right_bcast[i / static_cast<size_t>(kComplexTileRun)];
    }
    std::vector<std::complex<float>> tile_output(output_elements);
    std::vector<std::complex<float>> flat_output(output_elements);
    {
        std::vector<void*> tile_datas = {static_cast<void*>(left.data()), static_cast<void*>(right_bcast.data()),
                                         static_cast<void*>(tile_output.data())};
        auto tile_node = CreateMulNodeDef(
            {{kComplexTileRows, kComplexTileRun}, {kComplexTileRows, 1}, {kComplexTileRows, kComplexTileRun}},
            {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64}, tile_datas);
        RUN_KERNEL(tile_node, HOST, KERNEL_STATUS_OK);
    }

    {
        std::vector<void*> flat_datas = {static_cast<void*>(left.data()), static_cast<void*>(right_flat.data()),
                                         static_cast<void*>(flat_output.data())};
        auto flat_node = CreateMulNodeDef({{kComplexTileRows, kComplexTileRun},
                                           {kComplexTileRows, kComplexTileRun},
                                           {kComplexTileRows, kComplexTileRun}},
                                          {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64}, flat_datas);
        RUN_KERNEL(flat_node, HOST, KERNEL_STATUS_OK);
    }
    EXPECT_EQ(std::memcmp(tile_output.data(), flat_output.data(), output_elements * sizeof(tile_output[0])), 0);
}
