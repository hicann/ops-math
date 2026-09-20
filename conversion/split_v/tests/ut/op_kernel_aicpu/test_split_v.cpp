/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>

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

using namespace std;
using namespace aicpu;

class TEST_SPLITV_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, num_split)                \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();        \
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");                \
    node.Input({"x", data_types[0], shapes[0], datas[0]})                   \
        .Input({"size_splits", data_types[1], shapes[1], datas[1]})         \
        .Input({"split_dim", data_types[2], shapes[2], datas[2]})           \
        .Attr("num_split", num_split);                                      \
    for (int i = 0; i < num_split; i++) {                                   \
        node.Output({"y", data_types[i + 3], shapes[i + 3], datas[i + 3]}); \
    }

#define ADD_CASE(case_name, aicpu_type, base_type, split_dim, num_split)                               \
    TEST_F(TEST_SPLITV_UT, TestSplitV_##case_name##_##aicpu_type)                                      \
    {                                                                                                  \
        if (num_split == 1) {                                                                          \
            vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type};                \
            vector<vector<int64_t>> shapes = {{2, 2, 2}, {1}, {}, {2, 2, 2}};                          \
            base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,              \
                                  (base_type)5, (base_type)6, (base_type)7, (base_type)8};             \
            int64_t size_split[1] = {2};                                                               \
            base_type output[8] = {(base_type)0};                                                      \
            vector<void*> datas = {(void*)input, (void*)size_split, (void*)&split_dim, (void*)output}; \
            CREATE_NODEDEF(shapes, data_types, datas, 1);                                              \
            RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                              \
            base_type expect_out[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,         \
                                       (base_type)5, (base_type)6, (base_type)7, (base_type)8};        \
            EXPECT_EQ(CompareResult<base_type>(output, expect_out, 8), true);                          \
        } else if (split_dim == 1) {                                                                   \
            vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type, aicpu_type};    \
            vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {2, 1, 2}, {2, 1, 2}};               \
            base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,              \
                                  (base_type)5, (base_type)6, (base_type)7, (base_type)8};             \
            base_type output1[4] = {(base_type)0};                                                     \
            base_type output2[4] = {(base_type)0};                                                     \
            int64_t size_split[2] = {1, -1};                                                           \
            vector<void*> datas = {(void*)input, (void*)size_split, (void*)&split_dim, (void*)output1, \
                                   (void*)output2};                                                    \
            CREATE_NODEDEF(shapes, data_types, datas, 2);                                              \
            RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                              \
            base_type expect_out1[4] = {(base_type)1, (base_type)2, (base_type)5, (base_type)6};       \
            base_type expect_out2[4] = {(base_type)3, (base_type)4, (base_type)7, (base_type)8};       \
            EXPECT_EQ(CompareResult<base_type>(output1, expect_out1, 4), true);                        \
            EXPECT_EQ(CompareResult<base_type>(output2, expect_out2, 4), true);                        \
        } else if (split_dim == 0) {                                                                   \
            vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type, aicpu_type};    \
            vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {1, 2, 2}, {1, 2, 2}};               \
            base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,              \
                                  (base_type)5, (base_type)6, (base_type)7, (base_type)8};             \
            base_type output1[4] = {(base_type)0};                                                     \
            base_type output2[4] = {(base_type)0};                                                     \
            int64_t size_split[2] = {-1, 1};                                                           \
            vector<void*> datas = {(void*)input, (void*)size_split, (void*)&split_dim, (void*)output1, \
                                   (void*)output2};                                                    \
            CREATE_NODEDEF(shapes, data_types, datas, 2);                                              \
            RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                              \
            base_type expect_out1[4] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4};       \
            base_type expect_out2[4] = {(base_type)5, (base_type)6, (base_type)7, (base_type)8};       \
            EXPECT_EQ(CompareResult<base_type>(output1, expect_out1, 4), true);                        \
            EXPECT_EQ(CompareResult<base_type>(output2, expect_out2, 4), true);                        \
        }                                                                                              \
    }

#define ADD_CASE_FAILED(case_name, aicpu_type, base_type, split_dim, num_split)                                     \
    TEST_F(TEST_SPLITV_UT, TestSplitV_##case_name##_##aicpu_type)                                                   \
    {                                                                                                               \
        vector<DataType> data_types = {aicpu_type, DT_INT64, DT_INT32, aicpu_type, aicpu_type};                     \
        vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {2, 1, 2}, {2, 1, 2}};                                \
        base_type input[8] = {(base_type)1, (base_type)2, (base_type)3, (base_type)4,                               \
                              (base_type)5, (base_type)6, (base_type)7, (base_type)8};                              \
        int64_t size_split[2] = {1, 1};                                                                             \
        base_type output1[4] = {(base_type)0};                                                                      \
        base_type output2[4] = {(base_type)0};                                                                      \
        vector<void*> datas = {(void*)input, (void*)size_split, (void*)&split_dim, (void*)output1, (void*)output2}; \
        CREATE_NODEDEF(shapes, data_types, datas, num_split);                                                       \
        RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                                                    \
    }

int64_t split_dim1 = 1;
int64_t split_dim0 = 0;
int64_t split_dim2 = 4;

ADD_CASE(two_split_with_dim_1, DT_FLOAT, float, split_dim1, 2)

ADD_CASE(two_split_with_dim_1, DT_DOUBLE, double, split_dim1, 2)

ADD_CASE(two_split_with_dim_1, DT_FLOAT16, uint16_t, split_dim1, 2)

TEST_F(TEST_SPLITV_UT, TestSplitV_FLOAT16_RAW_BITS)
{
    uint16_t input[8] = {0x3C00, 0xC000, 0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x3555};
    uint16_t output1[4] = {0};
    uint16_t output2[4] = {0};
    int64_t size_split[2] = {1, 1};
    int32_t split_dim = 1;
    vector<DataType> data_types = {DT_FLOAT16, DT_INT64, DT_INT32, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 2, 2}, {2}, {}, {2, 1, 2}, {2, 1, 2}};
    vector<void*> datas = {input, size_split, &split_dim, output1, output2};

    CREATE_NODEDEF(shapes, data_types, datas, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint16_t expect1[4] = {0x3C00, 0xC000, 0x7C00, 0xFC00};
    uint16_t expect2[4] = {0x0000, 0x8000, 0x7E00, 0x3555};
    EXPECT_EQ(CompareResult<uint16_t>(output1, expect1, 4), true);
    EXPECT_EQ(CompareResult<uint16_t>(output2, expect2, 4), true);
}

ADD_CASE(two_split_with_dim_0, DT_INT32, int32_t, split_dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT16, int16_t, split_dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT64, int64_t, split_dim0, 2)

ADD_CASE(two_split_with_dim_0, DT_INT8, int8_t, split_dim0, 2)

ADD_CASE(one_split_with_dim_1, DT_BOOL, bool, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT8, uint8_t, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT16, uint16_t, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT32, uint32_t, split_dim1, 1)

ADD_CASE(one_split_with_dim_1, DT_UINT64, uint64_t, split_dim1, 1)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT64, int64_t, split_dim0, 0)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT32, int32_t, split_dim0, 1)

ADD_CASE_FAILED(split_num_not_equal_size_split_num, DT_INT16, int16_t, split_dim2, 2)

TEST_F(TEST_SPLITV_UT, TestSplitV_EMPTY_CASE1)
{
    double input[1] = {0};
    double output1[1] = {0};
    double output2[1] = {0};
    int64_t size_split[2] = {1, 1};
    vector<int64_t> input_shape = {0, 2, 2};
    vector<int64_t> splits_shape = {2};
    vector<int64_t> out1_shape = {0, 1, 2};
    vector<int64_t> out2_shape = {0, 1, 2};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_DOUBLE, out1_shape, output1})
        .Output({"y2", DT_DOUBLE, out2_shape, output2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_EMPTY_CASE2)
{
    double input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    double output1[1] = {0};
    double output2[8] = {0};
    int64_t size_split[2] = {0, 2};
    vector<int64_t> input_shape = {2, 2, 2};
    vector<int64_t> splits_shape = {2};
    vector<int64_t> out1_shape = {2, 0, 2};
    vector<int64_t> out2_shape = {2, 2, 2};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_DOUBLE, out1_shape, output1})
        .Output({"y2", DT_DOUBLE, out2_shape, output2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}
TEST_F(TEST_SPLITV_UT, TestSplitV_LARGE_NUM_SPLIT)
{
    const int64_t num_split = 1024;
    const int64_t inner = 2;
    vector<double> input(static_cast<size_t>(num_split * inner));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>(i);
    }
    vector<int64_t> size_split(static_cast<size_t>(num_split), 1);
    vector<vector<double>> outs(static_cast<size_t>(num_split), vector<double>(static_cast<size_t>(inner), 0.0));

    vector<int64_t> input_shape = {num_split, inner};
    vector<int64_t> splits_shape = {num_split};
    vector<int64_t> out_shape = {1, inner};
    int32_t split_dim = 0;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, input_shape, input.data()})
        .Input({"size_splits", DT_INT64, splits_shape, size_split.data()})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", num_split);
    for (int64_t i = 0; i < num_split; ++i) {
        node.Output({"y", DT_DOUBLE, out_shape, outs[static_cast<size_t>(i)].data()});
    }
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    for (int64_t i = 0; i < num_split; ++i) {
        EXPECT_EQ(outs[static_cast<size_t>(i)][0], static_cast<double>(i * inner));
        EXPECT_EQ(outs[static_cast<size_t>(i)][1], static_cast<double>(i * inner + 1));
    }
}

TEST_F(TEST_SPLITV_UT, TestSplitV_SIZE_SPLITS_INT32)
{
    float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float output1[4] = {0};
    float output2[4] = {0};
    int32_t size_split[2] = {1, 1};
    vector<int64_t> input_shape = {2, 2, 2};
    vector<int64_t> splits_shape = {2};
    vector<int64_t> out_shape = {2, 1, 2};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_FLOAT, input_shape, input})
        .Input({"size_splits", DT_INT32, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_FLOAT, out_shape, output1})
        .Output({"y2", DT_FLOAT, out_shape, output2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect1[4] = {0, 1, 4, 5};
    float expect2[4] = {2, 3, 6, 7};
    EXPECT_EQ(CompareResult<float>(output1, expect1, 4), true);
    EXPECT_EQ(CompareResult<float>(output2, expect2, 4), true);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_DIM1_UNEQUAL_THREE_WAY)
{
    int32_t input[16];
    for (int32_t i = 0; i < 16; ++i) {
        input[i] = i;
    }
    int32_t out1[4] = {0};
    int32_t out2[8] = {0};
    int32_t out3[4] = {0};
    int64_t size_split[3] = {1, 2, 1};
    vector<int64_t> input_shape = {2, 4, 2};
    vector<int64_t> splits_shape = {3};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_INT32, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 3)
        .Output({"y1", DT_INT32, {2, 1, 2}, out1})
        .Output({"y2", DT_INT32, {2, 2, 2}, out2})
        .Output({"y3", DT_INT32, {2, 1, 2}, out3});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect1[4] = {0, 1, 8, 9};
    int32_t expect2[8] = {2, 3, 4, 5, 10, 11, 12, 13};
    int32_t expect3[4] = {6, 7, 14, 15};
    EXPECT_EQ(CompareResult<int32_t>(out1, expect1, 4), true);
    EXPECT_EQ(CompareResult<int32_t>(out2, expect2, 8), true);
    EXPECT_EQ(CompareResult<int32_t>(out3, expect3, 4), true);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_DIM1_SMALL_PATH_BOUNDARY)
{
    constexpr int64_t kNumSplit = 4;
    constexpr size_t kOutputElements = 2;
    int32_t input[kNumSplit * kOutputElements] = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t outputs[kNumSplit][kOutputElements] = {};
    int64_t size_split[kNumSplit] = {1, 1, 1, 1};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_INT32, {kOutputElements, kNumSplit, 1}, input})
        .Input({"size_splits", DT_INT64, {kNumSplit}, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", kNumSplit);
    for (int64_t i = 0; i < kNumSplit; ++i) {
        node.Output({"y", DT_INT32, {kOutputElements, 1, 1}, outputs[i]});
    }
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    for (int64_t i = 0; i < kNumSplit; ++i) {
        EXPECT_EQ(outputs[i][0], i);
        EXPECT_EQ(outputs[i][1], i + kNumSplit);
    }
}

TEST_F(TEST_SPLITV_UT, TestSplitV_DIM1_UINT8_RAW_BITS)
{
    constexpr int64_t kNumSplit = 2;
    constexpr size_t kOutputElements = 4;
    uint8_t input[8] = {0, 1, 254, 255, 128, 127, 85, 170};
    uint8_t output1[kOutputElements] = {};
    uint8_t output2[kOutputElements] = {};
    uint8_t expect1[kOutputElements] = {0, 1, 128, 127};
    uint8_t expect2[kOutputElements] = {254, 255, 85, 170};
    int64_t size_split[kNumSplit] = {1, 1};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_UINT8, {2, kNumSplit, 2}, input})
        .Input({"size_splits", DT_INT64, {kNumSplit}, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", kNumSplit)
        .Output({"y1", DT_UINT8, {2, 1, 2}, output1})
        .Output({"y2", DT_UINT8, {2, 1, 2}, output2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(CompareResult<uint8_t>(output1, expect1, kOutputElements), true);
    EXPECT_EQ(CompareResult<uint8_t>(output2, expect2, kOutputElements), true);
}

// Covers the kGatherChunk (64) batching in SplitVCompute: num_split spans three batches
// (64 / 64 / 1), so the outer base loop, the cross-batch offset accumulation and the reuse
// of dst_chunk all run, including a zero-size split skipped in the middle of a batch.
TEST_F(TEST_SPLITV_UT, TestSplitV_DIM1_MULTI_GATHER_CHUNK)
{
    const int64_t num_split = 129;
    const int64_t prefix = 3;
    const int64_t subfix = 2;
    const int64_t mid = 129;
    const int64_t src_stride = mid * subfix;

    vector<int64_t> size_split(static_cast<size_t>(num_split), 1);
    size_split[70] = 0;
    size_split[71] = 2;

    vector<double> input(static_cast<size_t>(prefix * mid * subfix));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>(i);
    }

    vector<vector<double>> outs(static_cast<size_t>(num_split));
    vector<vector<int64_t>> out_shapes(static_cast<size_t>(num_split));
    for (int64_t i = 0; i < num_split; ++i) {
        const size_t idx = static_cast<size_t>(i);
        out_shapes[idx] = {prefix, size_split[idx], subfix};
        // One spare element keeps data() non-null for the zero-size split.
        outs[idx].assign(static_cast<size_t>(prefix * size_split[idx] * subfix) + 1U, 0.0);
    }

    vector<int64_t> input_shape = {prefix, mid, subfix};
    vector<int64_t> splits_shape = {num_split};
    int32_t split_dim = 1;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, input_shape, input.data()})
        .Input({"size_splits", DT_INT64, splits_shape, size_split.data()})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", num_split);
    for (int64_t i = 0; i < num_split; ++i) {
        node.Output({"y", DT_DOUBLE, out_shapes[static_cast<size_t>(i)], outs[static_cast<size_t>(i)].data()});
    }
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // input[idx] == idx, so every copied element must equal its own source index.
    int64_t offset = 0;
    for (int64_t i = 0; i < num_split; ++i) {
        const int64_t copy_num = subfix * size_split[static_cast<size_t>(i)];
        if (copy_num == 0) {
            continue;
        }
        for (int64_t j = 0; j < prefix; ++j) {
            for (int64_t k = 0; k < copy_num; ++k) {
                EXPECT_EQ(outs[static_cast<size_t>(i)][static_cast<size_t>(j * copy_num + k)],
                          static_cast<double>(offset + j * src_stride + k));
            }
        }
        offset += copy_num;
    }
}
TEST_F(TEST_SPLITV_UT, TestSplitV_SIZE_SPLITS_MINUS_ONE)
{
    float input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float out1[4] = {0};
    float out2[8] = {0};
    int64_t size_split[2] = {1, -1};
    vector<int64_t> input_shape = {3, 4};
    vector<int64_t> splits_shape = {2};
    int32_t split_dim = 0;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_FLOAT, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_FLOAT, {1, 4}, out1})
        .Output({"y2", DT_FLOAT, {2, 4}, out2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect1[4] = {0, 1, 2, 3};
    float expect2[8] = {4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_EQ(CompareResult<float>(out1, expect1, 4), true);
    EXPECT_EQ(CompareResult<float>(out2, expect2, 8), true);
}

// num_split is int64_t; narrowing it to uint32_t before comparing with the output count
// would let a value above UINT32_MAX wrap and pass. 4294967297 wraps to 1.
TEST_F(TEST_SPLITV_UT, TestSplitV_NUM_SPLIT_EXCEEDS_UINT32_FAILS)
{
    double input[4] = {0, 1, 2, 3};
    double out1[2] = {0};
    double out2[2] = {0};
    int64_t size_split[2] = {1, 1};
    vector<int64_t> input_shape = {2, 2};
    vector<int64_t> splits_shape = {2};
    int32_t split_dim = 0;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", static_cast<int64_t>(4294967297LL))
        .Output({"y1", DT_DOUBLE, {1, 2}, out1})
        .Output({"y2", DT_DOUBLE, {1, 2}, out2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_OUTPUT_COUNT_NOT_EQUAL_NUM_SPLIT_FAILS)
{
    constexpr int64_t kNumSplit = 2;
    constexpr size_t kOutputElements = 2;
    double input[kNumSplit * kOutputElements] = {0, 1, 2, 3};
    double out1[kOutputElements] = {0};
    double out2[kOutputElements] = {0};
    double extra_out[1] = {0};
    int64_t size_split[kNumSplit] = {1, 1};
    int32_t split_dim = 0;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, {kNumSplit, kOutputElements}, input})
        .Input({"size_splits", DT_INT64, {kNumSplit}, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", kNumSplit)
        .Output({"y1", DT_DOUBLE, {1, kOutputElements}, out1})
        .Output({"y2", DT_DOUBLE, {1, kOutputElements}, out2})
        .Output({"extra_y", DT_DOUBLE, {1}, extra_out});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// A zero-length split writes nothing, so its data pointer may be null; the output tensor
// itself is still resolved and validated for every slot on the dim-0 path.
TEST_F(TEST_SPLITV_UT, TestSplitV_DIM0_ZERO_SPLIT_NULL_DATA_OK)
{
    double input[4] = {0, 1, 2, 3};
    double out2[4] = {0};
    int64_t size_split[2] = {0, 2};
    vector<int64_t> input_shape = {2, 2};
    vector<int64_t> splits_shape = {2};
    int32_t split_dim = 0;
    double* null_out = nullptr;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_DOUBLE, {0, 2}, null_out})
        .Output({"y2", DT_DOUBLE, {2, 2}, out2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    double expect2[4] = {0, 1, 2, 3};
    EXPECT_EQ(CompareResult<double>(out2, expect2, 4), true);
}
TEST_F(TEST_SPLITV_UT, TestSplitV_TWO_MINUS_ONE_FAILS)
{
    float input[12] = {0};
    float out1[4] = {0};
    float out2[8] = {0};
    int64_t size_split[2] = {-1, -1};
    vector<int64_t> input_shape = {3, 4};
    vector<int64_t> splits_shape = {2};
    int32_t split_dim = 0;

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_FLOAT, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_FLOAT, {1, 4}, out1})
        .Output({"y2", DT_FLOAT, {2, 4}, out2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_SIZE_SPLITS_SUM_OVERFLOW_FAILS)
{
    const int64_t signed_max = std::numeric_limits<int64_t>::max();
    int64_t input[2] = {0, 1};
    int64_t out1[1] = {0};
    int64_t out2[1] = {0};
    int64_t size_split[2] = {signed_max, 1};
    int32_t split_dim = 0;
    std::vector<int64_t> input_shape = {2};
    std::vector<int64_t> splits_shape = {2};

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_INT64, input_shape, input})
        .Input({"size_splits", DT_INT64, splits_shape, size_split})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", 2)
        .Output({"y1", DT_INT64, {1}, out1})
        .Output({"y2", DT_INT64, {1}, out2});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPLITV_UT, TestSplitV_DIM0_GATHER_BOUNDARY_SUCCESS)
{
    constexpr int64_t kGatherCapacity = 64;
    constexpr int64_t kOutputCount = kGatherCapacity + 1;
    constexpr int64_t kColumns = 2;
    constexpr size_t kGuardElements = 1;
    constexpr double kOutputGuard = -17.0;
    std::vector<double> input(kOutputCount * kColumns);
    std::vector<int64_t> size_splits(kOutputCount, 1);
    std::vector<std::vector<double>> outputs(kOutputCount,
                                             std::vector<double>(kColumns + kGuardElements, kOutputGuard));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>(i);
    }
    int32_t split_dim = 0;
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder node(node_def.get(), "SplitV", "SplitV");
    node.Input({"x", DT_DOUBLE, {kOutputCount, kColumns}, input.data()})
        .Input({"size_splits", DT_INT64, {kOutputCount}, size_splits.data()})
        .Input({"split_dim", DT_INT32, {}, &split_dim})
        .Attr("num_split", kOutputCount);
    for (auto& output : outputs) {
        node.Output({"y", DT_DOUBLE, {1, kColumns}, output.data()});
    }
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    for (size_t row = 0; row < outputs.size(); ++row) {
        for (size_t col = 0; col < kColumns; ++col) {
            EXPECT_EQ(outputs[row][col], input[row * kColumns + col]);
        }
        EXPECT_EQ(outputs[row][kColumns], kOutputGuard);
    }
}
