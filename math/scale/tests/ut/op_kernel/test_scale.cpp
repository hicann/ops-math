/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/scale_tiling_struct.h"
#include "../../../op_kernel/arch35/scale_kernel.h"

#ifndef GET_TILING_DATA_WITH_STRUCT
#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)
#if defined(ASCENDC_CPU_DEBUG)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t* p, T* td)
{
    constexpr uint64_t sz = sizeof(T);
    constexpr uint32_t judge = sz > 15 ? sz - 15 : 0;
    uint32_t i = 0;
    if (judge > 0) {
        for (; i < judge; i += 16) {
            (*(uint64_t*)((uint8_t*)td + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p + i));
            (*(uint64_t*)((uint8_t*)td + i + 8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p + i + 8));
        }
    }
    if (sz & 0x08) {
        (*(uint64_t*)((uint8_t*)td + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p + i));
        i += 8;
    }
    if (sz & 0x04) {
        (*(uint32_t*)((uint8_t*)td + i)) = (*(const __gm__ uint32_t*)((const __gm__ uint8_t*)p + i));
        i += 4;
    }
    if (sz & 0x02) {
        (*(uint16_t*)((uint8_t*)td + i)) = (*(const __gm__ uint16_t*)((const __gm__ uint8_t*)p + i));
        i += 2;
    }
    if (sz & 0x01) {
        (*(uint8_t*)((uint8_t*)td + i)) = (*(const __gm__ uint8_t*)((const __gm__ uint8_t*)p + i));
    }
}
#endif
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                   \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);
#endif

void scale_float_rank4_no_bias(GM_ADDR x, GM_ADDR scale_in, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ScaleTilingData<4>, td, tiling);
    GM_ADDR ins[3] = {x, scale_in, bias};
    GM_ADDR outs[1] = {y};
    ScaleKernel<float, 4> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
}

void scale_float_rank4_with_bias(GM_ADDR x, GM_ADDR scale_in, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                 GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ScaleTilingData<4>, td, tiling);
    GM_ADDR ins[3] = {x, scale_in, bias};
    GM_ADDR outs[1] = {y};
    ScaleKernel<float, 4> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
}

void scale_float_rank8_coalesced(GM_ADDR x, GM_ADDR scale_in, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                 GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ScaleTilingData<8>, td, tiling);
    GM_ADDR ins[3] = {x, scale_in, bias};
    GM_ADDR outs[1] = {y};
    ScaleKernel<float, 8> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
}

class ScaleKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScaleKernelTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ScaleKernelTest TearDown" << std::endl; }
};

static void FillScaleTilingData4(ScaleTilingData<4>* td, bool hasBias)
{
    memset(td, 0, sizeof(ScaleTilingData<4>));
    td->split = {2, 3, 1, 3};
    td->multicore = {1, 1, 1, 0};
    td->rank = 2;
    td->per_buf_bytes = 15744;
    td->per_buf_elems = 3936;
    td->max_bro_shape[0] = 1;
    td->max_bro_shape[1] = 1;
    td->max_bro_shape[2] = 3;
    td->max_bro_shape[3] = 5;
    td->num_inputs = hasBias ? 3 : 2;
    td->num_outputs = 1;
    td->has_bias = hasBias ? 1 : 0;
    td->input_shapes[0][0] = 1;
    td->input_shapes[0][1] = 1;
    td->input_shapes[0][2] = 3;
    td->input_shapes[0][3] = 5;
    td->input_strides[0][0] = 0;
    td->input_strides[0][1] = 0;
    td->input_strides[0][2] = 5;
    td->input_strides[0][3] = 1;
    td->input_shapes[1][0] = 1;
    td->input_shapes[1][1] = 1;
    td->input_shapes[1][2] = 1;
    td->input_shapes[1][3] = 5;
    td->input_strides[1][0] = 0;
    td->input_strides[1][1] = 0;
    td->input_strides[1][2] = 0;
    td->input_strides[1][3] = 1;
    if (hasBias) {
        td->input_shapes[2][0] = 1;
        td->input_shapes[2][1] = 1;
        td->input_shapes[2][2] = 1;
        td->input_shapes[2][3] = 5;
        td->input_strides[2][0] = 0;
        td->input_strides[2][1] = 0;
        td->input_strides[2][2] = 0;
        td->input_strides[2][3] = 1;
    }
    td->output_shapes[0][0] = 1;
    td->output_shapes[0][1] = 1;
    td->output_shapes[0][2] = 3;
    td->output_shapes[0][3] = 5;
    td->output_strides[0][0] = 0;
    td->output_strides[0][1] = 0;
    td->output_strides[0][2] = 5;
    td->output_strides[0][3] = 1;
}

TEST_F(ScaleKernelTest, test_float_no_bias_rank4)
{
    constexpr int64_t N = 15;
    constexpr size_t ELEM_SIZE = sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* scale_in = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);

    float* xF = reinterpret_cast<float*>(x);
    float* scF = reinterpret_cast<float*>(scale_in);
    float* yF = reinterpret_cast<float*>(y);
    for (int i = 0; i < N; i++)
        xF[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 5; i++)
        scF[i] = 2.0f;
    memset(bias, 0, N * ELEM_SIZE);
    memset(y, 0, N * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScaleTilingData<4>));

    ScaleTilingData<4>* td = reinterpret_cast<ScaleTilingData<4>*>(tiling);
    FillScaleTilingData4(td, false);

    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scale_float_rank4_no_bias, 1, x, scale_in, bias, y, workspace, tiling);

    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i + 1) * 2.0f;
        EXPECT_FLOAT_EQ(yF[i], expected);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale_in);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ScaleKernelTest, test_float_with_bias_rank4)
{
    constexpr int64_t N = 15;
    constexpr size_t ELEM_SIZE = sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* scale_in = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);

    float* xF = reinterpret_cast<float*>(x);
    float* scF = reinterpret_cast<float*>(scale_in);
    float* biF = reinterpret_cast<float*>(bias);
    float* yF = reinterpret_cast<float*>(y);
    for (int i = 0; i < N; i++)
        xF[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 5; i++)
        scF[i] = 2.0f;
    for (int i = 0; i < 5; i++)
        biF[i] = 1.0f;
    memset(y, 0, N * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScaleTilingData<4>));

    ScaleTilingData<4>* td = reinterpret_cast<ScaleTilingData<4>*>(tiling);
    FillScaleTilingData4(td, true);

    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scale_float_rank4_with_bias, 1, x, scale_in, bias, y, workspace, tiling);

    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i + 1) * 2.0f + 1.0f;
        EXPECT_FLOAT_EQ(yF[i], expected);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale_in);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// rank8 大 shape 场景回归：x=(17,2,2,2,2,2)（rank6→RANK8 前补 2 维），scale/bias=(17,)
// split 轴落在 NDDMA 5 维窗口外，验证 run 合并后整块搬运的正确性（含尾段 a_i_tail）
static void FillScaleTilingData8(ScaleTilingData<8>* td)
{
    memset(td, 0, sizeof(ScaleTilingData<8>));
    td->split = {2, 4, 5, 1}; // axis=2(前补2), a_i=4, a_o=5, a_i_tail=1
    td->multicore = {1, 5, 5, 0};
    td->rank = 6;
    td->per_buf_bytes = 512;
    td->per_buf_elems = 128;
    td->max_bro_shape[0] = 1;
    td->max_bro_shape[1] = 1;
    td->max_bro_shape[2] = 17;
    td->max_bro_shape[3] = 2;
    td->max_bro_shape[4] = 2;
    td->max_bro_shape[5] = 2;
    td->max_bro_shape[6] = 2;
    td->max_bro_shape[7] = 2;
    td->num_inputs = 3;
    td->num_outputs = 1;
    td->has_bias = 1;
    // x: 稠密 (17,2,2,2,2,2)
    td->input_shapes[0][0] = 1;
    td->input_shapes[0][1] = 1;
    td->input_shapes[0][2] = 17;
    td->input_shapes[0][3] = 2;
    td->input_shapes[0][4] = 2;
    td->input_shapes[0][5] = 2;
    td->input_shapes[0][6] = 2;
    td->input_shapes[0][7] = 2;
    td->input_strides[0][0] = 0;
    td->input_strides[0][1] = 0;
    td->input_strides[0][2] = 32;
    td->input_strides[0][3] = 16;
    td->input_strides[0][4] = 8;
    td->input_strides[0][5] = 4;
    td->input_strides[0][6] = 2;
    td->input_strides[0][7] = 1;
    // scale: (17,1,1,1,1,1) 广播
    td->input_shapes[1][0] = 1;
    td->input_shapes[1][1] = 1;
    td->input_shapes[1][2] = 17;
    td->input_shapes[1][3] = 1;
    td->input_shapes[1][4] = 1;
    td->input_shapes[1][5] = 1;
    td->input_shapes[1][6] = 1;
    td->input_shapes[1][7] = 1;
    td->input_strides[1][0] = 0;
    td->input_strides[1][1] = 0;
    td->input_strides[1][2] = 1;
    td->input_strides[1][3] = 0;
    td->input_strides[1][4] = 0;
    td->input_strides[1][5] = 0;
    td->input_strides[1][6] = 0;
    td->input_strides[1][7] = 0;
    // bias: 同 scale
    td->input_shapes[2][0] = 1;
    td->input_shapes[2][1] = 1;
    td->input_shapes[2][2] = 17;
    td->input_shapes[2][3] = 1;
    td->input_shapes[2][4] = 1;
    td->input_shapes[2][5] = 1;
    td->input_shapes[2][6] = 1;
    td->input_shapes[2][7] = 1;
    td->input_strides[2][0] = 0;
    td->input_strides[2][1] = 0;
    td->input_strides[2][2] = 1;
    td->input_strides[2][3] = 0;
    td->input_strides[2][4] = 0;
    td->input_strides[2][5] = 0;
    td->input_strides[2][6] = 0;
    td->input_strides[2][7] = 0;
    // y: 同 x
    td->output_shapes[0][0] = 1;
    td->output_shapes[0][1] = 1;
    td->output_shapes[0][2] = 17;
    td->output_shapes[0][3] = 2;
    td->output_shapes[0][4] = 2;
    td->output_shapes[0][5] = 2;
    td->output_shapes[0][6] = 2;
    td->output_shapes[0][7] = 2;
    td->output_strides[0][0] = 0;
    td->output_strides[0][1] = 0;
    td->output_strides[0][2] = 32;
    td->output_strides[0][3] = 16;
    td->output_strides[0][4] = 8;
    td->output_strides[0][5] = 4;
    td->output_strides[0][6] = 2;
    td->output_strides[0][7] = 1;
}

TEST_F(ScaleKernelTest, test_float_rank8_split_axis_outside_nddma_window)
{
    // x=(17,2,2,2,2,2): y[n,a,b,c,d,e] = x * scale[n] + bias[n]
    constexpr int64_t N0 = 17;
    constexpr int64_t TOTAL = N0 * 32;
    constexpr size_t ELEM_SIZE = sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);
    uint8_t* scale_in = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);

    float* xF = reinterpret_cast<float*>(x);
    float* scF = reinterpret_cast<float*>(scale_in);
    float* biF = reinterpret_cast<float*>(bias);
    float* yF = reinterpret_cast<float*>(y);
    for (int64_t i = 0; i < TOTAL; i++)
        xF[i] = static_cast<float>(i % 97) - 10.0f;
    for (int64_t n = 0; n < N0; n++) {
        scF[n] = static_cast<float>(n) + 1.0f;
        biF[n] = 10.0f * static_cast<float>(n);
    }
    memset(y, 0, TOTAL * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScaleTilingData<8>));

    ScaleTilingData<8>* td = reinterpret_cast<ScaleTilingData<8>*>(tiling);
    FillScaleTilingData8(td);

    ICPU_SET_TILING_KEY(8);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scale_float_rank8_coalesced, 1, x, scale_in, bias, y, workspace, tiling);

    for (int64_t i = 0; i < TOTAL; i++) {
        int64_t n = i / 32; // 最外维 n，内侧 5 维共 32
        float expected = xF[i] * scF[n] + biF[n];
        EXPECT_FLOAT_EQ(yF[i], expected);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale_in);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// run 数超过 NDDMA 5 维上限的兜底路径：scale normal=(17,1,2,1,2,1) 产生 6 段稠密/广播交替
TEST_F(ScaleKernelTest, test_float_rank8_run_overflow_fallback)
{
    constexpr int64_t N0 = 17;
    constexpr int64_t TOTAL = N0 * 32;
    constexpr size_t ELEM_SIZE = sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);
    uint8_t* scale_in = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(TOTAL * ELEM_SIZE);

    float* xF = reinterpret_cast<float*>(x);
    float* scF = reinterpret_cast<float*>(scale_in);
    float* biF = reinterpret_cast<float*>(bias);
    float* yF = reinterpret_cast<float*>(y);
    for (int64_t i = 0; i < TOTAL; i++)
        xF[i] = static_cast<float>(i % 89) - 20.0f;
    // scale 稠密维为 0/2/4（normal (17,1,2,1,2,1)），flat 索引 = n*4 + b*2 + d
    for (int64_t n = 0; n < N0; n++) {
        for (int64_t b = 0; b < 2; b++) {
            for (int64_t d = 0; d < 2; d++) {
                int64_t fi = n * 4 + b * 2 + d;
                scF[fi] = static_cast<float>(fi) + 1.0f;
                biF[fi] = 0.5f * static_cast<float>(fi);
            }
        }
    }
    memset(y, 0, TOTAL * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScaleTilingData<8>));

    ScaleTilingData<8>* td = reinterpret_cast<ScaleTilingData<8>*>(tiling);
    FillScaleTilingData8(td);
    // 覆写 scale/bias 为 (17,1,2,1,2,1)：strides = (4,0,2,0,1,0)
    td->input_shapes[1][2] = 17;
    td->input_shapes[1][3] = 1;
    td->input_shapes[1][4] = 2;
    td->input_shapes[1][5] = 1;
    td->input_shapes[1][6] = 2;
    td->input_shapes[1][7] = 1;
    td->input_strides[1][2] = 4;
    td->input_strides[1][3] = 0;
    td->input_strides[1][4] = 2;
    td->input_strides[1][5] = 0;
    td->input_strides[1][6] = 1;
    td->input_strides[1][7] = 0;
    td->input_shapes[2][2] = 17;
    td->input_shapes[2][3] = 1;
    td->input_shapes[2][4] = 2;
    td->input_shapes[2][5] = 1;
    td->input_shapes[2][6] = 2;
    td->input_shapes[2][7] = 1;
    td->input_strides[2][2] = 4;
    td->input_strides[2][3] = 0;
    td->input_strides[2][4] = 2;
    td->input_strides[2][5] = 0;
    td->input_strides[2][6] = 1;
    td->input_strides[2][7] = 0;

    ICPU_SET_TILING_KEY(8);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scale_float_rank8_coalesced, 1, x, scale_in, bias, y, workspace, tiling);

    // y[n,a,b,c,d,e] = x * scale[n,b,d] + bias[n,b,d]
    for (int64_t n = 0; n < N0; n++) {
        for (int64_t a = 0; a < 2; a++) {
            for (int64_t b = 0; b < 2; b++) {
                for (int64_t c = 0; c < 2; c++) {
                    for (int64_t d = 0; d < 2; d++) {
                        for (int64_t e = 0; e < 2; e++) {
                            int64_t i = ((((n * 2 + a) * 2 + b) * 2 + c) * 2 + d) * 2 + e;
                            int64_t fi = n * 4 + b * 2 + d;
                            float expected = xF[i] * scF[fi] + biF[fi];
                            EXPECT_FLOAT_EQ(yF[i], expected);
                        }
                    }
                }
            }
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale_in);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
