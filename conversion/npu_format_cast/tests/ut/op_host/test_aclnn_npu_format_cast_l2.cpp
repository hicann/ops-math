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
#include "acl/acl.h"
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include <thread>
#include <vector>

#include "../../../op_host/op_api/aclnn_npu_format_cast.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "ut_stub.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
struct NpuFormatCastTestParam {
    string caseName;
    vector<int64_t> srcTensor;
    vector<int64_t> storageShape;
    aclDataType srcDtype;
    aclFormat srcFormat;
    aclFormat dstFormat;
    int additionalDtype;
    // out
    aclnnStatus expectRet;
};

class l2_npu_format_cast_test_910_95 : public testing::TestWithParam<NpuFormatCastTestParam> {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_npu_format_cast_test_910_95 SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_npu_format_cast_test_910_95 S" << endl;
    }
};

class l2_npu_format_cast_test_910_93 : public testing::TestWithParam<NpuFormatCastTestParam> {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_npu_format_cast_test_910_93 SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_npu_format_cast_test_910_93 S" << endl;
    }
};

static void TestOneParamCase(const NpuFormatCastTestParam& param)
{
    std::cout << "run case start: " << param.caseName << std::endl;
    void* srcDeviceAddr = nullptr;
    void* dstDeviceAddr = nullptr;
    std::vector<int64_t> strides(param.srcTensor.size(), 1);
    for (int64_t i = param.srcTensor.size() - 2; i >= 0; i--) {
        strides[i] = param.srcTensor[i + 1] * strides[i + 1];
    }

    aclTensor* srctensor = aclCreateTensor(
        param.srcTensor.data(), param.srcTensor.size(), param.srcDtype, strides.data(), 0, param.srcFormat,
        param.storageShape.data(), param.storageShape.size(), srcDeviceAddr);

    const int AdditionalDtype = param.additionalDtype;
    int dstFormat = static_cast<int>(param.dstFormat);
    int64_t* DstShape = nullptr;
    uint64_t DstShapeSize = 0;
    int ActualFormat = 0;

    // dstformat仅支持FRACTAL_NZ, 用29表示
    auto ret = aclnnNpuFormatCastCalculateSizeAndFormat(
        srctensor, dstFormat, AdditionalDtype, &DstShape, &DstShapeSize,
        &ActualFormat); // 参数待修改

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
        EXPECT_EQ(ret, param.expectRet);
        return;
    }
    aclTensor* dsttensor = aclCreateTensor(
        param.srcTensor.data(), param.srcTensor.size(), param.srcDtype, strides.data(), 0,
        static_cast<aclFormat>(ActualFormat), DstShape, DstShapeSize, dstDeviceAddr);

    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream;
    delete[] DstShape;

    // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
    ret = aclnnNpuFormatCastGetWorkspaceSize(srctensor, dsttensor, &workspaceSize, &exe);
    std::cout << "aclnnNpuFormatCastGetWorkspaceSize returned: " << ret << std::endl;

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret);
        EXPECT_EQ(ret, param.expectRet);
        return;
    }
    // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, exe, stream);
    std::cout << "aclnnNpuFormatCast returned: " << ret << std::endl;

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret);
        EXPECT_EQ(ret, param.expectRet);
        return;
    }

    std::cout << "run case end:" << param.caseName << std::endl;
    if (param.expectRet == ACLNN_SUCCESS) {
        EXPECT_NE(exe, nullptr);
    }
}

TEST_P(l2_npu_format_cast_test_910_95, ascend910_95_generalTest)
{
    NpuFormatCastTestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_npu_format_cast_test_910_93, ascend910_93_generalTest)
{
    NpuFormatCastTestParam param = GetParam();
    TestOneParamCase(param);
}

static NpuFormatCastTestParam casesParamsAscend910_95[] = {
    // casename srcTensor srcDtype srcFormat dstFormat additionalDtype expectOut
    {"ascend910_95_test_NpuformatCast_NZ_Align",
     {64, 128},
     {64, 128},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NCL_Align",
     {64, 128},
     {64, 128},
     ACL_INT8,
     ACL_FORMAT_NCL,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_SUCCESS}, // NCL
    {"ascend910_95_test_NpuformatCast_NZ_Align_fp16",
     {64, 128},
     {64, 128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FLOAT16,
     ACLNN_SUCCESS}, // 非量化FP16
    {"ascend910_95_test_NpuformatCast_NZ_Align_bf16",
     {64, 128},
     {64, 128},
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_BF16,
     ACLNN_SUCCESS}, // 非量化BF16
    {"ascend910_95_test_NpuformatCast_NZC016_Align_0",
     {64, 128},
     {64, 128},
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FLOAT16,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NZC016_Align_1",
     {64, 128},
     {64, 128},
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_BF16,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NZC016_NoAlign",
     {12, 31},
     {12, 31},
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FLOAT16,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NZC016_NoAlign",
     {12, 31},
     {12, 31},
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_BF16,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NZ_NoAlign",
     {35, 42},
     {35, 42},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NCL2NZ",
     {4, 64, 64},
     {4, 64, 64},
     ACL_FLOAT,
     ACL_FORMAT_NCL,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FLOAT16,
     ACLNN_SUCCESS},
    {"ascend910_95_test_NpuformatCast_NCL2NZ_NoAlign",
     {4, 15, 64},
     {4, 15, 64},
     ACL_FLOAT,
     ACL_FORMAT_NCL,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_BF16,
     ACLNN_SUCCESS},
    // 异常用例
    {"ascend910_95_test_NpuformatCast_InvalidAdditionalDtype",
     {64, 128},
     {64, 128},
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_NpuformatCast_InvalidsrcFormat",
     {16, 32},
     {16, 32},
     ACL_INT8,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_ERR_RUNTIME_ERROR},
    {"ascend910_95_test_NpuformatCast_InvalidsrcTensor",
     {32},
     {32},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_NpuformatCast_InvalidsrcDtype",
     {32, 64},
     {32, 64},
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_NpuformatCast_InvalidsrcDtype_uint8",
     {32, 64},
     {32, 64},
     ACL_UINT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT8,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_NpuformatCast_InvalidsrcDtype_fp32",
     {16, 64},
     {16, 64},
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FLOAT,
     ACLNN_ERR_PARAM_INVALID}, // 拦截非量化 fp32
    {"ascend910_95_test_NpuformatCast_InvalidsrcDtype_k_equal_1",
     {1, 64},
     {1, 64},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FLOAT16,
     ACLNN_ERR_PARAM_INVALID}, // 拦截非量化 k=1
    {"ascend910_95_test_NpuformatCast_InvalidsrcDtype",
     {16, 64},
     {16, 64},
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_INT16,
     ACLNN_ERR_PARAM_INVALID},
};

static NpuFormatCastTestParam casesParamsAscend910_93[] = {
    // casename srcTensor srcDtype srcFormat dstFormat additionalDtype expectOut
    {"ascend910_93_test_NpuformatCast_ND_NZ_Align",
     {64, 128},
     {64, 128},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_SUCCESS},
    {"ascend910_93_test_NpuformatCast_ND_NZ_Align_fp16",
     {64, 128},
     {64, 128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_SUCCESS}, // 非量化FP16
    {"ascend910_93_test_NpuformatCast_ND_NZ_Align_bf16",
     {64, 128},
     {64, 128},
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_SUCCESS}, // 非量化BF16
    {"ascend910_93_test_NpuformatCast_ND_NZ_NoAlign",
     {35, 42},
     {35, 42},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_SUCCESS},

    {"ascend910_93_test_NpuformatCast_NZ_ND_Align",
     {64, 128},
     {64, 128},
     ACL_INT8,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_ND,
     -1,
     ACLNN_SUCCESS},
    {"ascend910_93_test_NpuformatCast_NCDHW_NDC1HWC0_Align",
     {1, 18, 1, 2, 2},
     {1, 18, 1, 2, 2},
     ACL_INT8,
     ACL_FORMAT_NCDHW,
     ACL_FORMAT_NDC1HWC0,
     -1,
     ACLNN_SUCCESS},
    {"ascend910_93_test_NpuformatCast_NDC1HWC0_NCDHW_Align",
     {1, 17, 1, 2, 2},
     {1, 1, 3, 2, 2, 8},
     ACL_INT8,
     ACL_FORMAT_NDC1HWC0,
     ACL_FORMAT_NCDHW,
     -1,
     ACLNN_SUCCESS},
    {"ascend910_93_test_NpuformatCast_NCDHW_Z3D_Align",
     {1, 18, 1, 2, 2},
     {1, 18, 1, 2, 2},
     ACL_INT8,
     ACL_FORMAT_NCDHW,
     ACL_FRACTAL_Z_3D,
     -1,
     ACLNN_SUCCESS},
    {"ascend910_93_test_NpuformatCast_Z3D_NCDHW_Align",
     {1, 18, 1, 2, 2},
     {8, 1, 16, 8},
     ACL_INT8,
     ACL_FRACTAL_Z_3D,
     ACL_FORMAT_NCDHW,
     -1,
     ACLNN_SUCCESS},

    // 异常用例
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidAdditionalDtype",
     {64, 128},
     {64, 128},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_UINT8,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcFormat",
     {16, 32},
     {16, 32},
     ACL_INT8,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_RUNTIME_ERROR},
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcTensor",
     {32},
     {32},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcDtype",
     {32, 64},
     {32, 64},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcDtype_uint8",
     {32, 64},
     {32, 64},
     ACL_UINT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_PARAM_INVALID},
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcDtype_fp32",
     {16, 64},
     {16, 64},
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_PARAM_INVALID}, // 拦截非量化 fp32
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcDtype_k_equal_1",
     {1, 64},
     {1, 64},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_PARAM_INVALID}, // 拦截非量化 k=1
    {"ascend910_93_test_NpuformatCast_ND_NZ_InvalidsrcDtype",
     {16, 64},
     {16, 64},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     -1,
     ACLNN_ERR_PARAM_INVALID},
};

INSTANTIATE_TEST_SUITE_P(
    Ascend910_95_NpuFormatCast, l2_npu_format_cast_test_910_95, testing::ValuesIn(casesParamsAscend910_95));

INSTANTIATE_TEST_SUITE_P(
    Ascend910_93_NpuFormatCast, l2_npu_format_cast_test_910_93, testing::ValuesIn(casesParamsAscend910_93));

static void ThreadFunc(const NpuFormatCastTestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const NpuFormatCastTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(l2_npu_format_cast_test_910_95, ascend910_95_multi_thread)
{
    TestMultiThread(casesParamsAscend910_95, sizeof(casesParamsAscend910_95) / sizeof(NpuFormatCastTestParam), 3);
}

TEST_F(l2_npu_format_cast_test_910_93, ascend910_93_multi_thread)
{
    TestMultiThread(casesParamsAscend910_93, sizeof(casesParamsAscend910_93) / sizeof(NpuFormatCastTestParam), 3);
}