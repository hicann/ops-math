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

/*!
 * \file non_finite_check.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tensor_list_operate.h"
#include "tiling_function_def.h"
#include "../../../op_host/non_finite_check_tiling.h"
#include "data_utils.h"

using namespace NonFiniteCheckTest;

extern "C" __global__ __aicore__ void non_finite_check(
    GM_ADDR tensor_list, GM_ADDR found_flag, GM_ADDR workspace, GM_ADDR tiling);

class non_finite_check_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "non_finite_check_test SetUp\n" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "non_finite_check_test TearDown\n" << std::endl;
    }

    std::string GetShapesString(const std::vector<std::vector<uint64_t>>& shapeInfo)
    {
        std::string ret = "{";
        for (auto shape : shapeInfo) {
            ret += "{";
            for (auto dim : shape) {
                ret += std::to_string(dim) + ",";
            }
            ret += "},";
        }
        return ret + "}";
    }

    template <typename T>
    void SingleCallOperator(
        const std::vector<std::vector<uint64_t>>& shapeInfos, const ge::DataType dyDtype, const float expectRet = 1.0)
    {
        // get tiling data and tiling key
        NonFiniteCheckTiling tilingObject;
        tilingObject.Init(shapeInfos, dyDtype);
        uint32_t needBlockNum = tilingObject.RunBigKernelTiling();
        uint64_t tilingKey = tilingObject.GetTilingKeyVal();

        system(
            "cp -rf "
            "../../../../math/non_finite_check/tests/ut/op_kernel/non_finite_check_data ./");
        system("chmod -R 755 ./non_finite_check_data/ && rm -rf ./non_finite_check_data/*bin");
        std::string genCMD = "cd ./non_finite_check_data/ && python3 gen_data.py '" + GetShapesString(shapeInfos) +
                             "' " + std::to_string(tilingKey);
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
        size_t tilingSize = sizeof(NonFiniteCheckTilingData);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
        tilingObject.FillTilingData(reinterpret_cast<NonFiniteCheckTilingData*>(tiling));

        uint8_t* tensor_list = CreateTensorList<T>(shapeInfos);
        uint8_t* found_flag = (uint8_t*)AscendC::GmAlloc(sizeof(float));
        *((float*)found_flag) = 3.1415926;

        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(non_finite_check, needBlockNum, tensor_list, found_flag, workspace, tiling);
        float result = *((float*)found_flag);
        FreeTensorList<T>(tensor_list, shapeInfos);
        AscendC::GmFree((void*)found_flag);
        AscendC::GmFree((void*)workspace);
        AscendC::GmFree((void*)tiling);
        EXPECT_EQ(result, expectRet);
    }
};

TEST_F(non_finite_check_test, test_case_bfloat16_key_201)
{
    std::vector<std::vector<uint64_t>> shapeInfos({{3, 23}, {5, 7}});
    SingleCallOperator<bfloat16_t>(shapeInfos, ge::DT_BF16, 0.0);
}

TEST_F(non_finite_check_test, test_case_float_key_301)
{
    std::vector<std::vector<uint64_t>> shapeInfos({{3, 23}, {5, 7}});
    SingleCallOperator<float>(shapeInfos, ge::DT_FLOAT, 0.0);
}