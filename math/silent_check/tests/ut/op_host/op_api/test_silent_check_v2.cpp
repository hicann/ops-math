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

#include <gtest/gtest.h>
#include <iostream>

#include "opdev/make_op_executor.h"
#include "../../../../op_host/op_api/silent_check_v2.h"

const int64_t DATA_SIZE = static_cast<int64_t>(1024 * 1024);
class SilentCheckV2Test : public ::testing::Test {
public:
    SilentCheckV2Test() : exe(nullptr)
    {}
    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dataType)
    {
        return aclCreateTensor(
            shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(), data);
    }

    void Clear()
    {
        exe->kernelLaunchObjList_.clear();
    }

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override
    {
        delete exe;
        exe = nullptr;
    }

public:
    aclOpExecutor* exe;
    int64_t data[DATA_SIZE] = {0};
};

TEST_F(SilentCheckV2Test, SilentCheckV2_success)
{
    auto val = CreateAclTensor({1}, ACL_FLOAT);
    auto inputGrad = CreateAclTensor({4}, ACL_FLOAT);
    auto sfda = CreateAclTensor({3}, ACL_FLOAT);
    auto step = CreateAclTensor({3}, ACL_INT64);
    int32_t cMinSteps = 7;
    float cThreshL1 = 1000000;
    float cCoeffL1 = 100000;
    float cThreshL2 = 10000;
    float cCoeffL2 = 5000;
    int32_t npuAsdDetect = 1;
    auto out = l0op::SilentCheckV2(
        val, inputGrad, sfda, step, cMinSteps, cThreshL1, cCoeffL1, cThreshL2, cCoeffL2, npuAsdDetect, exe);
    ASSERT_NE(out, nullptr);
}