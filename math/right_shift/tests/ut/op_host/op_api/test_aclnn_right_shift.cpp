/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "opdev/make_op_executor.h"
#include "level0/right_shift.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 24;

class RightShiftTest : public ::testing::Test {
 public:
 RightShiftTest() : exe(nullptr) {
  }

  aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype) {
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           data);
  }

  void Clear() {
    exe->kernelLaunchObjList_.clear();
  }

  void SetUp() override {
    auto executor = &exe;
    auto unique_executor = CREATE_EXECUTOR();
    unique_executor.ReleaseTo(executor);
  }

  void TearDown() override {
    delete exe;
    exe = nullptr;
  }

 public:
  aclOpExecutor* exe;
  int64_t data[DATA_SIZE] = {1};
};

TEST_F(RightShiftTest, RightShiftTest_SUCC) {
  auto x = CreateAclTensor({16}, ACL_INT32);
  auto y = CreateAclTensor({1}, ACL_INT32);
  auto z = l0op::RightShift(x, y, exe);
  ASSERT_NE(y, nullptr);
}

TEST_F(RightShiftTest, RightShiftTest_FAILED_1) {
    auto x = CreateAclTensor({16}, ACL_FLOAT);
    auto y = CreateAclTensor({1}, ACL_INT32);
    auto z = l0op::RightShift(x, y, exe);
    ASSERT_NE(y, nullptr);
}

TEST_F(RightShiftTest, RightShiftTest_FAILED_2) {
    auto x = CreateAclTensor({16}, ACL_INT32);
    auto y = CreateAclTensor({1}, ACL_FLOAT);
    auto z = l0op::RightShift(x, y, exe);
    ASSERT_NE(y, nullptr);
}

