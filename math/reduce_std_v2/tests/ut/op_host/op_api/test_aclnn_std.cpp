/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_std.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_std_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_std_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_std_test TearDown" << std::endl;
  }
 public:
  template <typename T>
  int CreateAclScalarTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(0, 0, dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), nullptr);
    return 0;
  }
};

// 正常场景_float32_nd_dim为0_keepdim为true
TEST_F(l2_std_test, normal_dtype_float32_format_nd_dim_0_keepdim_true) {
  auto selfDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{0});
  int64_t correction = 1;
  bool keepdim = true;
  auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// 正常场景_float32_nchw_dim为1_keepdim为false
TEST_F(l2_std_test, normal_dtype_float32_format_nchw_dim_1_keepdim_false) {
  auto selfDesc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{1});
  int64_t correction = 1;
  bool keepdim = false;
  auto outDesc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// 正常场景_float32_nhwc_dim为2_keepdim为false
TEST_F(l2_std_test, normal_dtype_float32_format_nhwc_dim_2_keepdim_false) {
  auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  auto dimDesc = IntArrayDesc(vector<int64_t>{2});
  int64_t correction = 1;
  bool keepdim = false;
  auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}
