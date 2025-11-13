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
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_diag_flat.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class diag_flat_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "diag_flat_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "diag_flat_test TearDown" << endl;
    }
};

// FLOAT tensor
TEST_F(diag_flat_test, case_1)
{
    auto inputx_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outputy_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t diagonal = 0;
    auto ut = OP_API_UT(
        aclnnDiagFlat, INPUT(inputx_tensor_desc, diagonal),
        OUTPUT(outputy_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}