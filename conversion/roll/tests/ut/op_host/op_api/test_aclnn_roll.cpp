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
 * \file test_roll.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_roll.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_roll_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "roll_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "roll_test TearDown" << std::endl;
    }

    void test_run(
        aclDataType test_dtype, aclFormat test_format, vector<int64_t> view_dims, vector<int64_t> shifts,
        vector<int64_t> dims, bool skip_precision = false)
    {
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format).ValueRange(-2, 2);
        auto out = TensorDesc(view_dims_final, test_dtype, test_format).Precision(0.00001, 0.00001);

        auto shifts_final = IntArrayDesc(shifts);
        auto dims_final = IntArrayDesc(dims);

        auto ut = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, dims_final), OUTPUT(out));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        if (!skip_precision)
            // ut.TestPrecision();
            return;
    }

    void test_run_invalid(
        aclDataType test_dtype, aclFormat test_format, vector<int64_t> view_dims, vector<int64_t> shifts,
        vector<int64_t> dims)
    {
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format).ValueRange(-2, 2);
        auto out = TensorDesc(view_dims_final, test_dtype, test_format).Precision(0.00001, 0.00001);

        auto shifts_final = IntArrayDesc(shifts);
        auto dims_final = IntArrayDesc(dims);

        auto ut = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, dims_final), OUTPUT(out));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }

    void test_run_nullptr(
        aclDataType test_dtype, aclFormat test_format, vector<int64_t> view_dims, vector<int64_t> shifts,
        vector<int64_t> dims)
    {
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format).ValueRange(-2, 2);
        auto out = TensorDesc(view_dims_final, test_dtype, test_format).Precision(0.00001, 0.00001);

        auto shifts_final = IntArrayDesc(shifts);
        auto dims_final = IntArrayDesc(dims);
        uint64_t workspaceSize = 0;

        auto ut = OP_API_UT(aclnnRoll, INPUT(nullptr, shifts_final, dims_final), OUTPUT(out));
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

        auto ut1 = OP_API_UT(aclnnRoll, INPUT(self, nullptr, dims_final), OUTPUT(out));
        aclnnStatus getWorkspaceResult1 = ut1.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult1, ACLNN_ERR_PARAM_NULLPTR);

        auto ut2 = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, nullptr), OUTPUT(out));
        aclnnStatus getWorkspaceResult2 = ut2.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult2, ACLNN_ERR_PARAM_NULLPTR);

        auto ut3 = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, dims_final), OUTPUT(nullptr));
        aclnnStatus getWorkspaceResult3 = ut3.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult3, ACLNN_ERR_PARAM_NULLPTR);
    }
};

///////////////////////////////////////
/////         检查输出种类         /////
///////////////////////////////////////

// 检查 float16   aicore
TEST_F(l2_roll_test, l2_roll_test_01)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 检查 bfloat16   aicore
TEST_F(l2_roll_test, ascend910B2_l2_roll_test_01)
{
    test_run(ACL_BF16, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 检查 float32   aicore
TEST_F(l2_roll_test, l2_roll_test_02)
{
    test_run(ACL_FLOAT, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 检查 int8   aicore
TEST_F(l2_roll_test, l2_roll_test_03)
{
    test_run(ACL_INT8, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 检查 uint8   aicore
TEST_F(l2_roll_test, l2_roll_test_04)
{
    test_run(ACL_UINT8, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 检查 int32   aicore
TEST_F(l2_roll_test, l2_roll_test_05)
{
    test_run(ACL_INT32, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 检查 uint32  异常： golden不支持uint32      aicore
TEST_F(l2_roll_test, l2_roll_test_06)
{
    test_run(ACL_UINT32, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2}, true);
}

// 检查 bool: 算子底层不支持，通过cast实现
TEST_F(l2_roll_test, l2_roll_test_07)
{
    test_run(ACL_BOOL, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

// 不支持INT16, INT64, BFLOAT16, DOUBLE, COMPLEX64, COMPLEX128   后续新增了INT64二进制，现在支持INT64
TEST_F(l2_roll_test, l2_roll_test_08)
{
    test_run(ACL_INT64, ACL_FORMAT_ND, {1, 5, 3, 4}, {-1, -1}, {1, 1});

    test_run_invalid(ACL_INT16, ACL_FORMAT_ND, {1, 5, 3, 4}, {-1, -1}, {1, 1});
    test_run_invalid(ACL_DOUBLE, ACL_FORMAT_ND, {1, 5, 3, 4}, {-1, -1}, {1, 1});
    test_run_invalid(ACL_COMPLEX64, ACL_FORMAT_ND, {1, 5, 3, 4}, {-1, -1}, {1, 1});
    test_run_invalid(ACL_COMPLEX128, ACL_FORMAT_ND, {1, 5, 3, 4}, {-1, -1}, {1, 1});
}

// self和out的dtype不一致
TEST_F(l2_roll_test, l2_roll_test_09)
{
    auto self = TensorDesc({10, 20}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({10, 20}, ACL_INT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    auto shifts_final = IntArrayDesc({});
    auto dims_final = IntArrayDesc({});

    auto ut = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, dims_final), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

///////////////////////////////////////
/////       检查支持format         /////
///////////////////////////////////////

// 支持所有数据格式
TEST_F(l2_roll_test, l2_roll_test_010)
{
    test_run(ACL_INT32, ACL_FORMAT_ND, {3, 4, 5, 6, 2}, {1, 3}, {2, 2});
    test_run(ACL_INT32, ACL_FORMAT_NCHW, {3, 4, 5, 6}, {1, 3}, {2, 2});
    test_run(ACL_INT32, ACL_FORMAT_NHWC, {3, 4, 5, 6}, {1, 3}, {2, 2});
    test_run(ACL_INT32, ACL_FORMAT_NDHWC, {3, 4, 5, 6, 2}, {1, 3}, {2, 2});
    test_run(ACL_INT32, ACL_FORMAT_NCDHW, {3, 4, 5, 6, 2}, {1, 3}, {2, 2});
    test_run(ACL_INT32, ACL_FORMAT_HWCN, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

///////////////////////////////////////
/////          检查维度数          /////
///////////////////////////////////////

// roll 支持输入shape为1维     aicore
TEST_F(l2_roll_test, l2_roll_test_11)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {10}, {5, -1, 9}, {0, 0, 0});
}

// roll 支持输入shape为2维   aicore
TEST_F(l2_roll_test, l2_roll_test_12)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {10, 8}, {5, -1, 9}, {1, 1, 0});
}

// roll 支持输入shape为3维   aicore
TEST_F(l2_roll_test, l2_roll_test_13)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {2, 3, 4}, {5, -1, 9}, {-2, 2, -1});
}

// roll 支持输入shape为4维   aicore
TEST_F(l2_roll_test, l2_roll_test_14)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {2, 3, 1, 2}, {3, -1, 2}, {-1, -2, -2});
}

// roll 支持输入shape为5维   aicore
TEST_F(l2_roll_test, l2_roll_test_15)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {2, 3, 2, 3, 4}, {-1, -1, -1, -1, -1}, {1, 2, 1, 2, 3});
}

// roll 支持输入shape为6维   aicore
TEST_F(l2_roll_test, l2_roll_test_16)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {3, 2, 3, 2, 1, 1}, {5, -1, 9, 2, 20}, {-2, 2, -1, 3, 1});
}

// roll 支持输入shape为7维   aicore
TEST_F(l2_roll_test, l2_roll_test_17)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {3, 2, 3, 2, 1, 1, 3}, {20, -100, 2}, {-1, -2, -4});
}

// roll 支持输入shape为8维   aicore
TEST_F(l2_roll_test, l2_roll_test_18)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {3, 2, 3, 2, 1, 1, 1, 3}, {-1, -1, -40, -30, -1}, {1, 4, 1, 2, 3});
}

// roll 不支持超过8维的tensor  aicore
TEST_F(l2_roll_test, l2_roll_test_19)
{
    test_run_invalid(
        ACL_FLOAT16, ACL_FORMAT_ND, {2, 3, 3, 1, 3, 2, 3, 3, 5, 1}, {-1, -1, -40, -30, -1}, {1, 4, 1, 2, 3});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {2, 2, 2, 2, 1, 3, 1, 1, 3}, {-1, -1, -40, -30, -1}, {1, 4, 1, 2, 3});
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

// 非连续tensor    aicore
TEST_F(l2_roll_test, l2_roll_test_20)
{
    vector<int64_t> view_dims = {5, 4};
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);
    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(300, 350);
    auto out = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    auto shifts_final = IntArrayDesc({10, -50, 5, 3});
    auto dims_final = IntArrayDesc({0, 1, -1, -2});

    auto ut = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, dims_final), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();

    self = TensorDesc(view_dims_final, ACL_INT8, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(300, 350);
    out = TensorDesc(view_dims_final, ACL_INT8, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    ut = OP_API_UT(aclnnRoll, INPUT(self, shifts_final, dims_final), OUTPUT(out));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

// 不支持空指针   aicore
TEST_F(l2_roll_test, l2_roll_test_21)
{
    test_run_nullptr(ACL_FLOAT16, ACL_FORMAT_ND, {3, 4, 5, 6}, {1, 3}, {2, 2});
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

// 空tensor   aicore
TEST_F(l2_roll_test, l2_roll_test_22)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {10, 0, 10}, {-1, -1}, {1, 2});
}

///////////////////////////////////////
/////      shifts, dims 参数      /////
///////////////////////////////////////

// 不能shifts和dims都为空
TEST_F(l2_roll_test, l2_roll_test_23)
{
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {10, 8, 5}, {}, {});
}

// shifts和dims的大小不一样
TEST_F(l2_roll_test, l2_roll_test_24)
{
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {10, 8, 3, 5}, {-1, -1, -40, -10, -1}, {1, 4, 1, 2});
    test_run_invalid(ACL_INT32, ACL_FORMAT_ND, {10, 8, 5}, {-1, -1}, {1});
}

// shifts可以为任意范围的数，dims必须为 [-N, N-1]之间
TEST_F(l2_roll_test, l2_roll_test_25)
{
    // shifts可以为任意范围的数
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {2, 8, 3, 5}, {-1, 0, -100000, 100004, -25331}, {1, 3, 1, 2, 0});

    // dims必须为 [-N, N-1之间]
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {2, 8, 3, 5}, {-1, 0, -100000, 100004, -25331}, {-4, -3, 0, 1, 3});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {2, 8, 3, 5}, {-1, -1, -1, -1}, {-5, 0, 0, 0});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {2, 8, 3, 5}, {-1, -1, -1, -1}, {4, 0, 0, 0});
}

// dims为空时, shifts为1才可以正常运行，其余报错
TEST_F(l2_roll_test, l2_roll_test_26)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {2, 8, 3, 5}, {99}, {});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {2, 8, 3, 5}, {99, 1}, {});
}

///////////////////////////////////////
/////     支持torch.tensor(a)     /////
///////////////////////////////////////

// torch.tensor(a) 必须要dims为size 0, shifts为size 1
TEST_F(l2_roll_test, l2_roll_test_27)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {}, {100}, {});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {}, {99}, {0});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {}, {99}, {-1});
}

///////////////////////////////////////
///// 支持emtpy时, dim超出合适范围  /////
///////////////////////////////////////

// torch.tensor([])   dims值为5，超出范围也能正常返回空tensor
TEST_F(l2_roll_test, l2_roll_test_28)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {10, 0}, {100}, {});
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, {10, 0}, {100, 5}, {50, -50});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {10, 0}, {}, {});
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_ND, {10, 0}, {10, 5}, {3});
}