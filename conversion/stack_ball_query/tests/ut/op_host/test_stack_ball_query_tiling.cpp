/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include "common/utils/ut_op_util.h"
#include "matrix_calculation_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "experiment_ops.h"
#include "selection_ops.h"
#include "test_common.h"
#include "common_unittest.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "conversion/stack_ball_query/op_host/stack_ball_query_tiling.h"
#include "experiment_ops.h"

using namespace std;
using namespace ge;

class TestStackBallQueryTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "test TestStackBallQueryTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "test TestStackBallQueryTiling TearDown" << std::endl;
    }
};

namespace {
DataType StringToDtype(std::string dtype_string)
{
    auto find_it = optiling::STR_TO_DATATYPE.find(dtype_string);
    if (find_it != optiling::STR_TO_DATATYPE.end()) {
        return find_it->second;
    }
    return ge::DT_FLOAT16;
}

void add_input_desc_by_idx(
    Operator& op, int64_t idx, std::vector<int64_t> input_shape, std::string data_dtype, Format format)
{
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    op_info->MutableInputDesc(idx)->SetShape(GeShape(input_shape));
    op_info->MutableInputDesc(idx)->SetOriginShape(GeShape(input_shape));
    op_info->MutableInputDesc(idx)->SetFormat(format);
    op_info->MutableInputDesc(idx)->SetOriginFormat(format);
    op_info->MutableInputDesc(idx)->SetDataType(StringToDtype(data_dtype));
}

void add_output_desc_by_idx(
    Operator& op, int64_t idx, std::vector<int64_t> input_shape, std::string data_dtype, Format format)
{
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    op_info->MutableOutputDesc(idx)->SetShape(GeShape(input_shape));
    op_info->MutableOutputDesc(idx)->SetOriginShape(GeShape(input_shape));
    op_info->MutableOutputDesc(idx)->SetFormat(format);
    op_info->MutableOutputDesc(idx)->SetOriginFormat(format);
    op_info->MutableOutputDesc(idx)->SetDataType(StringToDtype(data_dtype));
}

void run_parse_test(optiling::StackBallQueryCompileInfo& compile_info)
{
    std::string compile_info_string = R"({
       "hardware_info": {
           "BT_SIZE": 0,
           "load3d_constraints": "1",
           "Intrinsic_fix_pipe_l0c2out": false,
           "Intrinsic_data_move_l12ub": true,
           "Intrinsic_data_move_l0c2ub": true,
           "Intrinsic_data_move_out2l1_nd2nz": false,
           "UB_SIZE": 262144,
           "L2_SIZE": 33554432,
           "L1_SIZE": 524288,
           "L0A_SIZE": 65536,
           "L0B_SIZE": 65536,
           "L0C_SIZE": 131072,
           "CORE_NUM": 48}})";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    std::string op_type("StackBallQuery");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

void run_case()
{
    auto test_op = op::StackBallQuery("StackBallQuery");

    std::vector<int64_t> xyz_shape = {3, 20};
    std::vector<int64_t> center_xyz_shape = {10, 3};
    std::vector<int64_t> xyz_batch_cnt_shape = {2};
    std::vector<int64_t> center_xyz_batch_cnt_shape = {2};
    std::vector<int64_t> idx_shape = {10, 5};

    add_input_desc_by_idx(test_op, 0, xyz_shape, "float", FORMAT_ND);
    add_input_desc_by_idx(test_op, 1, center_xyz_shape, "float", FORMAT_ND);
    add_input_desc_by_idx(test_op, 2, xyz_batch_cnt_shape, "int32", FORMAT_ND);
    add_input_desc_by_idx(test_op, 3, center_xyz_batch_cnt_shape, "int32", FORMAT_ND);
    add_output_desc_by_idx(test_op, 0, idx_shape, "int32", FORMAT_ND);

    optiling::StackBallQueryCompileInfo compile_info;
    run_parse_test(compile_info);
    std::unique_ptr<uint8_t[]> tilingdata;
    // EXPECT_EQ(TilingTest(test_op, &compile_info, tilingdata), ge::GRAPH_SUCCESS);
    // gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    // ASSERT_NE(raw_tiling_data, nullptr);
}
} // namespace

TEST_F(TestStackBallQueryTiling, test_case_1)
{
    run_case();
}