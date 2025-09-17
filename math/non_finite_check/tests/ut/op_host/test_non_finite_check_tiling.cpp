/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
 /*!
 * \file non_finite_check.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "op_log.h"

#include "runtime2_util.h"
#include "kernel_run_context_facker.h"
#include "experiment_ops.h"
#include "array_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "../../../op_host/non_finite_check_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

class NonFiniteCheckTiling : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        cout << "NonFiniteCheckTiling SetUpTestSuite" << endl;
        compileInfoStr = R"({"hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 48
                                        }})";

        GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
        platformInfo.Init();

        string opType("NonFiniteCheck");
        ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
        tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
        tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    }

    static void TearDownTestSuite()
    {
        cout << "NonFiniteCheckTiling TearDownTestSuite" << endl;
    }

protected:
    static map<string, string> socInfos;
    static map<string, string> aicoreSpec;
    static map<string, string> intrinsics;
    static string compileInfoStr;
    // platform info
    static fe::PlatFormInfos platformInfo;
    // compile info
    static optiling::NonFiniteCheckCompileInfo compileInfo;
    static gert::OpImplKernelRegistry::KernelFunc tilingParseFunc;
    static gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc;
};

map<string, string> NonFiniteCheckTiling::socInfos;
map<string, string> NonFiniteCheckTiling::aicoreSpec;
map<string, string> NonFiniteCheckTiling::intrinsics;
string NonFiniteCheckTiling::compileInfoStr;
// platform info
fe::PlatFormInfos NonFiniteCheckTiling::platformInfo;
// compile info
optiling::NonFiniteCheckCompileInfo NonFiniteCheckTiling::compileInfo;
gert::OpImplKernelRegistry::KernelFunc NonFiniteCheckTiling::tilingParseFunc = nullptr;
gert::OpImplKernelRegistry::TilingKernelFunc NonFiniteCheckTiling::tilingFunc = nullptr;

TEST_F(NonFiniteCheckTiling, non_finite_check_tiling_parse_func)
{
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(NonFiniteCheckTiling, non_finite_check_tiling_fp16_001)
{
    gert::StorageShape tensorShape1 = {{2, 3}, {2, 3}};
    gert::StorageShape tensorShape2 = {{4, 7}, {4, 7}};
    gert::StorageShape tensorShape3 = {{32, 16}, {32, 16}};
    gert::StorageShape outShape = {{1}, {1}};
    compileInfo = {48, 196608};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({3})
                            .InputShapes({&tensorShape1, &tensorShape2, &tensorShape3})
                            .OutputShapes({&outShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 101);
}

TEST_F(NonFiniteCheckTiling, non_finite_check_tiling_fp16_002)
{
    gert::StorageShape tensorShape1 = {{2, 3}, {2, 3}};
    gert::StorageShape tensorShape2 = {{4, 7}, {4, 7}};
    gert::StorageShape tensorShape3 = {{32, 16}, {32, 16}};
    gert::StorageShape outShape = {{1}, {1}};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({3})
                            .InputShapes({&tensorShape1, &tensorShape2, &tensorShape3})
                            .OutputShapes({&outShape})
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 101);
}

TEST_F(NonFiniteCheckTiling, non_finite_check_tiling_bfp16_001)
{
    gert::StorageShape tensorShape1 = {{2, 3}, {2, 3}};
    gert::StorageShape tensorShape2 = {{4, 7}, {4, 7}};
    gert::StorageShape tensorShape3 = {{32, 16}, {32, 16}};
    gert::StorageShape outShape = {{1}, {1}};
    compileInfo = {48, 196608};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({3})
                            .InputShapes({&tensorShape1, &tensorShape2, &tensorShape3})
                            .OutputShapes({&outShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 201);
}

TEST_F(NonFiniteCheckTiling, non_finite_check_tiling_fp32_001)
{
    gert::StorageShape tensorShape1 = {{2, 3}, {2, 3}};
    gert::StorageShape tensorShape2 = {{4, 7}, {4, 7}};
    gert::StorageShape tensorShape3 = {{32, 16}, {32, 16}};
    gert::StorageShape outShape = {{1}, {1}};
    compileInfo = {48, 196608};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({3})
                            .InputShapes({&tensorShape1, &tensorShape2, &tensorShape3})
                            .OutputShapes({&outShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 301);
}

TEST_F(NonFiniteCheckTiling, non_finite_check_tiling_fp32_002)
{
    gert::StorageShape tensorShape1 = {{2, 3}, {2, 3}};
    gert::StorageShape tensorShape2 = {{4, 7}, {4, 7}};
    gert::StorageShape outShape = {{1}, {1}};
    compileInfo = {48, 196608};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({2})
                            .InputShapes({&tensorShape1, &tensorShape2})
                            .OutputShapes({&outShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}
