/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "op_log.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "common_unittest.h"
#include "runtime/diag_util.h"
#include "../../../op_host/ring_attention_update_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "experiment_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class RingAttentionUpdateTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RingAttentionUpdateTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RingAttentionUpdateTiling TearDown" << std::endl;
  }
};

TEST_F(RingAttentionUpdateTiling, ring_attention_update_tiling_001) {
  dlog_setlevel(0,0,0);
  std::string input_layout = "SBH";
  gert::StorageShape attn_shape_0 = {{1024, 2, 1536}, {1024, 2, 1536}};
  gert::StorageShape attn_shape_1 = {{1024, 2, 1536}, {1024, 2, 1536}};
  gert::StorageShape attn_shape_2 = {{1024, 2, 1536}, {1024, 2, 1536}};
  gert::StorageShape softmax_shape_0 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_1 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_2 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_3 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_4 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_5 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};


  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct RingAttentionUpdateCompileInfo {};
  RingAttentionUpdateCompileInfo compile_info;

  std::string op_type("RingAttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(6, 3)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1})
                    .InputShapes({&attn_shape_0, &softmax_shape_0, &softmax_shape_1,
                                  &attn_shape_1, &softmax_shape_2, &softmax_shape_3})
                    .OutputShapes({&attn_shape_2, &softmax_shape_4, &softmax_shape_5})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"input_layout", ge::AnyValue::CreateFrom<string>(input_layout)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0,3,0);
}

TEST_F(RingAttentionUpdateTiling, ring_attention_update_tiling_002) {
  dlog_setlevel(0,0,0);
  std::string input_layout = "SBH";
  gert::StorageShape attn_shape_0 = {{1024, 2, 1536}, {1024, 2, 1536}};
  gert::StorageShape attn_shape_1 = {{1024, 2, 1536}, {1024, 2, 1536}};
  gert::StorageShape attn_shape_2 = {{1024, 2, 1536}, {1024, 2, 1536}};
  gert::StorageShape softmax_shape_0 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_1 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_2 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_3 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_4 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_5 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};
  gert::StorageShape softmax_shape_6 = {{2, 12, 1024, 8}, {2, 12, 1024, 8}};


  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct RingAttentionUpdateCompileInfo {};
  RingAttentionUpdateCompileInfo compile_info;

  std::string op_type("RingAttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(7, 3)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                    .InputShapes({&attn_shape_0, &softmax_shape_0, &softmax_shape_1,
                                  &attn_shape_1, &softmax_shape_2, &softmax_shape_3, &softmax_shape_6})
                    .OutputShapes({&attn_shape_2, &softmax_shape_4, &softmax_shape_5})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"input_layout", ge::AnyValue::CreateFrom<string>(input_layout)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0,3,0);
}

TEST_F(RingAttentionUpdateTiling, ring_attention_update_tiling_003) {
  dlog_setlevel(0,0,0);
  std::string input_layout = "TND";
  gert::StorageShape attn_shape_0 = {{3072, 12, 128}, {3072, 12, 128}};
  gert::StorageShape attn_shape_1 = {{3072, 12, 128}, {3072, 12, 128}};
  gert::StorageShape attn_shape_2 = {{3072, 12, 128}, {3072, 12, 128}};
  gert::StorageShape softmax_shape_0 = {{3072, 12, 8}, {3072, 12, 8}};
  gert::StorageShape softmax_shape_1 = {{3072, 12, 8}, {3072, 12, 8}};
  gert::StorageShape softmax_shape_2 = {{3072, 12, 8}, {3072, 12, 8}};
  gert::StorageShape softmax_shape_3 = {{3072, 12, 8}, {3072, 12, 8}};
  gert::StorageShape softmax_shape_4 = {{3072, 12, 8}, {3072, 12, 8}};
  gert::StorageShape softmax_shape_5 = {{3072, 12, 8}, {3072, 12, 8}};
  gert::StorageShape softmax_shape_6 = {{3073, }, {3073, }};


  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct RingAttentionUpdateCompileInfo {};
  RingAttentionUpdateCompileInfo compile_info;

  std::string op_type("RingAttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(7, 3)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                    .InputShapes({&attn_shape_0, &softmax_shape_0, &softmax_shape_1,
                                  &attn_shape_1, &softmax_shape_2, &softmax_shape_3, &softmax_shape_6})
                    .OutputShapes({&attn_shape_2, &softmax_shape_4, &softmax_shape_5})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"input_layout", ge::AnyValue::CreateFrom<string>(input_layout)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0,3,0);
}

TEST_F(RingAttentionUpdateTiling, ring_attention_update_tiling_004) {
  dlog_setlevel(0,0,0);
  std::string input_layout = "TND";
  gert::StorageShape attn_shape_0 = {{3072, 65, 128}, {3072, 65, 128}};
  gert::StorageShape attn_shape_1 = {{3072, 65, 128}, {3072, 65, 128}};
  gert::StorageShape attn_shape_2 = {{3072, 65, 128}, {3072, 65, 128}};
  gert::StorageShape softmax_shape_0 = {{3072, 65, 8}, {3072, 65, 8}};
  gert::StorageShape softmax_shape_1 = {{3072, 65, 8}, {3072, 65, 8}};
  gert::StorageShape softmax_shape_2 = {{3072, 65, 8}, {3072, 65, 8}};
  gert::StorageShape softmax_shape_3 = {{3072, 65, 8}, {3072, 65, 8}};
  gert::StorageShape softmax_shape_4 = {{3072, 65, 8}, {3072, 65, 8}};
  gert::StorageShape softmax_shape_5 = {{3072, 65, 8}, {3072, 65, 8}};
  gert::StorageShape softmax_shape_6 = {{3073, }, {3073, }};


  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct RingAttentionUpdateCompileInfo {};
  RingAttentionUpdateCompileInfo compile_info;

  std::string op_type("RingAttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(7, 3)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                    .InputShapes({&attn_shape_0, &softmax_shape_0, &softmax_shape_1,
                                  &attn_shape_1, &softmax_shape_2, &softmax_shape_3, &softmax_shape_6})
                    .OutputShapes({&attn_shape_2, &softmax_shape_4, &softmax_shape_5})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"input_layout", ge::AnyValue::CreateFrom<string>(input_layout)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0,3,0);
}