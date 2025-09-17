/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "op_log.h"
#include "../../../../math/grouped_bias_add_grad/op_host/grouped_bias_add_grad_tiling_def.h"
#include "../../../../math/grouped_bias_add_grad/op_host/grouped_bias_add_grad_tiling.h"
#include "array_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "experiment_ops.h"

using namespace std;

struct GroupedBiasAddGradData {
  // inputs
  bool exist_optional{true};
  gert::StorageShape grad_y_shape{{10, 32}, {10, 32}};
  gert::StorageShape group_idx_shape{{3}, {3}};

  // outputs
  gert::StorageShape grad_bias_shape{{3, 32}, {3, 32}};

  // data type
  ge::DataType data_type{ge::DT_FLOAT};
  ge::DataType data_type_optional{ge::DT_INT32};

  // test debug info
  string debug_info{"tiling_info:"};

  // expect
  ge::graphStatus expect_status{ge::GRAPH_FAILED};
  uint64_t expect_tiling_key{1000000};
  // others
  bool check_tiling_key{true};

  // attrs
  int64_t group_idx_type{0};
};

class TilingGroupedBiasAddGrad : public ::testing::TestWithParam<GroupedBiasAddGradData> {
 protected:
  void SetUp() override {
    std::cout << "TilingGroupedBiasAddGrad SetUp" << std::endl;
  }

  void TearDown() override {
    std::cout << "TilingGroupedBiasAddGrad TearDown" << std::endl;
  }
};

TEST_P(TilingGroupedBiasAddGrad, grouped_bias_add_grad_tiling) {
  string compile_info_string = R"({
              "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
              "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
              "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
              "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
              "CORE_NUM": 48}
              })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  optiling::GroupedBiasAddGradCompileInfo compile_info;

  string op_type("GroupedBiasAddGrad");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
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
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  auto test_params = GetParam();
  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  if (test_params.exist_optional) {
    auto holder = gert::TilingContextFaker()
                      .SetOpType("GroupedBiasAddGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&test_params.grad_y_shape, &test_params.group_idx_shape})
                      .OutputShapes({&test_params.grad_bias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, test_params.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, test_params.data_type_optional, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, test_params.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"group_idx_type", ge::AnyValue::CreateFrom<int64_t>(test_params.group_idx_type)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // check tiling result
    ge::graphStatus actual_staus = tiling_func(tiling_context);
    EXPECT_EQ(actual_staus, test_params.expect_status) << test_params.debug_info;

    if (test_params.check_tiling_key) {
      auto actual_tiling_key = tiling_context->GetTilingKey();
      ASSERT_EQ(actual_tiling_key, test_params.expect_tiling_key) << test_params.debug_info;
    }
  } else {
    auto holder = gert::TilingContextFaker()
                      .SetOpType("GroupedBiasAddGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&test_params.grad_y_shape, nullptr})
                      .OutputShapes({&test_params.grad_bias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, test_params.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, test_params.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"group_idx_type", ge::AnyValue::CreateFrom<int64_t>(test_params.group_idx_type)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // check tiling result
    ge::graphStatus actual_staus = tiling_func(tiling_context);
    EXPECT_EQ(actual_staus, test_params.expect_status) << test_params.debug_info;

    if (test_params.check_tiling_key) {
      auto actual_tiling_key = tiling_context->GetTilingKey();
      ASSERT_EQ(actual_tiling_key, test_params.expect_tiling_key) << test_params.debug_info;
    }
  }
}

const auto GroupedBiasAddGradTestCases = ::testing::Values(
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{10, 32}, {10, 32}},
        {{3}, {3}},          // input shape
        {{3, 32}, {3, 32}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT32,                         // dtype
        "exist_optional_input_float_use_ub",  // debug info
        ge::GRAPH_SUCCESS,
        1000111,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{10, 32}, {10, 32}},
        {{3}, {3}},          // input shape
        {{3, 32}, {3, 32}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT64,                         // dtype
        "exist_optional_input_float_grp_int64_use_ub",  // debug info
        ge::GRAPH_SUCCESS,
        1010111,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1999, 128}, {1999, 128}},
        {{10}, {10}},            // input shape
        {{10, 128}, {10, 128}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT32,                    // dtype
        "exist_optional_input_float16",  // debug info
        ge::GRAPH_SUCCESS,
        1000010,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1999, 128}, {1999, 128}},
        {{10}, {10}},            // input shape
        {{10, 128}, {10, 128}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT64,                    // dtype
        "exist_optional_input_float16_grp_int64",  // debug info
        ge::GRAPH_SUCCESS,
        1010010,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        false,  // exist optional_input
        {{199, 568, 122}, {199, 568, 122}},
        {{10}, {10}},              // input shape
        {{199, 122}, {199, 122}},  // output shape
        ge::DT_BF16,
        ge::DT_INT32,                                // dtype
        "not_exist_optional_input_bfloat16_use_ub",  // debug info
        ge::GRAPH_SUCCESS,
        1000102,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        false,  // exist optional_input
        {{199, 2000, 122}, {199, 2000, 122}},
        {{10}, {10}},              // input shape
        {{199, 122}, {199, 122}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT32,                      // dtype
        "not_exist_optional_input_float",  // debug info
        ge::GRAPH_SUCCESS,
        1000001,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1968, 4096}, {1968, 4096}},
        {{224}, {224}},              // input shape
        {{224, 4096}, {224, 4096}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT32,                         // dtype
        "exist_optional_input_float16_perf",  // debug info
        ge::GRAPH_SUCCESS,
        1001010,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1968, 4096}, {1968, 4096}},
        {{224}, {224}},              // input shape
        {{224, 4096}, {224, 4096}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT64,                         // dtype
        "exist_optional_input_float16_grp_int64_perf",  // debug info
        ge::GRAPH_SUCCESS,
        1011010,
        true  // check_tiling
    },
    GroupedBiasAddGradData{
        false,  // exist optional_input
        {{199, 2000, 122}, {199, 2000, 122}},
        {{10}, {10}},              // input shape
        {{199, 122}, {199, 122}},  // output shape
        ge::DT_INT32,
        ge::DT_INT32,               // dtype
        "input_dtype_not_support",  // debug info
        ge::GRAPH_FAILED,
        1000001,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{199, 2000, 122}, {199, 2000, 122}},
        {{10}, {10}},              // input shape
        {{199, 122}, {199, 122}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT32,                              // dtype
        "exist_optional_input_shape_not_support",  // debug info
        ge::GRAPH_FAILED,
        1000001,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{2000, 122}, {2000, 122}},
        {{10, 2}, {10, 2}},              // input shape
        {{199, 122}, {199, 122}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT32,                              // dtype
        "exist_optional_input_shape_dim_not_support",  // debug info
        ge::GRAPH_FAILED,
        1000001,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        false,  // exist optional_input
        {{199, 122}, {199, 122}},
        {{10}, {10}},              // input shape
        {{199, 122}, {199, 122}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT32,                                  // dtype
        "not_exist_optional_input_shape_not_support",  // debug info
        ge::GRAPH_FAILED,
        1000001,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{10, 32}, {10, 32}},
        {{3}, {3}},          // input shape
        {{3, 32}, {3, 32}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT8,                               // dtype
        "exist_optional_input_dtype_not_support",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{10000, 32}, {10000, 32}},
        {{2049}, {2049}},          // input shape
        {{2049, 32}, {2049, 32}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT32,                                   // dtype
        "exist_optional_input_dimG_greater_than_2048",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1000, 32}, {1000, 32}},
        {{20}, {20}},                // input shape
        {{20, 3, 32}, {20, 3, 32}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT32,                // dtype
        "output_shape_not_support",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1000, 32}, {1000, 32}},
        {{20}, {20}},            // input shape
        {{200, 32}, {200, 32}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT32,                                    // dtype
        "exist_optional_input_output_shape_dim0_error",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{1000, 32}, {1000, 32}},
        {{20}, {20}},            // input shape
        {{20, 302}, {20, 302}},  // output shape
        ge::DT_FLOAT16,
        ge::DT_INT32,                                    // dtype
        "exist_optional_input_output_shape_dim1_error",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        false,  // exist optional_input
        {{199, 568, 122}, {199, 568, 122}},
        {{10}, {10}},            // input shape
        {{19, 122}, {19, 122}},  // output shape
        ge::DT_BF16,
        ge::DT_INT32,                                        // dtype
        "not_exist_optional_input_output_shape_dim0_error",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    },
    GroupedBiasAddGradData{
        true,  // exist optional_input
        {{10, 32}, {10, 32}},
        {{3}, {3}},          // input shape
        {{3, 32}, {3, 32}},  // output shape
        ge::DT_FLOAT,
        ge::DT_INT64,                         // dtype
        "exist_optional_input_float_grp_int64_use_ub",  // debug info
        ge::GRAPH_SUCCESS,
        1010111,
        true,  // check_tiling
        1,
    },
    GroupedBiasAddGradData{
        false,  // exist optional_input
        {{199, 568, 122}, {199, 568, 122}},
        {{10}, {10}},            // input shape
        {{199, 12}, {199, 12}},  // output shape
        ge::DT_BF16,
        ge::DT_INT32,                                        // dtype
        "not_exist_optional_input_output_shape_dim1_error",  // debug info
        ge::GRAPH_FAILED,
        1000111,
        false  // check_tiling
    });

INSTANTIATE_TEST_SUITE_P(GroupedBiasAddGradTilingCases, TilingGroupedBiasAddGrad, GroupedBiasAddGradTestCases);