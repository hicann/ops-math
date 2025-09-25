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
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "test_common.h"
#include "comp_ops.h"
#include "../../../../math/hans_decode/op_host/hans_decode_tiling.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry_base.h"

using namespace std;
using namespace ge;

struct HansDecodeTilingTestParam {
  string caseName;
  string opType;

  // mantissa
  ge::Format mantissaFormat;
  ge::Format mantissaOrgFormat;
  ge::DataType mantissaDtype;
  std::initializer_list<int64_t> mantissaShape;
  // pdf
  ge::Format pdfFormat;
  ge::Format pdfOrgFormat;
  ge::DataType pdfDtype;
  std::initializer_list<int64_t> pdfShape;
  // fixed
  ge::Format fixedFormat;
  ge::Format fixedOrgFormat;
  ge::DataType fixedDtype;
  std::initializer_list<int64_t> fixedShape;
  // var
  ge::Format varFormat;
  ge::Format varOrgFormat;
  ge::DataType varDtype;
  std::initializer_list<int64_t> varShape;
  // recover
  ge::Format recoverFormat;
  ge::Format recoverOrgFormat;
  ge::DataType recoverDtype;
  std::initializer_list<int64_t> recoverShape;

  bool reshuff;
  uint32_t tilingKey;
};

struct HansDecodeCompileInfo {};

class HansDecodeTilingRuntime : public testing::TestWithParam<HansDecodeTilingTestParam> {
 protected:
  static void SetUpTestCase() {
    std::cout << "HansDecode SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HansDecode TearDown" << std::endl;
  }
};

TEST_P(HansDecodeTilingRuntime, generalCases) {
  HansDecodeTilingTestParam param = GetParam();

  std::string op_type("HansDecode");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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
  HansDecodeCompileInfo compile_info;
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

  // tilingFunc simulate
  auto tiling_data = gert::TilingData::CreateCap(4096);
  ASSERT_NE(tiling_data, nullptr);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  gert::StorageShape mantissaShape = {param.mantissaShape, param.mantissaShape};
  gert::StorageShape fixedShape = {param.fixedShape, param.fixedShape};
  gert::StorageShape varShape = {param.varShape, param.varShape};
  gert::StorageShape pdfShape = {param.pdfShape, param.pdfShape};
  gert::StorageShape recoverShape = {param.recoverShape, param.recoverShape};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&mantissaShape, &fixedShape, &varShape, &pdfShape})
                    .OutputShapes({&recoverShape})
                    .NodeInputTd(0, param.mantissaDtype, param.mantissaOrgFormat, param.mantissaFormat)
                    .NodeInputTd(1, param.fixedDtype, param.fixedOrgFormat, param.fixedFormat)
                    .NodeInputTd(2, param.varDtype, param.varOrgFormat, param.varFormat)
                    .NodeInputTd(3, param.pdfDtype, param.pdfOrgFormat, param.pdfFormat)
                    .NodeOutputTd(0, param.recoverDtype, param.recoverOrgFormat, param.recoverFormat)
                    .NodeAttrs({{"reshuff", ge::AnyValue::CreateFrom<bool>(param.reshuff)}})
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  std::cout << "[ATTENTION] varShape: " << param.varShape.begin() << std::endl;

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();
  ASSERT_EQ(tiling_key, param.tilingKey);
}

inline int64_t GetProcessBlockDim(int64_t size, int64_t aivNum) {
  int64_t properAivNum = size / 32768;
  return properAivNum > aivNum ? aivNum : properAivNum;
}

inline int64_t getFixedNumelByRatio(int64_t expSize, float ratio, ge::DataType dtype) {
  return int64_t(expSize * ratio) / GetSizeByDataType(dtype);
}

inline int64_t getVarNumelByRatio(int64_t expSize, float ratio, ge::DataType dtype) {
  int64_t fixedSize = getFixedNumelByRatio(expSize, ratio, dtype) * GetSizeByDataType(dtype);
  int64_t compressUpperBoundBytes = expSize + expSize / 64 + 8448 * GetProcessBlockDim(expSize, 40) + 512;
  return (compressUpperBoundBytes - fixedSize) / GetSizeByDataType(dtype) + 1;
}

static HansDecodeTilingTestParam testParamCases[] = {
    // case 1
    {"HansDecode_Basic_Test01",
     "HansDecode",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT,
     {1, 4096 * 4096 * 3 / 4},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_INT32,
     {1, 256},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT,
     {1, getFixedNumelByRatio(4096 * 4096, 0.2, ge::DT_FLOAT)},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT,
     {1, getVarNumelByRatio(4096 * 4096, 0.2, ge::DT_FLOAT)},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT,
     {4096, 4096},
     false,
     4},  // case 1
    {"HansDecode_Basic_Test02",
     "HansDecode",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT16,
     {1, 4096 * 4096 / 2},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_INT32,
     {1, 256},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT16,
     {1, getFixedNumelByRatio(4096 * 4096, 0.2, ge::DT_FLOAT16)},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT16,
     {1, getVarNumelByRatio(4096 * 4096, 0.2, ge::DT_FLOAT16)},
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::DT_FLOAT16,
     {4096, 4096},
     false,
     2},
};

INSTANTIATE_TEST_CASE_P(HansDecode, HansDecodeTilingRuntime, testing::ValuesIn(testParamCases));
