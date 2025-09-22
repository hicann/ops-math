/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_reflection_pad3d_grad_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "log/log.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "conversion/reflection_pad3d_grad/op_host/reflection_pad3d_grad_tiling.h"
#include "image_ops.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/op_tiling_util.h"
#include "test_cube_util.h"
class ReflectionPad3dGradTilingData : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReflectionPad3dGradTilingData  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ReflectionPad3dGradTilingData  TearDown" << std::endl;
    }
};

template <typename T>
void SetConstInput(
    size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor(
        {{data_size}, {data_size}},         // shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
        gert::kFollowing,                   // placement
        dtype,                              // dt
        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}
template <typename T>
void TilingTest(
    std::string opName, std::initializer_list<int64_t>& xShape, std::initializer_list<int64_t>& paddingShape,
    std::initializer_list<int64_t>& outShape, const int32_t inputNum, const int32_t outputNum,
    std::vector<T> paddingValues, const int32_t paddingNum, ge::DataType datatype, ge::Format format, string mode,
    bool paddingsContiguous, const ge::graphStatus status, uint64_t tilingKeyValue)
{
    std::string op_type(opName);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string compile_info_string = R"({
                                        "hardware_info": {
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
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::Tiling4PadV3GradV2CompileInfo compile_info;
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
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {xShape, xShape};
    gert::StorageShape paddings = {paddingShape, paddingShape};
    gert::StorageShape output = {outShape, outShape};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    std::vector<uint32_t> irInstanceNum(inputNum, 1);

    // 127-132:构造const输入，若不需要构造，请删除，同时去掉下面的 .ConstInput(const_tensors)
    T pad[paddingNum] = {0};
    for (size_t i = 4; i < paddingNum; i++) {
        pad[i] = paddingValues[i - 4];
    }
    SetConstInput(1, DT_INT32, pad, paddingNum, const_tensors);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(inputNum, outputNum)
                      .IrInstanceNum(irInstanceNum)
                      .InputShapes({&x, &paddings})
                      .OutputShapes({&output})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, datatype, format, format)
                      .NodeInputTd(1, ge::DT_INT32, format, format)
                      .NodeOutputTd(0, datatype, format, format)
                      // .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<std::string>(mode)},
                      //             {"paddings_contiguous", ge::AnyValue::CreateFrom<bool>(paddingsContiguous)}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_SUCCESS) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKeyValue);
    }
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float32_success_case0)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 66;
    int64_t iW = 66;
    int64_t oD = 16;
    int64_t oH = 64;
    int64_t oW = 64;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT,
        ge::FORMAT_ND, mode, true, status, 100);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float32_success_case1)
{
    int64_t N = 2;
    int64_t C = 64;
    int64_t iD = 18;
    int64_t iH = 130;
    int64_t iW = 130;
    int64_t oD = 16;
    int64_t oH = 128;
    int64_t oW = 128;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT,
        ge::FORMAT_ND, mode, true, status, 101);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float16_success_case2)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 66;
    int64_t iW = 66;
    int64_t oD = 16;
    int64_t oH = 64;
    int64_t oW = 64;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16,
        ge::FORMAT_ND, mode, true, status, 200);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float16_success_case3)
{
    int64_t N = 2;
    int64_t C = 64;
    int64_t iD = 18;
    int64_t iH = 130;
    int64_t iW = 130;
    int64_t oD = 16;
    int64_t oH = 128;
    int64_t oW = 128;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16,
        ge::FORMAT_ND, mode, true, status, 203);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_bf16_success_case4)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 66;
    int64_t iW = 66;
    int64_t oD = 16;
    int64_t oH = 64;
    int64_t oW = 64;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16,
        ge::FORMAT_ND, mode, true, status, 300);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_bf16_success_case5)
{
    int64_t N = 2;
    int64_t C = 64;
    int64_t iD = 18;
    int64_t iH = 130;
    int64_t iW = 130;
    int64_t oD = 16;
    int64_t oH = 128;
    int64_t oW = 128;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16,
        ge::FORMAT_ND, mode, true, status, 303);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float32_success_case6)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 22;
    int64_t iW = 660;
    int64_t oD = 16;
    int64_t oH = 20;
    int64_t oW = 658;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT,
        ge::FORMAT_ND, mode, true, status, 102);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float16_success_case7)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 22;
    int64_t iW = 1220;
    int64_t oD = 16;
    int64_t oH = 20;
    int64_t oW = 1218;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16,
        ge::FORMAT_ND, mode, true, status, 202);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_bf16_success_case8)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 22;
    int64_t iW = 1220;
    int64_t oD = 16;
    int64_t oH = 20;
    int64_t oW = 1218;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16,
        ge::FORMAT_ND, mode, true, status, 302);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float32_success_case9)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 34;
    int64_t iW = 1220;
    int64_t oD = 16;
    int64_t oH = 32;
    int64_t oW = 1218;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT,
        ge::FORMAT_ND, mode, true, status, 103);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_float16_success_case10)
{
    int64_t N = 2;
    int64_t C = 128;
    int64_t iD = 18;
    int64_t iH = 34;
    int64_t iW = 1220;
    int64_t oD = 16;
    int64_t oH = 32;
    int64_t oW = 1218;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16,
        ge::FORMAT_ND, mode, true, status, 203);
}

TEST_F(ReflectionPad3dGradTilingData, ReflectionPad3dGradTilingData_test_bf16_success_case11)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iD = 18;
    int64_t iH = 34;
    int64_t iW = 1220;
    int64_t oD = 16;
    int64_t oH = 32;
    int64_t oW = 1218;
    int32_t paddingNum = 10;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iD, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oD, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "ReflectionPad3dGrad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16,
        ge::FORMAT_ND, mode, true, status, 303);
}