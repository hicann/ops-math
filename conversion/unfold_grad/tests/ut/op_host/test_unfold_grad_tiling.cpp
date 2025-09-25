/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_unfold_grad_tiling.cpp
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
#include "conversion/unfold_grad/op_host/unfold_grad_tiling.h"
#include "image_ops.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/op_tiling_util.h"
#include "test_cube_util.h"
class UnfoldGradTilingData : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "UnfoldGradTilingData  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "UnfoldGradTilingData  TearDown" << std::endl;
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
    std::string opName, std::initializer_list<int64_t>& gradOutShape, std::initializer_list<int64_t>& inputSizeShape,
    int64_t dim, int64_t size, int64_t step, std::initializer_list<int64_t>& outShape, const int32_t inputNum,
    const int32_t outputNum, std::vector<T> inputSizeValues, const int32_t inputSizeNum, ge::DataType datatype,
    ge::Format format, const ge::graphStatus status, uint64_t tilingKeyValue)
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
    optiling::Tiling4UnfoldGradCompileInfo compile_info;
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
    gert::StorageShape gradOut = {gradOutShape, gradOutShape};
    gert::StorageShape inputSize = {inputSizeShape, inputSizeShape};
    gert::StorageShape output = {outShape, outShape};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    std::vector<uint32_t> irInstanceNum(inputNum, 1);

    // 127-132:构造const输入，若不需要构造，请删除，同时去掉下面的 .ConstInput(const_tensors)
    T targetShape[inputSizeNum] = {0};
    for (size_t i = 0; i < inputSizeNum; i++) {
        targetShape[i] = inputSizeValues[i];
    }
    SetConstInput(1, DT_INT64, targetShape, inputSizeNum, const_tensors);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(inputNum, outputNum)
                      .IrInstanceNum(irInstanceNum)
                      .InputShapes({&gradOut, &inputSize})
                      .OutputShapes({&output})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, datatype, format, format)
                      .NodeInputTd(1, ge::DT_INT64, format, format)
                      .NodeOutputTd(0, datatype, format, format)
                      .NodeAttrs(
                          {{"dim", ge::AnyValue::CreateFrom<int64_t>(dim)},
                           {"size", ge::AnyValue::CreateFrom<int64_t>(size)},
                           {"step", ge::AnyValue::CreateFrom<int64_t>(step)}})
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

TEST_F(UnfoldGradTilingData, UnfoldGradTilingData_test_float32_outputshape_8_2_dim_0_size_3_step_2_success_case0)
{
    int32_t inputSizeNum = 2;
    std::vector<int64_t> inputSizeValues = {8, 2};
    std::initializer_list<int64_t> gradOutShape = {3, 2, 3};
    std::initializer_list<int64_t> inputSizeShape = {2};
    std::initializer_list<int64_t> outShape = {8, 2};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t dim = 0;
    int64_t size = 3;
    int64_t step = 2;
    TilingTest<int64_t>(
        "UnfoldGrad", gradOutShape, inputSizeShape, dim, size, step, outShape, 2, 1, inputSizeValues, inputSizeNum,
        ge::DT_FLOAT, ge::FORMAT_ND, status, 212);
}

TEST_F(
    UnfoldGradTilingData,
    UnfoldGradTilingData_test_float16_outputshape_1_3_3_658_658_dim_4_size_20_step_2_success_case0)
{
    int32_t inputSizeNum = 5;
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    std::initializer_list<int64_t> gradOutShape = {1, 3, 3, 658, 320, 20};
    std::initializer_list<int64_t> inputSizeShape = {5};
    std::initializer_list<int64_t> outShape = {1, 3, 3, 658, 658};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t dim = 4;
    int64_t size = 20;
    int64_t step = 2;
    TilingTest<int64_t>(
        "UnfoldGrad", gradOutShape, inputSizeShape, dim, size, step, outShape, 2, 1, inputSizeValues, inputSizeNum,
        ge::DT_FLOAT16, ge::FORMAT_ND, status, 221);
}

TEST_F(
    UnfoldGradTilingData,
    UnfoldGradTilingData_test_bfloat16_outputshape_1_3_3_658_658_dim_3_size_20_step_2_success_case0)
{
    int32_t inputSizeNum = 5;
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    std::initializer_list<int64_t> gradOutShape = {1, 3, 3, 320, 658, 20};
    std::initializer_list<int64_t> inputSizeShape = {5};
    std::initializer_list<int64_t> outShape = {1, 3, 3, 658, 658};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t dim = 3;
    int64_t size = 20;
    int64_t step = 2;
    TilingTest<int64_t>(
        "UnfoldGrad", gradOutShape, inputSizeShape, dim, size, step, outShape, 2, 1, inputSizeValues, inputSizeNum,
        ge::DT_BF16, ge::FORMAT_ND, status, 232);
}

TEST_F(
    UnfoldGradTilingData,
    UnfoldGradTilingData_test_float32_outputshape_1_3_3_658_658_dim_3_size_2_step_20_success_case0)
{
    int32_t inputSizeNum = 5;
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    std::initializer_list<int64_t> gradOutShape = {1, 3, 3, 33, 658, 2};
    std::initializer_list<int64_t> inputSizeShape = {5};
    std::initializer_list<int64_t> outShape = {1, 3, 3, 658, 658};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t dim = 3;
    int64_t size = 2;
    int64_t step = 20;
    TilingTest<int64_t>(
        "UnfoldGrad", gradOutShape, inputSizeShape, dim, size, step, outShape, 2, 1, inputSizeValues, inputSizeNum,
        ge::DT_FLOAT, ge::FORMAT_ND, status, 312);
}

TEST_F(
    UnfoldGradTilingData,
    UnfoldGradTilingData_test_float16_outputshape_1_3_3_658_658_dim_4_size_2_step_20_success_case0)
{
    int32_t inputSizeNum = 5;
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    std::initializer_list<int64_t> gradOutShape = {1, 3, 3, 658, 33, 2};
    std::initializer_list<int64_t> inputSizeShape = {5};
    std::initializer_list<int64_t> outShape = {1, 3, 3, 658, 658};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t dim = 4;
    int64_t size = 2;
    int64_t step = 20;
    TilingTest<int64_t>(
        "UnfoldGrad", gradOutShape, inputSizeShape, dim, size, step, outShape, 2, 1, inputSizeValues, inputSizeNum,
        ge::DT_FLOAT16, ge::FORMAT_ND, status, 321);
}

TEST_F(UnfoldGradTilingData, UnfoldGradTilingData_test_bfloat16_outputshape_8_2_dim_0_size_2_step_3_success_case0)
{
    int32_t inputSizeNum = 2;
    std::vector<int64_t> inputSizeValues = {8, 2};
    std::initializer_list<int64_t> gradOutShape = {3, 2, 2};
    std::initializer_list<int64_t> inputSizeShape = {2};
    std::initializer_list<int64_t> outShape = {8, 2};
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t dim = 0;
    int64_t size = 2;
    int64_t step = 3;
    TilingTest<int64_t>(
        "UnfoldGrad", gradOutShape, inputSizeShape, dim, size, step, outShape, 2, 1, inputSizeValues, inputSizeNum,
        ge::DT_BF16, ge::FORMAT_ND, status, 332);
}