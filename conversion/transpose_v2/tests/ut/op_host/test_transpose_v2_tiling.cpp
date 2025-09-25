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
 * \file test_transpose_v2_tiling.cpp
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
#include "conversion/transpose_v2/op_host/transpose_v2_tiling.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/op_tiling_util.h"
#include "test_cube_util.h"
class TransposeV2Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "TransposeV2Tiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TransposeV2Tiling  TearDown" << std::endl;
    }
};

namespace {

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
    std::string opName,
    std::vector<std::initializer_list<int64_t>>& inputShapes,  // 动态输入形状
    std::vector<std::initializer_list<int64_t>>& outputShapes, // 动态输出形状
    std::vector<ge::DataType>& inputDataTypes,                 // 动态输入数据类型
    std::vector<ge::DataType>& outputDataTypes,                // 动态输出数据类型
    std::vector<ge::Format>& inputFormats,                     // 动态输入格式
    std::vector<ge::Format>& outputFormats,                    // 动态输出格式
    std::vector<std::vector<T>>& constInputValues,             // 动态常量输入值
    std::vector<std::string>& attrNames,                       // 动态属性名称
    std::vector<int64_t>& attrValues,                          // 动态属性值
    ge::graphStatus expectedStatus,                            // 期望的返回状态
    uint64_t tilingKeyValue                                    // Tiling Key 值
)
{
    std::string op_type(opName);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 编译信息
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

    // 平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // 编译信息
    optiling::Tiling4TransposeV2CompileInfo compile_info;

    // 模拟 tilingParseFunc
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

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    // 动态设置 IR 实例数量
    std::vector<uint32_t> irInstanceNum(inputShapes.size(), 1);

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    auto permNum = constInputValues[0].size();
    T perm[permNum] = {0};
    for (size_t i = 0; i < permNum; i++) {
        perm[i] = constInputValues[0][i];
    }
    SetConstInput(1, DT_INT64, perm, permNum, const_tensors);

    // 构建 TilingContext
    auto faker = gert::TilingContextFaker()
                     .SetOpType(op_type)
                     .NodeIoNum(inputShapes.size(), outputShapes.size()) // 动态设置输入输出数量
                     .IrInstanceNum(irInstanceNum)
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                     .Workspace(ws_size)
                     .TilingData(param.get());

    // 动态设置输入和输出的形状
    std::vector<gert::StorageShape> inputStorageShape(inputShapes.size());
    std::vector<void*> inputStorageShapeRef(inputShapes.size());
    for (int i = 0; i < inputShapes.size(); i++) {
        gert::StorageShape x = {inputShapes[i], inputShapes[i]};
        inputStorageShape[i] = x;
        inputStorageShapeRef[i] = &inputStorageShape[i];
    }
    faker = faker.InputShapes(inputStorageShapeRef);

    std::vector<gert::StorageShape> outputStorageShape(outputShapes.size());
    std::vector<void*> outputStorageShapeRef(outputShapes.size());
    for (int i = 0; i < outputShapes.size(); i++) {
        gert::StorageShape x = {outputShapes[i], outputShapes[i]};
        outputStorageShape[i] = x;
        outputStorageShapeRef[i] = &outputStorageShape[i];
    }
    faker = faker.OutputShapes(outputStorageShapeRef);

    // 动态设置输入和输出的数据类型和格式
    for (int64_t i = 0; i < inputShapes.size(); i++) {
        faker = faker.NodeInputTd(i, inputDataTypes[i], inputFormats[i], inputFormats[i]);
    }
    for (int64_t i = 0; i < outputShapes.size(); i++) {
        faker = faker.NodeOutputTd(i, outputDataTypes[i], outputFormats[i], outputFormats[i]);
    }
    if (!const_tensors.empty()) {
        faker.ConstInput(const_tensors);
    }

    // 设置平台信息
    auto holder = faker.Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // 执行 Tiling 函数并验证结果
    EXPECT_EQ(tiling_func(tiling_context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKeyValue);
    }
}
}

TEST_F(TransposeV2Tiling, transpose_v2_float16_021_success_case0)
{
    std::vector<std::initializer_list<int64_t>> inputShapes;
    std::vector<std::initializer_list<int64_t>> outputShapes;
    std::vector<ge::DataType> inputDataTypes;
    std::vector<ge::DataType> outputDataTypes;
    std::vector<ge::Format> inputFormats;
    std::vector<ge::Format> outputFormats;
    std::vector<std::vector<int64_t>> constInputValues;
    std::vector<std::string> attrNames;
    std::vector<int64_t> attrValues;
    ge::graphStatus expectedStatus;
    uint64_t tilingKeyValue;

    // 算子输入输出信息
    int64_t C = 1;
    int64_t H = 30;
    int64_t W = 68;

    std::initializer_list<int64_t> x = {C, H, W};
    std::initializer_list<int64_t> perm = {3};
    std::initializer_list<int64_t> y = {C, W, H};
    std::vector<int64_t> permValue = {0, 2, 1};
    constInputValues.push_back(permValue);
    inputShapes.push_back(x);
    inputShapes.push_back(perm);
    outputShapes.push_back(y);

    inputDataTypes.push_back(ge::DT_FLOAT16);
    inputDataTypes.push_back(ge::DT_INT64);
    outputDataTypes.push_back(ge::DT_FLOAT16);

    inputFormats.push_back(ge::FORMAT_ND);
    inputFormats.push_back(ge::FORMAT_ND);
    outputFormats.push_back(ge::FORMAT_ND);

    expectedStatus = ge::GRAPH_SUCCESS;
    tilingKeyValue = 20;

    TilingTest<int64_t>(
        "TransposeV2", inputShapes, outputShapes, inputDataTypes, outputDataTypes, inputFormats, outputFormats,
        constInputValues, attrNames, attrValues, expectedStatus, tilingKeyValue);
}

TEST_F(TransposeV2Tiling, transpose_v2_float16_021_success_case1)
{
    std::vector<std::initializer_list<int64_t>> inputShapes;
    std::vector<std::initializer_list<int64_t>> outputShapes;
    std::vector<ge::DataType> inputDataTypes;
    std::vector<ge::DataType> outputDataTypes;
    std::vector<ge::Format> inputFormats;
    std::vector<ge::Format> outputFormats;
    std::vector<std::vector<int64_t>> constInputValues;
    std::vector<std::string> attrNames;
    std::vector<int64_t> attrValues;
    ge::graphStatus expectedStatus;
    uint64_t tilingKeyValue;

    // 算子输入输出信息
    int64_t C = 1;
    int64_t H = 30;
    int64_t W = 64;

    std::initializer_list<int64_t> x = {C, H, W};
    std::initializer_list<int64_t> perm = {3};
    std::initializer_list<int64_t> y = {H, C, W};
    std::vector<int64_t> permValue = {1, 0, 2};
    constInputValues.push_back(permValue);
    inputShapes.push_back(x);
    inputShapes.push_back(perm);
    outputShapes.push_back(y);

    inputDataTypes.push_back(ge::DT_FLOAT16);
    inputDataTypes.push_back(ge::DT_INT64);
    outputDataTypes.push_back(ge::DT_FLOAT16);

    inputFormats.push_back(ge::FORMAT_ND);
    inputFormats.push_back(ge::FORMAT_ND);
    outputFormats.push_back(ge::FORMAT_ND);

    expectedStatus = ge::GRAPH_SUCCESS;
    tilingKeyValue = 121;

    TilingTest<int64_t>(
        "TransposeV2", inputShapes, outputShapes, inputDataTypes, outputDataTypes, inputFormats, outputFormats,
        constInputValues, attrNames, attrValues, expectedStatus, tilingKeyValue);
}

TEST_F(TransposeV2Tiling, transpose_v2_float16_021_success_case2)
{
    std::vector<std::initializer_list<int64_t>> inputShapes;
    std::vector<std::initializer_list<int64_t>> outputShapes;
    std::vector<ge::DataType> inputDataTypes;
    std::vector<ge::DataType> outputDataTypes;
    std::vector<ge::Format> inputFormats;
    std::vector<ge::Format> outputFormats;
    std::vector<std::vector<int64_t>> constInputValues;
    std::vector<std::string> attrNames;
    std::vector<int64_t> attrValues;
    ge::graphStatus expectedStatus;
    uint64_t tilingKeyValue;

    // 算子输入输出信息
    int64_t N = 1;
    int64_t C = 1;
    int64_t H = 32;
    int64_t W = 64;

    std::initializer_list<int64_t> x = {N, C, H, W};
    std::initializer_list<int64_t> perm = {4};
    std::initializer_list<int64_t> y = {N, H, C, W};
    std::vector<int64_t> permValue = {0, 2, 1, 3};
    constInputValues.push_back(permValue);
    inputShapes.push_back(x);
    inputShapes.push_back(perm);
    outputShapes.push_back(y);

    inputDataTypes.push_back(ge::DT_FLOAT16);
    inputDataTypes.push_back(ge::DT_INT64);
    outputDataTypes.push_back(ge::DT_FLOAT16);

    inputFormats.push_back(ge::FORMAT_ND);
    inputFormats.push_back(ge::FORMAT_ND);
    outputFormats.push_back(ge::FORMAT_ND);

    expectedStatus = ge::GRAPH_SUCCESS;
    tilingKeyValue = 221;

    TilingTest<int64_t>(
        "TransposeV2", inputShapes, outputShapes, inputDataTypes, outputDataTypes, inputFormats, outputFormats,
        constInputValues, attrNames, attrValues, expectedStatus, tilingKeyValue);
}