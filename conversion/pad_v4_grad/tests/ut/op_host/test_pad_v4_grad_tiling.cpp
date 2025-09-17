/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "op_log.h"

#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/pad_v4_grad_tiling.h"
#include "image_ops.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/op_tiling_util.h"
#include "test_cube_util.h"

class PadV3GradV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV4GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PadV4GradTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const uint32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

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
    bool paddingsContiguous, const ge::graphStatus status, uint64_t tilingKeyValue, string expectData)
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
    optiling::Tiling4PadV4GradCompileInfo compile_info;

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
                      .NodeAttrs(
                          {{"mode", ge::AnyValue::CreateFrom<std::string>(mode)},
                           {"paddings_contiguous", ge::AnyValue::CreateFrom<bool>(paddingsContiguous)}})
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
        auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
        std::cout << tiling_data_result << std::endl;
        ASSERT_EQ(tiling_data_result, expectData);
    }
}

// tiling key 1111
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case0)
{
    int64_t N = 33;
    int64_t C = 44;
    int64_t iH = 2000;
    int64_t iW = 2000;
    int64_t oH = 1998;
    int64_t oW = 1998;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1111, "33 44 2000 2000 2000 2000 1998 1998 2000 2000 1 1 1 1 48 576 30 12 1111 0 512000 0 ");
}

// tiling key 1010
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case1)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 150;
    int64_t iW = 30;
    int64_t oH = 142;
    int64_t oW = 26;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {3, 5, 3, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1010, "1 1 150 30 160 32 142 26 144 32 3 5 3 1 1 112 1 0 1010 0 8192 0 ");
}

// tiling key 1000
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case2)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 64;
    int64_t iW = 64;
    int64_t oH = 64;
    int64_t oW = 62;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1000, "1 1 64 64 64 64 64 62 64 64 0 0 1 1 1 112 1 0 1000 0 0 0 ");
}

// tiling key 1101
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case3)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 16;
    int64_t iW = 530;
    int64_t oH = 16;
    int64_t oW = 528;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1101, "1 1 16 530 16 544 16 528 16 528 0 0 1 1 16 11520 1 0 1101 8 512 0 ");
}

// tiling key 1101
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case4)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 200;
    int64_t iW = 200;
    int64_t oH = 200;
    int64_t oW = 198;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1101, "1 1 200 200 208 208 200 198 208 208 0 0 1 1 48 11520 4 8 1101 8 512 0 ");
}

// tiling key 1110
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case5)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 200;
    int64_t iW = 200;
    int64_t oH = 198;
    int64_t oW = 200;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 0, 0};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1110, "1 1 200 200 208 208 198 200 208 208 1 1 0 0 1 5760 1 0 1110 0 0 0 ");
}

// tiling key 1010
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case6)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 150;
    int64_t iW = 30;
    int64_t oH = 142;
    int64_t oW = 26;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {3, 5, 3, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1010, "1 1 150 30 160 32 142 26 144 32 3 5 3 1 1 112 1 0 1010 0 8192 0 ");
}

// tiling key 1100
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_success_case7)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 16;
    int64_t iW = 530;
    int64_t oH = 14;
    int64_t oW = 528;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 1100, "1 1 16 530 16 544 14 528 16 528 1 1 1 1 1 240 1 0 1100 0 139264 0 ");
}

// tiling key 2111
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case0)
{
    int64_t N = 33;
    int64_t C = 44;
    int64_t iH = 2000;
    int64_t iW = 2000;
    int64_t oH = 1998;
    int64_t oW = 1998;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2111,
        "33 44 2000 2000 2000 2000 1998 1998 2000 2000 1 1 1 1 48 1152 30 12 2111 0 256000 0 ");
}

// tiling key 2010
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case1)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 150;
    int64_t iW = 30;
    int64_t oH = 142;
    int64_t oW = 26;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {3, 5, 3, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2010, "1 1 150 30 160 32 142 26 144 32 3 5 3 1 1 240 1 0 2010 0 4096 0 ");
}

// tiling key 2000
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case2)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 64;
    int64_t iW = 64;
    int64_t oH = 64;
    int64_t oW = 62;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2000, "1 1 64 64 64 64 64 62 64 64 0 0 1 1 1 240 1 0 2000 0 0 0 ");
}

// tiling key 2101
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case3)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 16;
    int64_t iW = 530;
    int64_t oH = 16;
    int64_t oW = 528;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2101, "1 1 16 530 16 544 16 528 16 528 0 0 1 1 16 23040 1 0 2101 16 256 0 ");
}

// tiling key 2101
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case4)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 200;
    int64_t iW = 200;
    int64_t oH = 200;
    int64_t oW = 198;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2101, "1 1 200 200 208 208 200 198 208 208 0 0 1 1 48 23040 4 8 2101 16 256 0 ");
}

// tiling key 2110
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case5)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 200;
    int64_t iW = 200;
    int64_t oH = 198;
    int64_t oW = 200;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 0, 0};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2110, "1 1 200 200 208 208 198 200 208 208 1 1 0 0 1 11520 1 0 2110 0 0 0 ");
}

// tiling key 2010
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case6)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 150;
    int64_t iW = 30;
    int64_t oH = 142;
    int64_t oW = 26;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {3, 5, 3, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2010, "1 1 150 30 160 32 142 26 144 32 3 5 3 1 1 240 1 0 2010 0 4096 0 ");
}

// tiling key 2100
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float16_success_case7)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 16;
    int64_t iW = 530;
    int64_t oH = 14;
    int64_t oW = 528;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT16, ge::FORMAT_ND,
        mode, true, status, 2100, "1 1 16 530 16 544 14 528 16 528 1 1 1 1 1 480 1 0 2100 0 69632 0 ");
}

// tiling key 3000
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_bfloat16_success_case1)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 64;
    int64_t iW = 64;
    int64_t oH = 64;
    int64_t oW = 62;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16, ge::FORMAT_ND, mode,
        true, status, 3000, "1 1 64 64 64 64 64 62 64 64 0 0 1 1 1 112 1 0 3000 0 0 0 ");
}

// tiling key 3101
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_bfloat16_success_case2)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 16;
    int64_t iW = 530;
    int64_t oH = 16;
    int64_t oW = 528;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16, ge::FORMAT_ND, mode,
        true, status, 3101, "1 1 16 530 16 544 16 528 16 528 0 0 1 1 16 15360 1 0 3101 16 256 0 ");
}

// tiling key 3101
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_bfloat16_success_case3)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 200;
    int64_t iW = 200;
    int64_t oH = 200;
    int64_t oW = 198;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {0, 0, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16, ge::FORMAT_ND, mode,
        true, status, 3101, "1 1 200 200 208 208 200 198 208 208 0 0 1 1 48 15360 4 8 3101 16 256 0 ");
}

// tiling key 3110
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_bfloat16_success_case4)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 200;
    int64_t iW = 200;
    int64_t oH = 198;
    int64_t oW = 200;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 0, 0};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16, ge::FORMAT_ND, mode,
        true, status, 3110, "1 1 200 200 208 208 198 200 208 208 1 1 0 0 1 7680 1 0 3110 0 0 0 ");
}

// tiling key 3010
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_bfloat16_success_case5)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 150;
    int64_t iW = 30;
    int64_t oH = 142;
    int64_t oW = 26;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {3, 5, 3, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16, ge::FORMAT_ND, mode,
        true, status, 3010, "1 1 150 30 160 32 142 26 144 32 3 5 3 1 1 112 1 0 3010 0 4096 0 ");
}

// tiling key 3100
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_bfloat16_success_case6)
{
    int64_t N = 1;
    int64_t C = 1;
    int64_t iH = 16;
    int64_t iW = 530;
    int64_t oH = 14;
    int64_t oW = 528;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_BF16, ge::FORMAT_ND, mode,
        true, status, 3100, "1 1 16 530 16 544 14 528 16 528 1 1 1 1 1 240 1 0 3100 0 69632 0 ");
}

// outshape is not correct
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_failed_case0)
{
    int64_t N = 33;
    int64_t C = 14;
    int64_t iH = 50;
    int64_t iW = 50;
    int64_t oH = 48;
    int64_t oW = 48;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 2, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 11, "33 14 50 50 64 64 48 48 48 48 1 2 1 1 48 8960 9 30 11 0 ");
}

// x shape is not 4 dims
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_failed_case1)
{
    int64_t N = 33;
    int64_t C = 14;
    int64_t iH = 50;
    int64_t iW = 50;
    int64_t oH = 48;
    int64_t oW = 48;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 11, "33 14 50 50 64 64 48 48 48 48 1 1 1 1 48 8960 9 30 11 0 ");
}

// x dtype is not supported
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_failed_case2)
{
    int64_t N = 33;
    int64_t C = 14;
    int64_t iH = 50;
    int64_t iW = 50;
    int64_t oH = 48;
    int64_t oW = 48;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_DOUBLE, ge::FORMAT_ND,
        mode, true, status, 11, "33 14 50 50 64 64 48 48 48 48 1 1 1 1 48 8960 9 30 11 0 ");
}

// mode is not supported
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_failed_case3)
{
    int64_t N = 33;
    int64_t C = 14;
    int64_t iH = 50;
    int64_t iW = 50;
    int64_t oH = 48;
    int64_t oW = 48;
    int32_t paddingNum = 8;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflection";
    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 11, "33 14 50 50 64 64 48 48 48 48 1 1 1 1 48 8960 9 30 11 0 ");
}

// padding dtype is not supported
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_failed_case4)
{
    int64_t N = 33;
    int64_t C = 14;
    int64_t iH = 50;
    int64_t iW = 50;
    int64_t oH = 48;
    int64_t oW = 48;
    int32_t paddingNum = 8;
    std::vector<float> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest<float>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_DOUBLE, ge::FORMAT_ND,
        mode, true, status, 11, "33 14 50 50 64 64 48 48 48 48 1 1 1 1 48 8960 9 30 11 0 ");
}

// padding shape's first dim is not twice the size of x shape's dimNum
TEST_F(PadV3GradV2Tiling, PadV3GradV2Tiling_test_float32_failed_case5)
{
    int64_t N = 33;
    int64_t C = 14;
    int64_t iH = 50;
    int64_t iW = 50;
    int64_t oH = 48;
    int64_t oW = 48;
    int32_t paddingNum = 7;
    std::vector<int32_t> paddingValues = {1, 1, 1, 1};
    std::initializer_list<int64_t> xShape = {N, C, iH, iW};
    std::initializer_list<int64_t> paddingShape = {paddingNum};
    std::initializer_list<int64_t> outShape = {N, C, oH, oW};
    std::string mode = "reflect";
    const ge::graphStatus status = ge::GRAPH_FAILED;
    TilingTest<int32_t>(
        "PadV4Grad", xShape, paddingShape, outShape, 2, 1, paddingValues, paddingNum, ge::DT_FLOAT, ge::FORMAT_ND, mode,
        true, status, 11, "33 14 50 50 64 64 48 48 48 48 1 1 1 1 48 8960 9 30 11 0 ");
}