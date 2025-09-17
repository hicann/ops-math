/**
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <iostream>
#include <vector>
#include <tuple>
#include <gtest/gtest.h>
#include "math/sinkhorn/op_host/sinkhorn_tiling.h"
#include "math/sinkhorn/tests/ut/op_kernel/sinkhorn_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class SinkhornTiling : public testing::Test {
protected:
 static void SetUpTestCase() {
   std::cout << "SinkhornTiling SetUp" << std::endl;
 }

 static void TearDownTestCase() {
   std::cout << "SinkhornTiling TearDown" << std::endl;
 }
};

 const string compile_info_string = R"({
   "hardware_info": {
     "BT_SIZE": 0,
     "load3d_constraints": "1",
     "Intrinsic_fix_pipe_l0c2out": false,
     "Intrinsic_data_move_l12ub": true,
     "Intrinsic_data_move_l0c2ub": true,
     "Intrinsic_data_move_out2l1_nd2nz": false,
     "UB_SIZE": 196352,
     "L2_SIZE": 33554432,
     "L1_SIZE": 524288,
     "L0A_SIZE": 65536,
     "L0B_SIZE": 65536,
     "L0C_SIZE": 131072,
     "CORE_NUM": 40}
   })";

using void (*CompareFunc)(void *, void *);

void CompareTilingData(void * result, void * expect)
{

 SinkhornTilingDataUT * rt = (SinkhornTilingDataUT*) result;
 SinkhornTilingDataUT * et = (SinkhornTilingDataUT*) expect;

 ASSERT_EQ(rt->formerNum, et->formerNum);
 ASSERT_EQ(rt->formerRow, et->formerRow);
 ASSERT_EQ(rt->formerLength, et->formerLength);

 ASSERT_EQ(rt->formerTileNum, et->formerTileNum);
 ASSERT_EQ(rt->formerLastTileRow, et->formerLastTileRow);
 ASSERT_EQ(rt->formerLastTileLength, et->formerLastTileLength);

 ASSERT_EQ(rt->tailNum, et->tailNum);
 ASSERT_EQ(rt->tailRow, et->tailRow);
 ASSERT_EQ(rt->tailLength, et->tailLength);

 ASSERT_EQ(rt->tailTileNum, et->tailTileNum);
 ASSERT_EQ(rt->tailLastTileRow, et->tailLastTileRow);
 ASSERT_EQ(rt->tailLastTileLength, et->tailLastTileLength);

 ASSERT_EQ(rt->tileRow, et->tileRow);
 ASSERT_EQ(rt->tileLength, et->tileLength);

 ASSERT_EQ(rt->totalRow, et->totalRow);
 ASSERT_EQ(rt->totalCol, et->totalCol);
 ASSERT_EQ(rt->totalColAligned, et->totalColAligned);

 ASSERT_FLOAT_EQ(rt->tol, et->tol);

}

void CompareCompileInfo(void * result, void * expect)
{
 optiling::SinkhornCompileInfo * rt = (optiling::SinkhornCompileInfo*) result;
 optiling::SinkhornCompileInfo * et = (optiling::SinkhornCompileInfo*) expect;

 ASSERT_EQ(rt->aivNum, et->aivNum);
 // ASSERT_EQ(rt->sysWorkspaceSize, et->sysWorkspaceSize);
 ASSERT_EQ(rt->ubSize, et->ubSize);
}

struct TestTilingTestInput {
 string op_name;
 void * compile_info;
 std::vector<void *> inputShapes;
 std::vector<void *> outputShapes;
 std::vector<std::pair<std::string, ge::AnyValue>> attrs;
 std::vector<std::tuple<ge::DataType, ge::Format>> inputs;
 std::vector<std::tuple<ge::DataType, ge::Format>> outputs;
 std::vector<uint32_t> irnum;
};

struct TestTilingTestCompare {
 uint32_t tilingKey;
 uint32_t blockDim;
 CompareFunc cf;
 void *tilingData;
};

struct TestTilingParseTestInput {
 string op_name;
 string compile_info_string;
};

struct TestTilingParseTestCompare {
 CompareFunc cf;
 void * compileInfoResult;
 void * compileInfoExpect;
};

void CommonTilingParseTest(const TestTilingParseTestInput &input, const TestTilingParseTestCompare &cmp)
{
 map<string, string> soc_infos;
 map<string, string> aicore_spec;
 map<string, string> intrinsics;
 GetPlatFormInfos(input.compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

 // platform info
 fe::PlatFormInfos platform_info;
 platform_info.Init();

 std::string op_type = input.op_name;
 ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
 auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

 // tilingParseFunc simulate
 auto kernel_holder = gert::KernelRunContextFaker()
                   .KernelIONum(2, 1)
                   .Inputs({const_cast<char *>(input.compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                   .Outputs({cmp.compileInfoResult})
                   .Build();
 ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());

 kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
 kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
 kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
 kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
 ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

 if (cmp.cf != nullptr && cmp.compileInfoExpect != nullptr) {
   cmp.cf(cmp.compileInfoResult, cmp.compileInfoExpect);
 }
}

void CommonTilingTest(const TestTilingTestInput &input, const TestTilingTestCompare &cmp)
{
 std::string op_type = input.op_name;
 ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
 auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

 auto param = gert::TilingData::CreateCap(4096);
 ASSERT_NE(param, nullptr);
 auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
 auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());

 auto faker = gert::TilingContextFaker()
                   .NodeIoNum(input.inputShapes.size(), input.outputShapes.size())
                   .IrInstanceNum(input.irnum)
                   .InputShapes(input.inputShapes)
                   .OutputShapes(input.outputShapes)
                   .CompileInfo(input.compile_info)
                   .NodeAttrs(input.attrs)
                   .TilingData(param.get())
                   .Workspace(ws_size);
 for (int i = 0; i < input.inputs.size(); i++) {
   auto a_input = input.inputs[i];
   faker.NodeInputTd(i, std::get<0>(a_input), std::get<1>(a_input), std::get<1>(a_input));
 }
 for (int i = 0; i < input.outputs.size(); i++) {
   auto a_output = input.outputs[i];
   faker.NodeOutputTd(i, std::get<0>(a_output), std::get<1>(a_output), std::get<1>(a_output));
 }
 auto holder = faker.Build();
 gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

 EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

 auto tilingKeyR = tiling_context->GetTilingKey();
 ASSERT_EQ(tilingKeyR, cmp.tilingKey);

 auto blockDimR = tiling_context->GetBlockDim();
 ASSERT_EQ(blockDimR, cmp.blockDim);

 void * tilingDataR = (void *)(tiling_context->GetRawTilingData()->GetData());
 ASSERT_NE(tilingDataR, nullptr);
 if (cmp.cf != nullptr && cmp.tilingData != nullptr)
 {
   cmp.cf(tilingDataR, cmp.tilingData);
 }
}

TEST_F(SinkhornTiling, sinkhorn_tiling_parse) {
 TestTilingParseTestInput input;
 input.op_name = "Sinkhorn";
 input.compile_info_string = compile_info_string;

 optiling::SinkhornCompileInfo result_compile_info;
 result_compile_info.aivNum = -1;
 result_compile_info.sysWorkspaceSize = -1;
 result_compile_info.ubSize = -1;

 // compile info
 optiling::SinkhornCompileInfo compile_info;
 compile_info.aivNum = 40;                                  // AIV核数
 compile_info.sysWorkspaceSize = 16 * 1024 * 1024;          // 系统WorkSpace大小
 compile_info.ubSize = 196352;                              // UB大小

 TestTilingParseTestCompare cmp;
 cmp.cf = CompareCompileInfo;
 cmp.compileInfoResult = (void *)&result_compile_info;
 cmp.compileInfoExpect = (void *)&compile_info;

 CommonTilingParseTest(input, cmp);
}

TEST_F(SinkhornTiling, sinkhorn_tiling_48_2) {
 SinkhornTilingDataUT tilingData;
 tilingData.formerNum = 1;              // former 数量
 tilingData.formerRow = 48;             // former cost行数
 tilingData.formerLength = 96;          // former cost总长
 tilingData.formerTileNum = 1;          // former Tile数量
 tilingData.formerLastTileRow = 48;     // fomer last Tile行数
 tilingData.formerLastTileLength = 96;  // fomer last Tile长度
 tilingData.tailNum = 0;                // tail 数量
 tilingData.tailRow = 0;                // tail cost行数
 tilingData.tailLength = 0;             // tail cost总长
 tilingData.tailTileNum = 0;            // tail Tile数量
 tilingData.tailLastTileRow = 0;        // tail last Tile行数
 tilingData.tailLastTileLength = 0;     // tail last Tile长度
 tilingData.tileRow = 2082;             // Tile行数(非Last)
 tilingData.tileLength = 4164;          // Tile长度(非Last)
 tilingData.totalRow = 48;              // 总行数
 tilingData.totalCol = 2;               // 总列数
 tilingData.totalColAligned = 8;        // 对齐后的总列数
 tilingData.tol = 0.0001;               // 误差

 gert::StorageShape input_cost_shape = {{48, 2}, {48, 2}};
 gert::StorageShape output_p_shape = {{48, 2}, {48, 2}};

 // compile info
 optiling::SinkhornCompileInfo compile_info;
 compile_info.aivNum = 40;                                  // AIV核数
 compile_info.sysWorkspaceSize = 16 * 1024 * 1024;          // 系统WorkSpace大小
 compile_info.ubSize = 196352;                              // UB大小

 TestTilingTestInput input;
 input.op_name = "Sinkhorn";
 input.compile_info = &compile_info;
 input.inputShapes = {&input_cost_shape};
 input.outputShapes = {&output_p_shape};
 input.attrs = {{"tol", ge::AnyValue::CreateFrom<float>(0.0001)}};
 input.inputs = {{ge::DT_FLOAT, ge::FORMAT_ND}};
 input.outputs = {{ge::DT_FLOAT, ge::FORMAT_ND}};
 input.irnum = {1};

 TestTilingTestCompare cmp;
 cmp.tilingKey = 0;
 cmp.blockDim = 1;
 cmp.cf = CompareTilingData;
 cmp.tilingData = &tilingData;

 CommonTilingTest(input, cmp);
}

TEST_F(SinkhornTiling, sinkhorn_tiling_10000_160) {
 SinkhornTilingDataUT tilingData;
 tilingData.formerNum = 40;                // former 数量
 tilingData.formerRow = 250;               // former cost行数
 tilingData.formerLength = 40000;          // former cost总长
 tilingData.formerTileNum = 3;             // former Tile数量
 tilingData.formerLastTileRow = 2;         // fomer last Tile行数
 tilingData.formerLastTileLength = 320;    // fomer last Tile长度
 tilingData.tailNum = 0;                   // tail 数量
 tilingData.tailRow = 0;                   // tail cost行数
 tilingData.tailLength = 0;                // tail cost总长
 tilingData.tailTileNum = 0;               // tail Tile数量
 tilingData.tailLastTileRow = 0;           // tail last Tile行数
 tilingData.tailLastTileLength = 0;        // tail last Tile长度
 tilingData.tileRow = 124;                 // Tile行数(非Last)
 tilingData.tileLength = 19840;            // Tile长度(非Last)
 tilingData.totalRow = 10000;              // 总行数
 tilingData.totalCol = 160;                // 总列数
 tilingData.totalColAligned = 160;         // 对齐后的总列数
 tilingData.tol = 0.00001;                 // 误差

 gert::StorageShape input_cost_shape = {{10000, 160}, {10000, 160}};
 gert::StorageShape output_p_shape = {{10000, 160}, {10000, 160}};

 // compile info
 optiling::SinkhornCompileInfo compile_info;
 compile_info.aivNum = 40;                                  // AIV核数
 compile_info.sysWorkspaceSize = 16 * 1024 * 1024;          // 系统WorkSpace大小
 compile_info.ubSize = 196352;                              // UB大小

 TestTilingTestInput input;
 input.op_name = "Sinkhorn";
 input.compile_info = &compile_info;
 input.inputShapes = {&input_cost_shape};
 input.outputShapes = {&output_p_shape};
 input.attrs = {{"tol", ge::AnyValue::CreateFrom<float>(0.00001)}};
 input.inputs = {{ge::DT_FLOAT, ge::FORMAT_ND}};
 input.outputs = {{ge::DT_FLOAT, ge::FORMAT_ND}};
 input.irnum = {1};

 TestTilingTestCompare cmp;
 cmp.tilingKey = 0;
 cmp.blockDim = 40;
 cmp.cf = CompareTilingData;
 cmp.tilingData = &tilingData;

 CommonTilingTest(input, cmp);
}
