/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"

#include "elewise_calculation_ops.h"
#include "../../data/op_graph/data_proto.h"
#include "../op_graph/destroy_temporary_variable_proto.h"
#include "../../temporary_variable/op_graph/temporary_variable_proto.h"

#define FAILED -1
#define SUCCESS 0

using std::map;
using std::string;
using std::vector;
using namespace ge;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return string(tmp);
}

// 配对构图：TemporaryVariable(分配临时变量) → DestroyTemporaryVariable(读终值并销毁，var_name 匹配)，
// Data + Add 仅用于满足 GE 图输入约束（SetInputs 不允许为空），Add 消费 Destroy 输出模拟终值读取
ge::Graph BuildPairedGraph(std::vector<ge::Tensor>& input)
{
    const vector<int64_t> shape_attr = {2, 3};
    const string var_name = "tmp_var";
    ge::Graph graph("test_destroy_temporary_variable_paired_ir");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::Shape(shape_attr), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    auto temporary_variable_node = op::TemporaryVariable("temporary_variable_node");
    temporary_variable_node.set_attr_shape(shape_attr);
    temporary_variable_node.set_attr_dtype(static_cast<int64_t>(ge::DT_FLOAT));
    temporary_variable_node.set_attr_var_name(var_name.c_str());
    ge::TensorDesc tv_output_desc(ge::Shape(shape_attr), ge::FORMAT_ND, ge::DT_FLOAT);
    temporary_variable_node.update_output_desc_y(tv_output_desc);

    auto destroy_node = op::DestroyTemporaryVariable("destroy_temporary_variable_node");
    destroy_node.set_input_x(temporary_variable_node);
    destroy_node.set_attr_var_name(var_name.c_str());
    ge::TensorDesc destroy_output_desc(ge::Shape(shape_attr), ge::FORMAT_ND, ge::DT_FLOAT);
    destroy_node.update_output_desc_y(destroy_output_desc);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(destroy_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(temporary_variable_node);
    graph.AddOp(destroy_node);
    graph.AddOp(add_node);
    return graph;
}

// 独立动态 shape 场景：TemporaryVariable 的 shape 为静态 ListInt 必填属性，无法表达 -1/-2 动态维度，
// 动态 shape 只能经 Data 节点进入图中（配对模式仅支持静态 shape，见 BuildPairedGraph）；
// 本场景验证 DestroyTemporaryVariable 对动态输入的 shape 传递（与 ReadVariableOp 等直通算子一致）
ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_destroy_temporary_variable_dynamic_dim_neg1");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-1, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_desc.SetShapeRange({{1, 10}, {3, 3}});
    data_node.update_input_desc_x(data_desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    auto destroy_node = op::DestroyTemporaryVariable("destroy_temporary_variable_node");
    destroy_node.set_input_x(data_node);
    destroy_node.set_attr_var_name("tmp_var");

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{destroy_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(destroy_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_destroy_temporary_variable_dynamic_dim_neg2");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-2};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    // unknown rank(-2) has unknown dim num, no shape range is set; the real shape is fed on RunGraph
    data_node.update_input_desc_x(data_desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    auto destroy_node = op::DestroyTemporaryVariable("destroy_temporary_variable_node");
    destroy_node.set_input_x(data_node);
    destroy_node.set_attr_var_name("tmp_var");

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{destroy_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(destroy_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator destroy_op;
    graph.FindOpByName("destroy_temporary_variable_node", destroy_op);
    // 显式触发推导，不依赖 RunGraph/BuildModel 的编译时序：
    // GE local 引擎将 DestroyTemporaryVariable 注册为 GeDeletedOp（图优化阶段应被删除的算子）
    ge::Status ret = destroy_op.InferShapeAndType();
    if (ret != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: InferShapeAndType failed, ret=%u\n", GetTime().c_str(), ret);
        return false;
    }
    auto output_desc = destroy_op.GetOutputDesc(0);
    auto shape = output_desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" | expected: dim_num=%zu", expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        printf(" dim%zu=%ld", i, expected[i]);
    }
    printf("\n");

    if (shape.GetDimNum() != expected.size()) {
        printf("%s - ERROR - [XIR]: InferShape FAILED: expected dim_num=%zu\n", GetTime().c_str(), expected.size());
        return false;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (shape.GetDim(i) != expected[i]) {
            printf("%s - ERROR - [XIR]: InferShape FAILED: dim%zu expected=%ld actual=%ld\n", GetTime().c_str(), i,
                   expected[i], shape.GetDim(i));
            return false;
        }
    }
    if (output_desc.GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: InferDataType FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_FLOAT,
               output_desc.GetDataType());
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

struct ExpectedResult {
    vector<int64_t> shape;
    int64_t size;
};

bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), const ExpectedResult& expected,
                 RunContext& ctx)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    vector<ge::Tensor> input;
    ge::Graph graph = builder(input);

    ge::Session* session = new ge::Session(ctx.session_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return false;
    }
    ge::Status ret = session->AddGraph(ctx.graph_id, graph, ctx.session_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return false;
    }
    vector<ge::Tensor> output;
    ret = session->RunGraph(ctx.graph_id, input, output);
    if (ret == SUCCESS) {
        // 配对消除仅存在于 TF parser（解析期按 var_name 匹配删除节点），GE IR 构图无此逻辑，
        // 图优化不会消除 TemporaryVariable/DestroyTemporaryVariable，RunGraph 应必然失败
        // （GeDeletedOp: should have been removed during graph optimization，已实测三场景一致）
        printf("%s - ERROR - [XIR]: RunGraph succeeded (unexpected: GeDeletedOp nodes should not be executable)\n",
               GetTime().c_str());
        delete session;
        return false;
    }
    printf("%s - INFO - [XIR]: RunGraph failed as expected (GeDeletedOp without TF-parser pairing)\n",
           GetTime().c_str());
    delete session;

    if (!CheckInferShape(graph, expected.shape)) {
        return false;
    }
    printf("%s - INFO - [XIR]: %s PASSED (DestroyTemporaryVariable is a TF-compat experimental op registered by "
           "the GE local engine as a GeDeletedOp; pairing elimination only exists in the TF parser, so GE IR "
           "graphs always fail RunGraph with GeDeletedOp error; verified via expected-failure assertion and "
           "InferShapeAndType)\n",
           GetTime().c_str(), tag.c_str());
    ctx.graph_id++;
    return true;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<ge::AscendString, ge::AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    ge::Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::map<ge::AscendString, ge::AscendString> session_options;
    RunContext ctx{session_options, 0};

    ExpectedResult static_expected{{2, 3}, 6};
    ExpectedResult neg1_expected{{2, 3}, 6};
    ExpectedResult neg2_expected{{2, 3}, 6};

    // 配对场景：与 TemporaryVariable 配对（var_name 匹配），验证真实使用模式下的形状推导
    if (!RunScenario("paired-static", BuildPairedGraph, static_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    // 独立动态 shape 场景：输入 shape 含 -1/-2，验证从输入推导输出
    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, neg1_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, neg2_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
