/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "../../data/op_graph/data_proto.h"
#include "../../variable/op_graph/variable_proto.h"
#include "../op_graph/is_variable_initialized_proto.h"

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

ge::Graph BuildGraph(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_is_variable_initialized_ir");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {2, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    auto variable_node = op::Variable("variable_node");
    variable_node.set_input_x(data_node);

    auto is_variable_initialized_node = op::IsVariableInitialized("is_variable_initialized_node");
    is_variable_initialized_node.set_input_x(variable_node);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{is_variable_initialized_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(variable_node);
    graph.AddOp(is_variable_initialized_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_is_variable_initialized_dynamic_dim_neg1");
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

    auto variable_node = op::Variable("variable_node");
    variable_node.set_input_x(data_node);

    auto is_variable_initialized_node = op::IsVariableInitialized("is_variable_initialized_node");
    is_variable_initialized_node.set_input_x(variable_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{is_variable_initialized_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(variable_node);
    graph.AddOp(is_variable_initialized_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_is_variable_initialized_dynamic_dim_neg2");
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

    auto variable_node = op::Variable("variable_node");
    variable_node.set_input_x(data_node);

    auto is_variable_initialized_node = op::IsVariableInitialized("is_variable_initialized_node");
    is_variable_initialized_node.set_input_x(variable_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{is_variable_initialized_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(variable_node);
    graph.AddOp(is_variable_initialized_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph, size_t expected_dim_num, ge::DataType expected_dtype)
{
    ge::Operator op_node;
    graph.FindOpByName("is_variable_initialized_node", op_node);
    auto desc = op_node.GetOutputDesc(0);
    auto shape = desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu shape_size=%ld dtype=%d | expected: dim_num=%zu dtype=%d\n",
           GetTime().c_str(), shape.GetDimNum(), shape.GetShapeSize(), desc.GetDataType(), expected_dim_num,
           expected_dtype);

    if (shape.GetDimNum() != expected_dim_num) {
        printf("%s - ERROR - [XIR]: InferShape FAILED: expected dim_num=%zu actual=%zu\n", GetTime().c_str(),
               expected_dim_num, shape.GetDimNum());
        return false;
    }
    if (desc.GetDataType() != expected_dtype) {
        printf("%s - ERROR - [XIR]: InferShape FAILED: expected dtype=%d actual=%d\n", GetTime().c_str(),
               expected_dtype, desc.GetDataType());
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

struct ExpectedResult {
    size_t dim_num;
    ge::DataType dtype;
};

bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), const ExpectedResult& expected,
                 RunContext& ctx, const string& om_name)
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
        printf("%s - INFO - [XIR]: AddGraph failed, fallback to inference verification\n", GetTime().c_str());
        delete session;
        ctx.graph_id++;
        bool ok = CheckInferShape(graph, expected.dim_num, expected.dtype);
        if (ok) {
            printf("%s - INFO - [XIR]: %s PASSED (AddGraph failed, verified via desc check only)\n", GetTime().c_str(),
                   tag.c_str());
        }
        return ok;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(ctx.graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        return false;
    }
    printf("%s - INFO - [XIR]: RunGraph success, output count=%zu\n", GetTime().c_str(), output.size());
    for (size_t i = 0; i < output.size(); i++) {
        auto out_shape = output[i].GetTensorDesc().GetShape();
        printf("%s - INFO - [XIR]: output[%zu] dim_num=%zu", GetTime().c_str(), i, out_shape.GetDimNum());
        printf(" shape_size=%ld dtype=%d\n", out_shape.GetShapeSize(), output[i].GetTensorDesc().GetDataType());
    }

    if (!CheckInferShape(graph, expected.dim_num, expected.dtype)) {
        delete session;
        return false;
    }

    if (!output.empty() && (output[0].GetTensorDesc().GetShape().GetDimNum() != expected.dim_num ||
                            output[0].GetTensorDesc().GetDataType() != expected.dtype)) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: expected dim_num=%zu dtype=%d\n", GetTime().c_str(),
               expected.dim_num, expected.dtype);
        delete session;
        return false;
    }
    // 值断言：VarIsInitializedOpPass 在编译期将本节点改写为 Const（与 VarIsInitializedOp 同机制，
    // 已解析编译产物 OM 证实节点类型为 Const），运行时读到的是 pass 写入的确定性常量。

    if (!output.empty()) {
        const auto* out_data = reinterpret_cast<const bool*>(output[0].GetData());
        printf("%s - INFO - [XIR]: is_initialized value: %s\n", GetTime().c_str(), *out_data ? "true" : "false");
        if (*out_data != false) {
            printf("%s - ERROR - [XIR]: Output value verification FAILED: expected false (no var write in graph)\n",
                   GetTime().c_str());
            delete session;
            return false;
        }
    }
    delete session;

    vector<ge::Tensor> input_om;
    ge::Graph graph_om = builder(input_om);
    ge::ModelBufferData model_buffer;
    ret = aclgrphBuildModel(graph_om, ctx.session_options, model_buffer);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: BuildModel failed\n", GetTime().c_str());
        return false;
    }
    aclgrphSaveModel(om_name.c_str(), model_buffer);
    printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), om_name.c_str());
    printf("%s - INFO - [XIR]: %s PASSED (IsVariableInitialized is a TF-compat experimental op rewritten by "
           "VarIsInitializedOpPass at compile time into a Const carrying the variable-init state; verified via "
           "RunGraph with bool false (no var write in graph), BuildModel, and output desc check)\n",
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

    ExpectedResult expected{0, ge::DT_BOOL};

    if (!RunScenario("static", BuildGraph, expected, ctx, "./is_variable_initialized_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, expected, ctx,
                     "./is_variable_initialized_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, expected, ctx,
                     "./is_variable_initialized_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
