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
#include "../op_graph/read_variable_op_proto.h"

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
    ge::Graph graph("test_read_variable_op_ir");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {2, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    auto read_variable_op_node = op::ReadVariableOp("read_variable_op_node");
    read_variable_op_node.set_input_x(data_node);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{read_variable_op_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(read_variable_op_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_read_variable_op_dynamic_dim_neg1");
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

    auto read_variable_op_node = op::ReadVariableOp("read_variable_op_node");
    read_variable_op_node.set_input_x(data_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{read_variable_op_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(read_variable_op_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_read_variable_op_dynamic_dim_neg2");
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

    auto read_variable_op_node = op::ReadVariableOp("read_variable_op_node");
    read_variable_op_node.set_input_x(data_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{read_variable_op_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(read_variable_op_node);
    return graph;
}

// 校验推导后的输出 desc：shape 与 expected 一致且 dtype 为 DT_FLOAT
bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator read_variable_op;
    graph.FindOpByName("read_variable_op_node", read_variable_op);
    auto output_desc = read_variable_op.GetOutputDesc(0);
    auto shape = output_desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" dtype=%d | expected: dim_num=%zu", output_desc.GetDataType(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        printf(" dim%zu=%ld", i, expected[i]);
    }
    printf(" dtype=%d\n", ge::DT_FLOAT);

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

// runtime_shape 为 RunGraph 执行后用户侧 graph desc 更新出的实际 shape；
// 静态场景编译期即确定，动态场景（-1/-2）依赖 RunGraph 把 {-1,3}/{-2} 更新为运行时值，
struct ExpectedResult {
    vector<int64_t> runtime_shape;
    int64_t size;
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
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return false;
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
        for (size_t d = 0; d < out_shape.GetDimNum(); d++) {
            printf(" dim%zu=%ld", d, out_shape.GetDim(d));
        }
        printf(" shape_size=%ld dtype=%d\n", out_shape.GetShapeSize(), output[i].GetTensorDesc().GetDataType());
    }

    if (!CheckInferShape(graph, expected.runtime_shape)) {
        delete session;
        return false;
    }

    if (output.empty() || output[0].GetTensorDesc().GetShape().GetShapeSize() != expected.size ||
        output[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: expected shape_size=%ld dtype=%d\n", GetTime().c_str(),
               expected.size, ge::DT_FLOAT);
        delete session;
        return false;
    }
    // ReadVariableOp 为直通算子：输出值应与输入一致
    {
        const auto* out_data = reinterpret_cast<const float*>(output[0].GetData());
        const auto* in_data = reinterpret_cast<const float*>(input[0].GetData());
        bool values_ok = true;
        for (int64_t i = 0; i < expected.size; i++) {
            if (out_data[i] != in_data[i]) {
                printf("%s - ERROR - [XIR]: Output value verification FAILED: out[%ld]=%.4f expected=%.4f\n",
                       GetTime().c_str(), i, out_data[i], in_data[i]);
                values_ok = false;
                break;
            }
        }
        if (!values_ok) {
            delete session;
            return false;
        }
        printf("%s - INFO - [XIR]: Output value verification success (passthrough: output == input)\n",
               GetTime().c_str());
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
    printf("%s - INFO - [XIR]: %s PASSED\n", GetTime().c_str(), tag.c_str());
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

    // 静态场景 shape 编译期确定；动态场景运行时喂数 {2,3}，RunGraph 后 desc 更新为实际 shape
    ExpectedResult static_expected{{2, 3}, 6};
    ExpectedResult neg1_expected{{2, 3}, 6};
    ExpectedResult neg2_expected{{2, 3}, 6};

    if (!RunScenario("static", BuildGraph, static_expected, ctx, "./read_variable_op_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, neg1_expected, ctx,
                     "./read_variable_op_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, neg2_expected, ctx,
                     "./read_variable_op_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
