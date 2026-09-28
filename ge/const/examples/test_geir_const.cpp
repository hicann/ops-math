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

#include "elewise_calculation_ops.h"
#include "../../data/op_graph/data_proto.h"
#include "../op_graph/const_proto.h"

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
    ge::Graph graph("test_const_ir");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::Shape({3}), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    float* pData = new float[3];
    pData[0] = 10.0f;
    pData[1] = 20.0f;
    pData[2] = 30.0f;
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 12);
    input.push_back(tensor_data);

    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, inDtype);
    const_desc.SetSize(3 * sizeof(float));
    const_tensor.SetTensorDesc(const_desc);
    float const_data[3] = {1.0f, 2.0f, 3.0f};
    const_tensor.SetData((uint8_t*)const_data, 3 * sizeof(float));

    auto const_node = op::Const("const_node");
    const_node.set_attr_value(const_tensor);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(const_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(const_node);
    graph.AddOp(add_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_const_dynamic_dim_neg1");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::TensorDesc(ge::Shape({-1, 3}), ge::FORMAT_ND, inDtype));
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_desc.SetShapeRange({{1, 10}, {3, 3}});
    data_node.update_input_desc_x(data_desc);

    float* pData = new float[6];
    pData[0] = 10.0f;
    pData[1] = 20.0f;
    pData[2] = 30.0f;
    pData[3] = 40.0f;
    pData[4] = 50.0f;
    pData[5] = 60.0f;
    ge::TensorDesc real_desc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    const_desc.SetSize(6 * sizeof(float));
    const_tensor.SetTensorDesc(const_desc);
    float const_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const_tensor.SetData((uint8_t*)const_data, 6 * sizeof(float));

    auto const_node = op::Const("const_node");
    const_node.set_attr_value(const_tensor);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(const_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(const_node);
    graph.AddOp(add_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_const_dynamic_dim_neg2");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::TensorDesc(ge::Shape({-2}), ge::FORMAT_ND, inDtype));
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    // unknown rank(-2) has unknown dim num, no shape range is set; the real shape is fed on RunGraph
    data_node.update_input_desc_x(data_desc);

    float* pData = new float[6];
    pData[0] = 10.0f;
    pData[1] = 20.0f;
    pData[2] = 30.0f;
    pData[3] = 40.0f;
    pData[4] = 50.0f;
    pData[5] = 60.0f;
    ge::TensorDesc real_desc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    const_desc.SetSize(6 * sizeof(float));
    const_tensor.SetTensorDesc(const_desc);
    float const_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const_tensor.SetData((uint8_t*)const_data, 6 * sizeof(float));

    auto const_node = op::Const("const_node");
    const_node.set_attr_value(const_tensor);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(const_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(const_node);
    graph.AddOp(add_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator const_op;
    graph.FindOpByName("const_node", const_op);
    auto shape = const_op.GetOutputDesc(0).GetShape();
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
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

struct ExpectedResult {
    vector<int64_t> shape;
    vector<float> values;
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

    if (!CheckInferShape(graph, expected.shape)) {
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

    if (!output.empty() && !expected.values.empty()) {
        auto* data = reinterpret_cast<const float*>(output[0].GetData());
        bool values_ok = true;
        for (size_t i = 0; i < expected.values.size(); i++) {
            printf("%s - INFO - [XIR]: output[%zu]=%.1f expected=%.1f\n", GetTime().c_str(), i, data[i],
                   expected.values[i]);
            if (data[i] != expected.values[i]) {
                values_ok = false;
            }
        }
        if (!values_ok) {
            printf("%s - ERROR - [XIR]: Output value verification FAILED\n", GetTime().c_str());
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

    ExpectedResult static_expected{{3}, {11.0f, 22.0f, 33.0f}};
    ExpectedResult dynamic_expected{{2, 3}, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}};

    if (!RunScenario("static", BuildGraph, static_expected, ctx, "./const_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, dynamic_expected, ctx,
                     "./const_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, dynamic_expected, ctx,
                     "./const_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
