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

#include "../op_graph/ref_data_proto.h"

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
    ge::Graph graph("test_ref_data_ir");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {2, 3};

    auto ref_data_node = op::RefData("ref_data_node");
    ref_data_node.set_attr_index(0);
    ge::TensorDesc desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    ref_data_node.update_input_desc_x(desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{ref_data_node};
    std::vector<ge::Operator> outputs{ref_data_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(ref_data_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_ref_data_dynamic_dim_neg1");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-1, 3};

    auto ref_data_node = op::RefData("ref_data_node");
    ref_data_node.set_attr_index(0);
    ge::TensorDesc desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetShapeRange({{1, 10}, {3, 3}});
    ref_data_node.update_input_desc_x(desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{ref_data_node};
    std::vector<ge::Operator> outputs{ref_data_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(ref_data_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_ref_data_dynamic_dim_neg2");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-2};

    auto ref_data_node = op::RefData("ref_data_node");
    ref_data_node.set_attr_index(0);
    ge::TensorDesc desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    // unknown rank(-2) has unknown dim num, no shape range is set; the real shape is fed on RunGraph
    ref_data_node.update_input_desc_x(desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{ref_data_node};
    std::vector<ge::Operator> outputs{ref_data_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(ref_data_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator ref_data_op;
    graph.FindOpByName("ref_data_node", ref_data_op);
    auto shape = ref_data_op.GetOutputDesc(0).GetShape();
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

    if (!CheckInferShape(graph, expected.shape)) {
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

    ExpectedResult static_expected{{2, 3}, 6};
    ExpectedResult neg1_expected{{2, 3}, 6};
    ExpectedResult neg2_expected{{2, 3}, 6};

    if (!RunScenario("static", BuildGraph, static_expected, ctx, "./ref_data_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, neg1_expected, ctx,
                     "./ref_data_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, neg2_expected, ctx,
                     "./ref_data_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
