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
#include "../op_graph/file_constant_proto.h"

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

static bool CreateWeightFile(const string& path, int64_t elem_num)
{
    std::ofstream ofs(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        printf("%s - ERROR - [XIR]: Create weight file failed: %s\n", GetTime().c_str(), path.c_str());
        return false;
    }
    // 非零权重（全 1.0）：若权重未被加载（缺省为 0），Add 结果将与预期不符，从而暴露加载失败
    vector<float> weight(elem_num, 1.0f);
    ofs.write(reinterpret_cast<const char*>(weight.data()), weight.size() * sizeof(float));
    ofs.close();
    return true;
}

ge::Graph BuildGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& shape_attr, const string& weight_file)
{
    ge::Graph graph("test_file_constant_ir");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::Shape(shape_attr), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    int64_t elem_num = 1;
    for (size_t i = 0; i < shape_attr.size(); i++) {
        elem_num *= shape_attr[i];
    }
    float* pData = new float[elem_num];
    for (int64_t i = 0; i < elem_num; i++) {
        pData[i] = static_cast<float>(i + 1);
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, elem_num * sizeof(float));
    input.push_back(tensor_data);

    auto file_constant_node = op::FileConstant("file_constant_node");
    file_constant_node.set_attr_file_path(weight_file.c_str());
    file_constant_node.set_attr_file_id("file");
    file_constant_node.set_attr_shape(shape_attr);
    file_constant_node.set_attr_dtype(ge::DT_FLOAT);
    ge::TensorDesc output_desc(ge::Shape(shape_attr), ge::FORMAT_ND, ge::DT_FLOAT);
    file_constant_node.update_output_desc_y(output_desc);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(file_constant_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(file_constant_node);
    graph.AddOp(add_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator file_constant_op;
    graph.FindOpByName("file_constant_node", file_constant_op);
    auto shape = file_constant_op.GetOutputDesc(0).GetShape();
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

bool RunScenario(const string& tag, const ExpectedResult& expected, RunContext& ctx, const string& om_name)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    string weight_file = "./file_constant_weight_" + tag + ".bin";
    if (!CreateWeightFile(weight_file, expected.size)) {
        return false;
    }

    vector<ge::Tensor> input;
    ge::Graph graph = BuildGraph(input, expected.shape, weight_file);

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

    if (!output.empty() && output[0].GetTensorDesc().GetShape().GetShapeSize() != expected.size) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: expected shape_size=%ld\n", GetTime().c_str(),
               expected.size);
        delete session;
        return false;
    }

    if (!output.empty()) {
        const auto* out_data = reinterpret_cast<const float*>(output[0].GetData());
        const auto* in_data = reinterpret_cast<const float*>(input[0].GetData());
        constexpr float kWeightValue = 1.0f;
        bool values_ok = true;
        for (int64_t i = 0; i < expected.size; i++) {
            if (out_data[i] != in_data[i] + kWeightValue) {
                printf("%s - ERROR - [XIR]: Output value verification FAILED: out[%ld]=%.4f expected=%.4f "
                       "(input %.4f + weight %.4f)\n",
                       GetTime().c_str(), i, out_data[i], in_data[i] + kWeightValue, in_data[i], kWeightValue);
                values_ok = false;
                break;
            }
        }
        if (!values_ok) {
            delete session;
            return false;
        }
        printf("%s - INFO - [XIR]: Output value verification success (input + non-zero weight from file)\n",
               GetTime().c_str());
    }
    delete session;

    vector<ge::Tensor> input_om;
    ge::Graph graph_om = BuildGraph(input_om, expected.shape, weight_file);
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

    ExpectedResult shape_2_3{{2, 3}, 6};
    ExpectedResult shape_4_5{{4, 5}, 20};
    ExpectedResult shape_1_2_3{{1, 2, 3}, 6};

    if (!RunScenario("static", shape_2_3, ctx, "./file_constant_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(4,5)", shape_4_5, ctx, "./file_constant_shape_4_5_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(1,2,3)", shape_1_2_3, ctx, "./file_constant_shape_1_2_3_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
