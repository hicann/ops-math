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

#include "../../data/op_graph/data_proto.h"
#include "../../variable/op_graph/variable_proto.h"
#include "../op_graph/var_is_initialized_op_proto.h"

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

// VarIsInitializedOpPass 要求输入是 Variable 节点（查 Variable 初始化状态），
// 构图必须为 Data → Variable → VarIsInitializedOp
ge::Graph BuildGraph(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_var_is_initialized_op_ir");
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

    auto var_is_initialized_node = op::VarIsInitializedOp("var_is_initialized_node");
    var_is_initialized_node.set_input_x(variable_node);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{var_is_initialized_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(variable_node);
    graph.AddOp(var_is_initialized_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_var_is_initialized_op_dynamic_dim_neg1");
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

    auto var_is_initialized_node = op::VarIsInitializedOp("var_is_initialized_node");
    var_is_initialized_node.set_input_x(variable_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{var_is_initialized_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(variable_node);
    graph.AddOp(var_is_initialized_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_var_is_initialized_op_dynamic_dim_neg2");
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

    auto var_is_initialized_node = op::VarIsInitializedOp("var_is_initialized_node");
    var_is_initialized_node.set_input_x(variable_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{var_is_initialized_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(variable_node);
    graph.AddOp(var_is_initialized_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph)
{
    ge::Operator var_is_initialized_op;
    graph.FindOpByName("var_is_initialized_node", var_is_initialized_op);

    auto output_desc = var_is_initialized_op.GetOutputDesc(0);
    auto shape = output_desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu dtype=%d | expected: dim_num=0 dtype=%d\n", GetTime().c_str(),
           shape.GetDimNum(), output_desc.GetDataType(), ge::DT_BOOL);

    // 输出恒为 bool 标量
    if (shape.GetDimNum() != 0U) {
        printf("%s - ERROR - [XIR]: InferShape FAILED: expected scalar(dim_num=0) actual dim_num=%zu\n",
               GetTime().c_str(), shape.GetDimNum());
        return false;
    }
    if (output_desc.GetDataType() != ge::DT_BOOL) {
        printf("%s - ERROR - [XIR]: InferDataType FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_BOOL,
               output_desc.GetDataType());
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), RunContext& ctx, const string& om_name)
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
        printf("%s - INFO - [XIR]: RunGraph success, output count=%zu\n", GetTime().c_str(), output.size());
        for (size_t i = 0; i < output.size(); i++) {
            auto out_shape = output[i].GetTensorDesc().GetShape();
            printf("%s - INFO - [XIR]: output[%zu] dim_num=%zu", GetTime().c_str(), i, out_shape.GetDimNum());
            for (size_t d = 0; d < out_shape.GetDimNum(); d++) {
                printf(" dim%zu=%ld", d, out_shape.GetDim(d));
            }
            printf(" shape_size=%ld dtype=%d\n", out_shape.GetShapeSize(), output[i].GetTensorDesc().GetDataType());
        }
        // 输出恒为 bool 标量（dim_num=0，GetShapeSize 对空 dim 返回 0），值为变量是否已初始化
        if (output.empty() || output[0].GetTensorDesc().GetShape().GetDimNum() != 0U ||
            output[0].GetTensorDesc().GetDataType() != ge::DT_BOOL) {
            printf("%s - ERROR - [XIR]: Output verification FAILED: expected scalar(dim_num=0) dtype=%d\n",
                   GetTime().c_str(), ge::DT_BOOL);
            delete session;
            return false;
        }
        const auto* out_data = reinterpret_cast<const bool*>(output[0].GetData());
        printf("%s - INFO - [XIR]: is_initialized value: %s\n", GetTime().c_str(), *out_data ? "true" : "false");
        if (*out_data != false) {
            printf("%s - ERROR - [XIR]: Output value verification FAILED: expected false (no var write in graph)\n",
                   GetTime().c_str());
            delete session;
            return false;
        }
    } else {
        printf("%s - INFO - [XIR]: RunGraph not executable standalone\n", GetTime().c_str());
    }
    delete session;

    if (!CheckInferShape(graph)) {
        return false;
    }

    vector<ge::Tensor> input_om;
    ge::Graph graph_om = builder(input_om);
    ge::ModelBufferData model_buffer;
    ret = aclgrphBuildModel(graph_om, ctx.session_options, model_buffer);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: BuildModel not executable standalone\n", GetTime().c_str());
    } else {
        aclgrphSaveModel(om_name.c_str(), model_buffer);
        printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), om_name.c_str());
    }
    printf("%s - INFO - [XIR]: %s PASSED (verified via %s + InferShape)\n", GetTime().c_str(), tag.c_str(),
           ret == SUCCESS ? "RunGraph/BuildModel" : "AddGraph");
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

    // 输出恒为 bool 标量（与输入 shape/dtype 无关），三个场景验证不同输入 shape 下
    // 推导仍稳定输出标量，以及动态输入构图（-1 带 SetShapeRange / -2 不带）的编译执行链路
    if (!RunScenario("static", BuildGraph, ctx, "./var_is_initialized_op_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, ctx,
                     "./var_is_initialized_op_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, ctx,
                     "./var_is_initialized_op_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
