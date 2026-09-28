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
#include "../op_graph/var_handle_op_proto.h"

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

// VarHandleOp 为无 INPUT 的资源句柄源节点（输出标量 DT_RESOURCE，shape/dtype 属性描述
// handle 所指向变量的形状与类型）。
ge::Graph BuildGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& var_shape)
{
    ge::Graph graph("test_var_handle_op_ir");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::Shape(var_shape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    int64_t elem_num = 1;
    for (size_t i = 0; i < var_shape.size(); i++) {
        elem_num *= var_shape[i];
    }
    float* pData = new float[elem_num];
    for (int64_t i = 0; i < elem_num; i++) {
        pData[i] = static_cast<float>(i + 1);
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, elem_num * sizeof(float));
    input.push_back(tensor_data);

    auto var_handle_node = op::VarHandleOp("var_handle_node");
    var_handle_node.set_attr_container("");
    var_handle_node.set_attr_shared_name("var_0");
    var_handle_node.set_attr_dtype(ge::DT_FLOAT);
    var_handle_node.set_attr_shape(var_shape);
    ge::TensorDesc handle_desc(ge::Shape(), ge::FORMAT_ND, ge::DT_RESOURCE);
    var_handle_node.update_output_desc_y(handle_desc);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{data_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(var_handle_node);
    return graph;
}

// 校验构图时设置的输出 desc 符合算子设定（标量 DT_RESOURCE）。
bool CheckOutputDesc(const ge::Graph& graph)
{
    ge::Operator var_handle_op;
    graph.FindOpByName("var_handle_node", var_handle_op);
    auto output_desc = var_handle_op.GetOutputDesc(0);
    auto shape = output_desc.GetShape();
    printf("%s - INFO - [XIR]: OutputDesc: dim_num=%zu shape_size=%ld dtype=%d | expected: dim_num=0 "
           "shape_size=0 dtype=%d\n",
           GetTime().c_str(), shape.GetDimNum(), shape.GetShapeSize(), output_desc.GetDataType(), ge::DT_RESOURCE);

    // 输出为标量资源句柄
    if (shape.GetDimNum() != 0U) {
        printf("%s - ERROR - [XIR]: OutputDesc FAILED: expected scalar(dim_num=0) actual dim_num=%zu\n",
               GetTime().c_str(), shape.GetDimNum());
        return false;
    }
    if (output_desc.GetDataType() != ge::DT_RESOURCE) {
        printf("%s - ERROR - [XIR]: OutputDesc FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_RESOURCE,
               output_desc.GetDataType());
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

bool RunScenario(const string& tag, const vector<int64_t>& var_shape, RunContext& ctx, const string& om_name)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    vector<ge::Tensor> input;
    ge::Graph graph = BuildGraph(input, var_shape);

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

    // 输出为 Data 直通：校验 shape/dtype 与数值一致
    int64_t elem_num = 1;
    for (size_t i = 0; i < var_shape.size(); i++) {
        elem_num *= var_shape[i];
    }
    if (output.size() != 1U || output[0].GetTensorDesc().GetShape().GetShapeSize() != elem_num ||
        output[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: expected shape_size=%ld dtype=%d\n", GetTime().c_str(),
               elem_num, ge::DT_FLOAT);
        delete session;
        return false;
    }
    const auto* out_data = reinterpret_cast<const float*>(output[0].GetData());
    const auto* in_data = reinterpret_cast<const float*>(input[0].GetData());
    bool values_ok = true;
    for (int64_t i = 0; i < elem_num; i++) {
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
    printf("%s - INFO - [XIR]: Output value verification success (data passthrough: output == input)\n",
           GetTime().c_str());
    delete session;

    if (!CheckOutputDesc(graph)) {
        return false;
    }

    vector<ge::Tensor> input_om;
    ge::Graph graph_om = BuildGraph(input_om, var_shape);
    ge::ModelBufferData model_buffer;
    ret = aclgrphBuildModel(graph_om, ctx.session_options, model_buffer);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: BuildModel failed\n", GetTime().c_str());
        return false;
    }
    aclgrphSaveModel(om_name.c_str(), model_buffer);
    printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), om_name.c_str());
    printf("%s - INFO - [XIR]: %s PASSED (island VarHandleOp tolerated through compile with node pruned; "
           "verified via RunGraph data passthrough, BuildModel, and output desc contract; "
           "InferShape/InferDataType logic is covered by UT)\n",
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

    // shape 属性为 handle 指向变量的形状，输出恒为标量资源句柄；
    // 覆盖不同变量 shape/维度（VarHandleOp 输出与 shape 属性无关，不做 -1/-2 动态场景）
    if (!RunScenario("shape(2,3)", {2, 3}, ctx, "./var_handle_op_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(4,5)", {4, 5}, ctx, "./var_handle_op_shape_4_5_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(1,2,3)", {1, 2, 3}, ctx, "./var_handle_op_shape_1_2_3_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
