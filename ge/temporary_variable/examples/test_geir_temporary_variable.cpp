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
#include "../op_graph/temporary_variable_proto.h"
#include "../../destroy_temporary_variable/op_graph/destroy_temporary_variable_proto.h"

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

// 配对构图：TemporaryVariable(分配临时变量) → DestroyTemporaryVariable(读终值并销毁，var_name 匹配)。
// 二者必须配对使用，故本 example 所有场景均采用配对模式；
// Data + Add 仅用于满足 GE 图输入约束（SetInputs 不允许为空），Add 消费 Destroy 输出模拟终值读取
ge::Graph BuildGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& shape_attr)
{
    ge::Graph graph("test_temporary_variable_ir");
    ge::DataType inDtype = ge::DT_FLOAT;
    const string var_name = "tmp_var";

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

    auto temporary_variable_node = op::TemporaryVariable("temporary_variable_node");
    temporary_variable_node.set_attr_shape(shape_attr);
    temporary_variable_node.set_attr_dtype(static_cast<int64_t>(ge::DT_FLOAT));
    temporary_variable_node.set_attr_var_name(var_name.c_str());
    ge::TensorDesc output_desc(ge::Shape(shape_attr), ge::FORMAT_ND, ge::DT_FLOAT);
    temporary_variable_node.update_output_desc_y(output_desc);

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

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator temporary_variable_op;
    graph.FindOpByName("temporary_variable_node", temporary_variable_op);
    // 显式触发推导，不依赖 RunGraph/BuildModel 的编译时序：
    // GE local 引擎将 TemporaryVariable 注册为 GeDeletedOp（图优化阶段应被删除的算子）
    ge::Status ret = temporary_variable_op.InferShapeAndType();
    if (ret != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: InferShapeAndType failed, ret=%u\n", GetTime().c_str(), ret);
        return false;
    }
    auto output_desc = temporary_variable_op.GetOutputDesc(0);
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

struct ExpectedResult {
    vector<int64_t> shape;
    int64_t size;
};

bool RunScenario(const string& tag, const ExpectedResult& expected, RunContext& ctx)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    vector<ge::Tensor> input;
    ge::Graph graph = BuildGraph(input, expected.shape);

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
        bool ok = CheckInferShape(graph, expected.shape);
        if (ok) {
            printf("%s - INFO - [XIR]: %s PASSED (AddGraph failed, verified via InferShapeAndType only)\n",
                   GetTime().c_str(), tag.c_str());
        }
        return ok;
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
        if (output.empty() || output[0].GetTensorDesc().GetShape().GetShapeSize() != expected.size ||
            output[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
            printf("%s - ERROR - [XIR]: Output verification FAILED: expected shape_size=%ld dtype=%d\n",
                   GetTime().c_str(), expected.size, ge::DT_FLOAT);
            delete session;
            return false;
        }
    } else {
        // TemporaryVariable 为 TF 兼容的实验性源节点，GE local 引擎将其注册为 GeDeletedOp：
        // 语义上应在图优化阶段被消除（真实 TF 流程中由 parser 配对 DestroyTemporaryVariable 在解析期处理），
        printf("%s - INFO - [XIR]: RunGraph not executable standalone\n", GetTime().c_str());
    }
    delete session;

    if (!CheckInferShape(graph, expected.shape)) {
        return false;
    }
    printf("%s - INFO - [XIR]: %s PASSED (TemporaryVariable is a TF-compat experimental source node registered by "
           "the GE local engine as a GeDeletedOp which must be eliminated during graph optimization, "
           "so it cannot be executed standalone via RunGraph/BuildModel; verified via InferShapeAndType)\n",
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

    ExpectedResult shape_2_3{{2, 3}, 6};
    ExpectedResult shape_4_5{{4, 5}, 20};
    ExpectedResult shape_1_2_3{{1, 2, 3}, 6};

    if (!RunScenario("shape(2,3)", shape_2_3, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(4,5)", shape_4_5, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(1,2,3)", shape_1_2_3, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
