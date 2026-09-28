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
#include "../op_graph/queue_data_proto.h"

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

static ge::Graph BuildGraphImpl(const string& graph_name, const vector<int64_t>& output_shape_attr,
                                int64_t expected_size)
{
    ge::Graph graph(graph_name.c_str());

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc(ge::Shape({expected_size}), ge::FORMAT_ND, ge::DT_UINT8);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    auto queue_data_node = op::QueueData("queue_data_node");
    queue_data_node.set_attr_index(0);
    queue_data_node.set_attr_queue_name("");
    queue_data_node.set_attr_output_types({ge::DT_INT8});
    queue_data_node.set_attr_output_shapes({output_shape_attr});
    ge::TensorDesc output_desc(ge::Shape({expected_size}), ge::FORMAT_ND, ge::DT_UINT8);
    queue_data_node.update_output_desc_y(output_desc);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(queue_data_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(queue_data_node);
    graph.AddOp(add_node);
    return graph;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
};

struct ExpectedResult {
    vector<int64_t> shape_attr;
    int64_t expected_size;
};

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator queue_data_op;
    graph.FindOpByName("queue_data_node", queue_data_op);
    auto shape = queue_data_op.GetOutputDesc(0).GetShape();
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

bool RunScenario(const string& tag, const ExpectedResult& expected, RunContext& ctx, const string& om_name)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    string graph_name = "test_queue_data_" + tag;
    ge::Graph graph = BuildGraphImpl(graph_name, expected.shape_attr, expected.expected_size);

    ge::ModelBufferData model_buffer;
    ge::Status ret = aclgrphBuildModel(graph, ctx.session_options, model_buffer);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: BuildModel failed\n", GetTime().c_str());
        return false;
    }

    if (!CheckInferShape(graph, {expected.expected_size})) {
        return false;
    }
    aclgrphSaveModel(om_name.c_str(), model_buffer);
    printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), om_name.c_str());
    printf("%s - INFO - [XIR]: %s PASSED\n", GetTime().c_str(), tag.c_str());
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
    RunContext ctx{session_options};

    ExpectedResult shape_2_3{{2, 3}, 86};
    ExpectedResult shape_4_5{{4, 5}, 100};
    ExpectedResult shape_1_2_3{{1, 2, 3}, 94};

    if (!RunScenario("shape(2,3)", shape_2_3, ctx, "./queue_data_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(4,5)", shape_4_5, ctx, "./queue_data_shape_4_5_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(1,2,3)", shape_1_2_3, ctx, "./queue_data_shape_1_2_3_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
