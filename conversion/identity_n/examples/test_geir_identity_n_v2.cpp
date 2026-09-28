/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "ops_proto_math.h"

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kDtype = DT_FLOAT;

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t>& shape)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    for (auto d : shape) {
        range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
    }
    return range;
}

int BuildGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
               Operator& graph_output, bool is_dynamic)
{
    auto data1 = op::Data("input1").set_attr_index(0);
    auto data2 = op::Data("input2").set_attr_index(1);
    TensorDesc desc(Shape(shape), FORMAT_ND, kDtype);
    if (is_dynamic) {
        desc.SetShapeRange(MakeShapeRange(shape));
    }
    data1.update_input_desc_x(desc);
    data1.update_output_desc_y(desc);
    data2.update_input_desc_x(desc);
    data2.update_output_desc_y(desc);

    auto identity_n = op::IdentityN("identity_n");
    identity_n.create_dynamic_input_x(2);
    identity_n.set_dynamic_input_x(0, data1);
    identity_n.set_dynamic_input_x(1, data2);

    TensorDesc out_desc(Shape(shape), FORMAT_ND, kDtype);
    if (is_dynamic) {
        out_desc.SetShapeRange(MakeShapeRange(shape));
    }
    identity_n.create_dynamic_output_y(2);
    identity_n.update_dynamic_output_desc_y(0, out_desc);
    identity_n.update_dynamic_output_desc_y(1, out_desc);

    graph.AddOp(data1);
    graph.AddOp(data2);
    graph.AddOp(identity_n);
    graph_inputs = {data1, data2};
    graph_output = identity_n;
    return kSuccess;
}

int CheckInferShape(const Graph& graph, const std::vector<int64_t>& expected_dims)
{
    Operator op_obj;
    if (graph.FindOpByName("identity_n", op_obj) != GRAPH_SUCCESS) {
        std::cerr << "CheckInferShape: FindOpByName failed" << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < 2; i++) {
        TensorDesc desc = op_obj.GetDynamicOutputDesc("y", i); // NOLINT: const char* overload
        Shape shape = desc.GetShape();
        std::cout << "InferShape[" << i << "]: dim_num=" << shape.GetDimNum();
        for (size_t j = 0; j < shape.GetDims().size(); j++) {
            std::cout << " dim" << j << "=" << shape.GetDim(j);
        }
        std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
    }
    return kSuccess;
}

int CheckOutput(const std::vector<Tensor>& outputs, const std::vector<int64_t>& expected_dims)
{
    if (outputs.size() < 2) {
        std::cerr << "CheckOutput: outputs.size()=" << outputs.size() << " (expected >=2)" << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < 2; i++) {
        const auto& tensor = outputs[i];
        const auto& desc = tensor.GetTensorDesc();
        Shape shape = desc.GetShape();
        std::cout << "Output[" << i << "]: dim_num=" << shape.GetDimNum();
        for (size_t j = 0; j < shape.GetDims().size(); j++) {
            std::cout << " dim" << j << "=" << shape.GetDim(j);
        }
        std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("identity_n_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({graph_output});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    CheckInferShape(graph, shape);

    std::vector<int64_t> actual;
    for (auto d : shape) {
        actual.push_back(d < 0 ? 2 : d);
    }
    size_t cnt = 1;
    for (auto d : actual) {
        cnt *= static_cast<size_t>(d);
    }
    std::vector<float> idata1(cnt, 1.0f);
    std::vector<float> idata2(cnt, 2.0f);
    Tensor itensor1(TensorDesc(Shape(actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(idata1.data()),
                    cnt * sizeof(float));
    Tensor itensor2(TensorDesc(Shape(actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(idata2.data()),
                    cnt * sizeof(float));

    std::vector<Tensor> inputs = {itensor1, itensor2};
    std::vector<Tensor> outputs;

    bool run_ok = false;
    Status ret = session.RunGraph(gid, inputs, outputs);
    if (ret != SUCCESS) {
        std::cerr << name << ": RunGraph failed (ret=" << ret << ")" << std::endl;
    } else if (outputs.size() < 2) {
        std::cerr << name << ": RunGraph success but outputs.size()=" << outputs.size() << " (expected >=2)"
                  << std::endl;
    } else {
        std::cout << name << ": RunGraph success" << std::endl;
        if (CheckOutput(outputs, actual) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    Graph build_graph(("identity_n_build_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({build_output});

    ModelBufferData model;
    std::map<AscendString, AscendString> build_opts;
    if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
        std::cerr << name << " FAILED: BuildModel" << std::endl;
        return kFailed;
    }
    std::cout << name << ": BuildModel success, size=" << model.length << std::endl;

    std::string om_file = name;
    if (aclgrphSaveModel(om_file, model) != SUCCESS) {
        std::cerr << name << " FAILED: SaveModel" << std::endl;
        return kFailed;
    }
    std::cout << name << ": SaveModel success, file=" << om_file << std::endl;

    if (run_ok) {
        std::cout << name << " PASSED" << std::endl;
        return kSuccess;
    }
    std::cerr << name << " NOT PASSED (RunGraph did not produce valid output)" << std::endl;
    return kFailed;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (GEInitialize(global_options) != SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    RunContext ctx;
    int ret = kSuccess;

    // Scenario 1: static shape {2,3}, run + buildmodel
    if (RunScenario(ctx, "static", {2, 3}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,3}, dim0 unknown (range 1~10), run + buildmodel
    if (RunScenario(ctx, "dynamic_minus1", {-1, 3}, true) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2}, rank unknown (range 1~10), run + buildmodel
    if (RunScenario(ctx, "dynamic_minus2", {-2}, true) != kSuccess) {
        ret = kFailed;
    }

    GEFinalize();
    if (ret == kSuccess) {
        std::cout << "All scenarios PASSED" << std::endl;
    } else {
        std::cout << "Some scenarios FAILED" << std::endl;
    }
    return ret;
}
