/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <ctime>
#include <iostream>
#include <map>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "nn_other.h"
#include "tensor.h"
#include "types.h"

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr size_t kExpectedOutputDimNum = 2U;
constexpr size_t kExpectedOutputElementNum = 6U;
constexpr float kFloatCompareTolerance = 1e-5F;
const std::vector<int64_t> kExpectedOutputShape = {3, 2};

std::string GetTime()
{
    time_t now;
    time(&now);
    char buffer[64] = {0};
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buffer;
}

std::string ShapeToString(const ge::Shape& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t idx = 0U; idx < shape.GetDimNum(); ++idx) {
        if (idx != 0U) {
            oss << ",";
        }
        oss << shape.GetDim(idx);
    }
    oss << "]";
    return oss.str();
}

std::string VectorToString(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t idx = 0U; idx < shape.size(); ++idx) {
        if (idx != 0U) {
            oss << ",";
        }
        oss << shape[idx];
    }
    oss << "]";
    return oss.str();
}

ge::TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, ge::DataType dtype)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    return desc;
}

ge::Tensor MakeFloatTensor(const std::vector<int64_t>& shape)
{
    ge::TensorDesc desc = MakeTensorDesc(shape, ge::DT_FLOAT);

    size_t elem_count = 1U;
    for (auto dim : shape) {
        elem_count *= static_cast<size_t>(dim);
    }
    auto* buffer = new (std::nothrow) float[elem_count];
    for (size_t idx = 0U; idx < elem_count; ++idx) {
        buffer[idx] = static_cast<float>(idx + 1U);
    }
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(buffer), elem_count * sizeof(float));
    delete[] buffer;
    return tensor;
}

ge::Tensor MakeInt64Tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& values)
{
    ge::TensorDesc desc = MakeTensorDesc(shape, ge::DT_INT64);

    auto* buffer = new (std::nothrow) int64_t[values.size()];
    for (size_t idx = 0U; idx < values.size(); ++idx) {
        buffer[idx] = values[idx];
    }
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(buffer), values.size() * sizeof(int64_t));
    delete[] buffer;
    return tensor;
}

int CreateGraph(const std::string& case_name, const std::vector<int64_t>& shape_value, ge::Graph& graph,
                std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graph_inputs,
                std::vector<ge::Operator>& graph_outputs, bool use_dynamic_shape_input)
{
    auto reshape = op::Reshape((case_name + "_reshape").c_str());
    auto data = op::Data((case_name + "_x").c_str()).set_attr_index(0);

    const std::vector<int64_t> x_shape = {2, 3};
    const std::vector<int64_t> graph_x_shape = use_dynamic_shape_input ? std::vector<int64_t>({-1, -1}) : x_shape;
    ge::TensorDesc x_desc = MakeTensorDesc(graph_x_shape, ge::DT_FLOAT);
    data.update_input_desc_x(x_desc);
    data.update_output_desc_y(x_desc);
    inputs.push_back(MakeFloatTensor(x_shape));
    graph.AddOp(data);
    reshape.set_input_x(data);
    reshape.update_input_desc_x(x_desc);
    graph_inputs.push_back(data);

    const std::vector<int64_t> shape_shape = {2};
    ge::TensorDesc shape_desc = MakeTensorDesc(shape_shape, ge::DT_INT64);
    if (use_dynamic_shape_input) {
        auto shape_data = op::Data((case_name + "_shape").c_str()).set_attr_index(1);
        shape_data.update_input_desc_x(shape_desc);
        shape_data.update_output_desc_y(shape_desc);
        graph.AddOp(shape_data);
        reshape.set_input_shape(shape_data);
        graph_inputs.push_back(shape_data);
        inputs.push_back(MakeInt64Tensor(shape_shape, shape_value));
    } else {
        auto shape_const = op::Const((case_name + "_shape").c_str());
        shape_const.SetAttr("value", MakeInt64Tensor(shape_shape, shape_value));
        shape_const.update_output_desc_y(shape_desc);
        graph.AddOp(shape_const);
        reshape.set_input_shape(shape_const);
    }
    reshape.update_input_desc_shape(shape_desc);

    const std::vector<int64_t> y_shape = use_dynamic_shape_input ? graph_x_shape : kExpectedOutputShape;
    ge::TensorDesc y_desc = MakeTensorDesc(y_shape, ge::DT_FLOAT);
    reshape.update_output_desc_y(y_desc);
    graph_outputs.push_back(reshape);
    return kSuccess;
}

bool VerifyOutput(const std::string& case_name, const ge::Tensor& output)
{
    const auto& output_desc = output.GetTensorDesc();
    const auto& output_shape = output_desc.GetShape();
    if (output_shape.GetDimNum() != kExpectedOutputDimNum || output_shape.GetDim(0) != kExpectedOutputShape[0] ||
        output_shape.GetDim(1) != kExpectedOutputShape[1]) {
        std::cerr << GetTime() << " - ERROR - " << case_name
                  << " unexpected output shape: " << ShapeToString(output_shape) << std::endl;
        return false;
    }

    auto* output_data = reinterpret_cast<const float*>(output.GetData());
    for (size_t idx = 0U; idx < kExpectedOutputElementNum; ++idx) {
        const float expected = static_cast<float>(idx + 1U);
        if (std::fabs(output_data[idx] - expected) > kFloatCompareTolerance) {
            std::cerr << GetTime() << " - ERROR - " << case_name << " unexpected output value at " << idx << ": "
                      << output_data[idx] << std::endl;
            return false;
        }
    }
    return true;
}

int RunCase(const std::string& case_name, const std::vector<int64_t>& shape_value, bool use_dynamic_shape_input,
            uint32_t graph_id)
{
    ge::Graph graph((case_name + "_graph").c_str());
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graph_inputs;
    std::vector<ge::Operator> graph_outputs;

    std::cout << GetTime() << " - INFO - ===== begin case: " << case_name << " =====" << std::endl;
    std::cout << GetTime() << " - INFO - " << case_name
              << " graph mode: " << (use_dynamic_shape_input ? "dynamic-shape-input" : "static-const-shape")
              << std::endl;
    std::cout << GetTime() << " - INFO - " << case_name << " case input: x=[2,3], shape=" << VectorToString(shape_value)
              << std::endl;

    if (CreateGraph(case_name, shape_value, graph, inputs, graph_inputs, graph_outputs, use_dynamic_shape_input) !=
        kSuccess) {
        std::cerr << GetTime() << " - ERROR - CreateGraph failed for " << case_name << std::endl;
        std::cerr << GetTime() << " - ERROR - ===== case failed: " << case_name << " =====" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);

    std::map<ge::AscendString, ge::AscendString> build_options = {};
    ge::Session session(build_options);
    std::map<ge::AscendString, ge::AscendString> graph_options = {};
    std::cout << GetTime() << " - INFO - AddGraph for " << case_name << std::endl;
    if (session.AddGraph(graph_id, graph, graph_options) != ge::GRAPH_SUCCESS) {
        std::cerr << GetTime() << " - ERROR - AddGraph failed for " << case_name << std::endl;
        std::cerr << GetTime() << " - ERROR - ===== case failed: " << case_name << " =====" << std::endl;
        return kFailed;
    }

    std::cout << GetTime() << " - INFO - RunGraph for " << case_name << std::endl;
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graph_id, inputs, outputs) != ge::GRAPH_SUCCESS) {
        std::cerr << GetTime() << " - ERROR - RunGraph failed for " << case_name << std::endl;
        std::cerr << GetTime() << " - ERROR - ===== case failed: " << case_name << " =====" << std::endl;
        return kFailed;
    }

    if (outputs.size() != 1U) {
        std::cerr << GetTime() << " - ERROR - no output returned for " << case_name << std::endl;
        std::cerr << GetTime() << " - ERROR - ===== case failed: " << case_name << " =====" << std::endl;
        return kFailed;
    }

    if (!VerifyOutput(case_name, outputs[0])) {
        std::cerr << GetTime() << " - ERROR - ===== case failed: " << case_name << " =====" << std::endl;
        return kFailed;
    }

    const auto& output_desc = outputs[0].GetTensorDesc();
    std::cout << GetTime() << " - INFO - " << case_name << " case output: y=" << ShapeToString(output_desc.GetShape())
              << std::endl;
    std::cout << GetTime() << " - INFO - ===== case passed: " << case_name << " =====" << std::endl;

    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - initialize ge" << std::endl;
    std::map<ge::AscendString, ge::AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != ge::GRAPH_SUCCESS) {
        std::cerr << GetTime() << " - ERROR - GEInitialize failed" << std::endl;
        return kFailed;
    }

    if (RunCase("reshape_dynamic", {3, 2}, true, 0U) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }

    if (RunCase("reshape_static", {3, 2}, false, 1U) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }

    ge::GEFinalize();
    return kSuccess;
}
