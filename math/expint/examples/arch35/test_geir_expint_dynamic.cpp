/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ge_api.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "../../op_graph/expint_proto.h"

namespace {

constexpr int32_t SUCCESS = 0;
constexpr int32_t FAILED = -1;
constexpr float ATOL = 1.0e-4f;
constexpr float RTOL = 2.0e-4f;

std::string ShapeString(const std::vector<int64_t>& shape)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << (i == 0 ? "" : ",") << shape[i];
    }
    os << "]";
    return os.str();
}

size_t ElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

// Covers all 7 intervals, special values (NaN/+/-inf/+/-0) and the
// overflow-safe large-x correction path (x > EXP_CLAMP=88, where the
// fp32 result saturates to +inf beyond ~93.2).
std::vector<float> MakeInput(const std::vector<int64_t>& shape)
{
    static const std::array<float, 14> values = {-1.0f, -0.0f, 0.0f,  0.25f, 1.0f,  2.0f,  4.0f,
                                                 8.0f,  16.0f, 32.0f, 64.0f, 88.5f, 93.0f, 100.0f};
    std::vector<float> input(ElementCount(shape));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = values[i % values.size()];
    }
    return input;
}

ge::Tensor MakeTensor(const std::vector<int64_t>& shape, std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}

ge::Graph MakeGraph(const std::string& graphName, const std::vector<int64_t>& declaredShape)
{
    ge::Graph graph(graphName.c_str());
    auto data = ge::op::Data((graphName + "_data").c_str()).set_attr_index(0);
    ge::TensorDesc inputDesc(ge::Shape(declaredShape), ge::FORMAT_ND, ge::DT_FLOAT);
    data.update_input_desc_x(inputDesc);
    data.update_output_desc_y(inputDesc);
    auto expint = ge::op::Expint((graphName + "_expint").c_str());
    expint.set_input_x(data);
    expint.update_input_desc_x(inputDesc);
    expint.update_output_desc_y(inputDesc);
    graph.AddOp(data);
    graph.SetInputs(std::vector<ge::Operator>{data}).SetOutputs(std::vector<ge::Operator>{expint});
    return graph;
}

bool ShapeEquals(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected)
{
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

float Reference(float x)
{
    if (std::isnan(x) || x < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == 0.0f) {
        return -std::numeric_limits<float>::infinity();
    }
    if (std::isinf(x)) {
        return std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(std::expint(static_cast<double>(x)));
}

bool Verify(const ge::Tensor& output, const std::vector<int64_t>& expectedShape, const std::vector<float>& input)
{
    const ge::TensorDesc& desc = output.GetTensorDesc();
    const std::vector<int64_t> actualShape = desc.GetShape().GetDims();
    if (desc.GetDataType() != ge::DT_FLOAT || !ShapeEquals(actualShape, expectedShape) ||
        ElementCount(actualShape) != input.size()) {
        std::cerr << "Output descriptor mismatch for " << ShapeString(expectedShape) << std::endl;
        return false;
    }
    const auto* actual = reinterpret_cast<const float*>(output.GetData());
    if (actual == nullptr) {
        std::cerr << "Output data is null" << std::endl;
        return false;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        const float expected = Reference(input[i]);
        const bool sameNan = std::isnan(expected) && std::isnan(actual[i]);
        const bool sameInf = std::isinf(expected) && std::isinf(actual[i]) &&
                             std::signbit(expected) == std::signbit(actual[i]);
        const float limit = ATOL + RTOL * std::fabs(expected);
        if (!sameNan && !sameInf && !(std::isfinite(expected) && std::fabs(actual[i] - expected) <= limit)) {
            std::cerr << "Value mismatch at " << i << ": actual=" << actual[i] << ", expected=" << expected
                      << std::endl;
            return false;
        }
    }
    std::cout << "Shape, dtype and values PASSED for " << ShapeString(expectedShape) << std::endl;
    return true;
}

// RunOne centralizes ge::Session::RunGraph and prints "Shape, dtype and values PASSED" only after full
// verification.
bool RunOne(ge::Session& session, uint32_t graphId, const std::vector<int64_t>& shape)
{
    std::vector<float> values = MakeInput(shape);
    ge::Tensor tensor = MakeTensor(shape, values);
    std::vector<ge::Tensor> inputs{tensor};
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graphId, inputs, outputs) != ge::SUCCESS || outputs.size() != 1U) {
        return false;
    }
    return Verify(outputs[0], shape, values);
}

// RunScenario runs one declared shape against several concrete shapes on the same graph/session.
bool RunScenario(const std::string& scenario, uint32_t graphId, const std::vector<int64_t>& declaredShape,
                 const std::vector<std::vector<int64_t>>& concreteShapes)
{
    std::cout << "Scenario " << scenario << ", declared shape " << ShapeString(declaredShape) << std::endl;
    const std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    ge::Graph graph = MakeGraph("expint_" + scenario, declaredShape);
    const std::map<ge::AscendString, ge::AscendString> graphOptions = {
        {"ge.exec.dynamicGraphExecuteMode", "dynamic_execute"}};
    if (session.AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        return false;
    }
    size_t passed = 0;
    for (const auto& shape : concreteShapes) {
        std::cout << "Run concrete shape " << ShapeString(shape) << std::endl;
        passed += RunOne(session, graphId, shape) ? 1U : 0U;
    }
    std::cout << "Scenario " << scenario << " summary: " << passed << "/" << concreteShapes.size() << " passed"
              << std::endl;
    return passed == concreteShapes.size();
}

} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"},
                                                                        {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        return FAILED;
    }
    const bool unknownDim = RunScenario("unknown_dim_minus_1", 0, {-1, -1}, {{4, 2}, {1, 8}, {3, 5}});
    const bool unknownRank = RunScenario("unknown_rank_minus_2", 1, {-2}, {{}, {8}, {4, 2}, {2, 3, 4}});
    ge::GEFinalize();
    if (!unknownDim || !unknownRank) {
        return FAILED;
    }
    std::cout << "Expint dynamic GEIR verification PASSED (-1 and -2)" << std::endl;
    return SUCCESS;
}
