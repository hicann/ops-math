/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Static GEIR example for Spence (Ascend950, arch35).
 *
 * Deterministic input: shape [2,4], every element = 2.0f. Golden:
 * spence(2.0) = Li2(1-2) = Li2(-1) = -pi^2/12 = -0.82246703342411321824.
 * The output shape, dtype and element values are verified against the golden.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"
#include "ge_api.h"

#include "../op_graph/spence_proto.h"

namespace {

constexpr double kSpenceTwo = -0.82246703342411321824; // spence(2.0) = -pi^2/12
constexpr double kTolerance = 1.0e-5;

std::vector<int64_t> kXShape = {2, 4};

bool MakeFilledTensor(const std::vector<int64_t>& shape, float value, ge::Tensor& tensor)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    size_t numel = 1;
    for (int64_t dim : shape) {
        numel *= static_cast<size_t>(dim);
    }
    float* data = new (std::nothrow) float[numel];
    if (data == nullptr) {
        return false;
    }
    for (size_t index = 0; index < numel; ++index) {
        data[index] = value;
    }
    tensor = ge::Tensor(desc, reinterpret_cast<uint8_t*>(data), numel * sizeof(float));
    return true;
}

void WriteDataToFile(const std::string& binFile, uint64_t dataSize, const uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "w");
    if (fp == nullptr) {
        return;
    }
    fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
}

} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::printf("Spence static GEIR initialization failed\n");
        return -1;
    }

    ge::Graph graph("spence_static_ge");
    auto x = ge::op::Data("x_static").set_attr_index(0);
    ge::TensorDesc xDesc(ge::Shape(kXShape), ge::FORMAT_ND, ge::DT_FLOAT);
    xDesc.SetPlacement(ge::kPlacementHost);
    x.update_input_desc_x(xDesc);

    auto spence = ge::op::Spence("spence_static");
    spence.set_input_x(x);
    ge::TensorDesc yDesc(ge::Shape(kXShape), ge::FORMAT_ND, ge::DT_FLOAT);
    spence.update_output_desc_y(yDesc);

    graph.AddOp(x);
    graph.SetInputs({x}).SetOutputs({spence});

    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph, sessionOptions) != ge::SUCCESS) {
        ge::GEFinalize();
        return -1;
    }

    ge::Tensor input;
    if (!MakeFilledTensor(kXShape, 2.0F, input)) {
        ge::GEFinalize();
        return -1;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, {input}, outputs) != ge::SUCCESS) {
        ge::GEFinalize();
        return -1;
    }

    // Verify output count, shape, dtype and element values.
    if (outputs.size() != 1U) {
        ge::GEFinalize();
        return -1;
    }
    const ge::TensorDesc& outDesc = outputs[0].GetTensorDesc();
    if (outDesc.GetDataType() != ge::DT_FLOAT || outDesc.GetShape().GetDimNum() != kXShape.size() ||
        outDesc.GetShape().GetDim(0) != kXShape[0] || outDesc.GetShape().GetDim(1) != kXShape[1]) {
        ge::GEFinalize();
        return -1;
    }
    const size_t numel = static_cast<size_t>(kXShape[0]) * static_cast<size_t>(kXShape[1]);
    const float* outData = reinterpret_cast<const float*>(outputs[0].GetData());
    if (numel != 0U && outData == nullptr) {
        ge::GEFinalize();
        return -1;
    }
    for (size_t index = 0; index < numel; ++index) {
        if (std::fabs(static_cast<double>(outData[index]) - kSpenceTwo) > kTolerance) {
            ge::GEFinalize();
            return -1;
        }
    }

    WriteDataToFile("./tc_ge_irrun_test_npu_input_0.bin", numel * sizeof(float), input.GetData());
    WriteDataToFile("./tc_ge_irrun_test_npu_output_0.bin", numel * sizeof(float), outputs[0].GetData());

    if (ge::GEFinalize() != ge::SUCCESS) {
        return -1;
    }

    std::printf("Shape, dtype and values PASSED for [2,4]\n");
    std::printf("Spence static GEIR verification PASSED\n");
    return 0;
}
