/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <stdint.h>
#include <cstdio>
#include <ctime>
#include <array>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <limits>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "../../op_graph/expint_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// IEEE 754 bit layout constants for the fp32 -> fp16/bf16 test data encoding
constexpr uint32_t FP16_SIGN_MASK = 0x8000U;
constexpr uint32_t FP32_EXP_MASK = 0xFFU;
constexpr uint32_t FP16_MANT_MASK = 0x3FFU;
constexpr uint32_t FP16_INF_BITS = 0x7C00U;
constexpr int32_t FP16_MANT_BITS = 10;
constexpr int32_t FP32_SIGN_SHIFT = 16;
constexpr int32_t FP32_EXP_SHIFT = 23;
constexpr int32_t FP32_MANT_SHIFT = 13;
constexpr int32_t FP32_EXP_BIAS = 127;
constexpr int32_t FP16_EXP_BIAS = 15;
constexpr int32_t FP16_EXP_MAX = 31;
// Test fixture parameters: 14-value coverage (all intervals + special values +
// x>88 overflow-safe correction path, aligned with expint_geir_v2_common.h MakeInput)
constexpr int32_t INPUT_FILL_VALUE = 2;
constexpr std::array<float, 14> INPUT_VALUES = {-1.0f, -0.0f, 0.0f,  0.25f, 1.0f,  2.0f,  4.0f,
                                                8.0f,  16.0f, 32.0f, 64.0f, 88.5f, 93.0f, 100.0f};
constexpr int64_t EXPECTED_OUTPUT_ELEMS = 16; // 4x4 input shape
constexpr float ATOL = 1.0e-4f;
constexpr float RTOL = 2.0e-4f;

static float InputValueAt(size_t idx) { return INPUT_VALUES[idx % INPUT_VALUES.size()]; }

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data(std::string("placeholder") + std::to_string(inputIndex))                   \
                                       .set_attr_index(0);                                                             \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND,      \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                               \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, INPUT_FILL_VALUE);                                                                   \
    if (ret != SUCCESS) {                                                                                              \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                                 \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                       \
    input.push_back(tensor_placeholder##inputIndex);                                                                   \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    expintOp.set_input_##inputName(placeholder##inputIndex);                                                           \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    expintOp.update_output_desc_##outputName(outputName##outputIndex##_desc);

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int32_t GenOnesData(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, DataType dataType,
                    int32_t value)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (int64_t dim : shapes) {
        size *= static_cast<size_t>(dim);
    }
    size_t dataLen = size * static_cast<size_t>(ge::GetSizeByDataType(dataType));
    std::vector<uint8_t> data(dataLen);
    if (dataType == DT_FLOAT) {
        std::vector<float> fData(size);
        for (size_t i = 0; i < size; ++i) {
            fData[i] = InputValueAt(i);
        }
        std::copy(fData.begin(), fData.end(), reinterpret_cast<float*>(data.data()));
    } else if (dataType == DT_FLOAT16) {
        std::vector<uint16_t> hData(size);
        for (size_t i = 0; i < size; ++i) {
            const float fval = InputValueAt(i);
            const uint32_t fbits = __builtin_bit_cast(uint32_t, fval);
            const uint32_t sign = (fbits >> FP32_SIGN_SHIFT) & FP16_SIGN_MASK;
            const int32_t rawExp = static_cast<int32_t>((fbits >> FP32_EXP_SHIFT) & FP32_EXP_MASK);
            const int32_t exp = rawExp - FP32_EXP_BIAS + FP16_EXP_BIAS;
            const uint32_t mant = (fbits >> FP32_MANT_SHIFT) & FP16_MANT_MASK;
            uint16_t hval;
            if (exp <= 0) {
                hval = static_cast<uint16_t>(sign);
            } else if (exp >= FP16_EXP_MAX) {
                hval = static_cast<uint16_t>(sign | FP16_INF_BITS);
            } else {
                hval = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << FP16_MANT_BITS) | mant);
            }
            hData[i] = hval;
        }
        std::copy(hData.begin(), hData.end(), reinterpret_cast<uint16_t*>(data.data()));
    } else if (dataType == DT_BF16) {
        std::vector<uint16_t> bData(size);
        for (size_t i = 0; i < size; ++i) {
            const float fval = InputValueAt(i);
            const uint32_t fbits = __builtin_bit_cast(uint32_t, fval);
            bData[i] = static_cast<uint16_t>(fbits >> FP32_SIGN_SHIFT);
        }
        std::copy(bData.begin(), bData.end(), reinterpret_cast<uint16_t*>(data.data()));
    } else {
        const std::vector<int32_t> iData(size, value);
        std::copy(iData.begin(), iData.end(), reinterpret_cast<int32_t*>(data.data()));
    }
    inputTensor = Tensor(inputTensorDesc, data.data(), dataLen);
    return SUCCESS;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, const uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    if (written != dataSize) {
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto expintOp = op::Expint("expint1");
    std::vector<int64_t> xShape = {4, 4};
    ADD_INPUT(1, x, inDtype, xShape);

    ADD_OUTPUT(1, y, inDtype, xShape);

    outputs.push_back(expintOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graphName = "tc_ge_irrun_test_expint";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> buildOptions = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    std::unique_ptr<ge::Session> session(new (std::nothrow) ge::Session(buildOptions));

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph to session failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int outputNum = output.size();
    bool verified = outputNum == 1;
    for (int i = 0; i < outputNum; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string outputFile = "./tc_ge_irrun_test_expint_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t* outputDataI = output[i].GetData();
        int64_t outputShape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << outputShape << std::endl;
        uint32_t dataSize = outputShape *
                            static_cast<uint32_t>(ge::GetSizeByDataType(output[i].GetTensorDesc().GetDataType()));
        WriteDataToFile(outputFile.c_str(), dataSize, outputDataI);
        verified = verified && output[i].GetTensorDesc().GetDataType() == DT_FLOAT &&
                   outputShape == EXPECTED_OUTPUT_ELEMS;
        std::vector<float> actual(static_cast<size_t>(outputShape));
        std::copy(reinterpret_cast<const float*>(outputDataI),
                  reinterpret_cast<const float*>(outputDataI) + actual.size(), actual.begin());
        for (int64_t j = 0; verified && j < outputShape; ++j) {
            const float inVal = InputValueAt(static_cast<size_t>(j));
            // kernel 边界语义: x<0 -> NaN, x=0 -> -inf, x>~93.2 -> +inf（IEEE 溢出）
            float expected;
            if (inVal < 0.0f) {
                expected = std::numeric_limits<float>::quiet_NaN();
            } else if (inVal == 0.0f) {
                expected = -std::numeric_limits<float>::infinity();
            } else if (inVal > 93.2f) {
                expected = std::numeric_limits<float>::infinity();
            } else {
                expected = static_cast<float>(std::expint(static_cast<double>(inVal)));
            }
            if (std::isnan(expected)) {
                verified = std::isnan(actual[static_cast<size_t>(j)]);
            } else if (std::isinf(expected)) {
                verified = std::isinf(actual[static_cast<size_t>(j)]) &&
                           std::signbit(actual[static_cast<size_t>(j)]) == std::signbit(expected);
            } else {
                verified = std::abs(actual[static_cast<size_t>(j)] - expected) <= ATOL + RTOL * std::abs(expected);
            }
        }
    }

    if (!verified) {
        printf("%s - ERROR - [XIR]: Output shape, dtype or value verification failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }
    std::cout << "Shape, dtype and values PASSED for [4,4]" << std::endl;
    std::cout << "Expint static GEIR verification PASSED" << std::endl;

    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    session.reset();
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
