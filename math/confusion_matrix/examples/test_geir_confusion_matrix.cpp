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
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/confusion_matrix_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data("placeholder" + std::to_string(inputIndex));                               \
    placeholder##inputIndex.set_attr_index(0);                                                                         \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND,      \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                               \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 1);                                                                                  \
    if (ret != SUCCESS) {                                                                                              \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                                 \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                       \
    input.push_back(tensor_placeholder##inputIndex);                                                                   \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    op1.set_input_##inputName(placeholder##inputIndex);                                                                \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    op1.update_output_desc_##outputName(outputName##outputIndex##_desc);

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    if (dt == ge::DT_FLOAT) {
        dilation = 4;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = 2;
    } else if (dt == ge::DT_BF16) {
        dilation = 2;
    } else if (dt == ge::DT_INT16) {
        dilation = 2;
    } else if (dt == ge::DT_UINT16) {
        dilation = 2;
    } else if (dt == ge::DT_INT32) {
        dilation = 4;
    } else if (dt == ge::DT_UINT32) {
        dilation = 4;
    } else if (dt == ge::DT_INT64) {
        dilation = 8;
    } else if (dt == ge::DT_UINT64) {
        dilation = 8;
    } else if (dt == ge::DT_INT8) {
        dilation = 1;
    }
    return dilation;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    size_t data_len = size * GetDataTypeSize(data_type);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }
    // 按 dtype 实际宽度逐元素写入：固定按 int32 写会在 dtype < 4 字节时堆越界，
    // dtype > 4 字节时留下未初始化数据，且 float 会被写成 1.4e-45 而非 1.0
    for (size_t i = 0; i < size; ++i) {
        switch (data_type) {
            case ge::DT_FLOAT:
                *(reinterpret_cast<float*>(pData) + i) = static_cast<float>(value);
                break;
            case ge::DT_FLOAT16: // 1.0 的 FP16 位型
                *(reinterpret_cast<uint16_t*>(pData) + i) = 0x3C00;
                break;
            case ge::DT_BF16: // 1.0 的 BF16 位型
                *(reinterpret_cast<uint16_t*>(pData) + i) = 0x3F80;
                break;
            case ge::DT_INT8:
                *(reinterpret_cast<int8_t*>(pData) + i) = static_cast<int8_t>(value);
                break;
            case ge::DT_UINT8:
                *(reinterpret_cast<uint8_t*>(pData) + i) = static_cast<uint8_t>(value);
                break;
            case ge::DT_INT16:
                *(reinterpret_cast<int16_t*>(pData) + i) = static_cast<int16_t>(value);
                break;
            case ge::DT_UINT16:
                *(reinterpret_cast<uint16_t*>(pData) + i) = static_cast<uint16_t>(value);
                break;
            case ge::DT_INT32:
                *(reinterpret_cast<int32_t*>(pData) + i) = value;
                break;
            case ge::DT_UINT32:
                *(reinterpret_cast<uint32_t*>(pData) + i) = static_cast<uint32_t>(value);
                break;
            case ge::DT_INT64:
                *(reinterpret_cast<int64_t*>(pData) + i) = static_cast<int64_t>(value);
                break;
            case ge::DT_UINT64:
                *(reinterpret_cast<uint64_t*>(pData) + i) = static_cast<uint64_t>(value);
                break;
            default:
                printf("%s - ERROR - [XIR]: GenOnesData unsupported dtype %d\n", GetTime().c_str(),
                       static_cast<int32_t>(data_type));
                return FAILED;
        }
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // ConfusionMatrix: 3 inputs (labels, predictions, weights), 1 output
    auto op1 = op::ConfusionMatrix("confusion_matrix1");

    // labels: shape [8], dtype int32
    std::vector<int64_t> labelsShape = {8};
    ADD_INPUT(1, labels, ge::DT_INT32, labelsShape);

    // predictions: shape [8], dtype int32
    std::vector<int64_t> predictionsShape = {8};
    ADD_INPUT(2, predictions, ge::DT_INT32, predictionsShape);

    // weights: shape [8], dtype float32
    std::vector<int64_t> weightsShape = {8};
    ADD_INPUT(3, weights, ge::DT_FLOAT, weightsShape);

    // output: shape [num_classes, num_classes], dtype float32
    std::vector<int64_t> outputShape = {4, 4};
    ADD_OUTPUT(1, y, ge::DT_FLOAT, outputShape);

    // attrs: num_classes (required int), dtype (required string)
    op1.set_attr_num_classes(4);
    op1.set_attr_dtype("float32");

    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
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
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
    }

    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
