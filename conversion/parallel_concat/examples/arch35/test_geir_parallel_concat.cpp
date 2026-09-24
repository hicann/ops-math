/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../../op_graph/parallel_concat_proto.h"

#define FAILED -1
#define SUCCESS 0

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_DYNAMIC_INPUT(inputIndex, inputDtype, inputShape, dataName)                                         \
    do {                                                                                                        \
        vector<int64_t> shape##dataName = inputShape;                                                           \
        auto placeholder##dataName = op::Data("placeholder" #dataName).set_attr_index(0);                       \
        TensorDesc desc##dataName(ge::Shape(shape##dataName), FORMAT_ND, inputDtype);                           \
        desc##dataName.SetFormat(FORMAT_ND);                                                                    \
        placeholder##dataName.update_input_desc_x(desc##dataName);                                              \
        placeholder##dataName.update_output_desc_y(desc##dataName);                                             \
        Tensor tensor##dataName;                                                                                \
        ret = GenOnesData(shape##dataName, tensor##dataName, desc##dataName, inputDtype, (inputIndex + 1));     \
        if (ret != SUCCESS) {                                                                                   \
            LOG_PRINT("%s - ERROR - [PARALLEL_CONCAT_GE_IR]: Generate input data failed\n", GetTime().c_str()); \
            return FAILED;                                                                                      \
        }                                                                                                       \
        parallelConcat1.UpdateDynamicInputDesc("values", inputIndex, desc##dataName);                           \
        parallelConcat1.set_dynamic_input_values(inputIndex, placeholder##dataName);                            \
        input.push_back(tensor##dataName);                                                                      \
        graph.AddOp(placeholder##dataName);                                                                     \
        inputs.push_back(placeholder##dataName);                                                                \
    } while (0)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                           \
    do {                                                                                                        \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        parallelConcat1.update_output_desc_##outputName(outputName##outputIndex##_desc);                        \
    } while (0)

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
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenOnesData(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, DataType dataType,
                    int64_t value)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    // RAII host buffer; the Tensor(const uint8_t*, size) constructor copies
    // the bytes, so the vector may die at scope exit.
    vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(value);
    }
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<const uint8_t*>(data.data()),
                         size * GetDataTypeSize(dataType));
    return SUCCESS;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, const uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t written = fwrite(inputData, 1, dataSize, fp);
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
    // ParallelConcat: 动态输入values（N个首维为1的同形同dtype张量）+ 属性shape/N，输出output_data
    auto parallelConcat1 = op::ParallelConcat("parallel_concat1").create_dynamic_input_values(2, false);

    // values: 2个shape为[1, 8]的FLOAT输入（首维必须为1）
    std::vector<int64_t> valuesShape = {1, 8};
    ADD_DYNAMIC_INPUT(0, inDtype, valuesShape, A);
    ADD_DYNAMIC_INPUT(1, inDtype, valuesShape, B);

    // 属性shape: 显式输出shape [N, 8]；属性N: 动态输入个数
    parallelConcat1.set_attr_shape(std::vector<int64_t>{2, 8});
    parallelConcat1.set_attr_N(2);

    // output_data: [N, 8]，dtype与values一致
    std::vector<int64_t> outShape = {2, 8};
    ADD_OUTPUT(1, output_data, inDtype, outShape);

    outputs.push_back(parallelConcat1);

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graphName = "tc_ge_irrun_test";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;

    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Start to initialize ge using ge global options\n",
              GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [PARALLEL_CONCAT_GE_IR]: Initialize ge using ge global options failed\n",
                  GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Initialize ge using ge global options success\n",
              GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        LOG_PRINT("argv[1] = %s\n", argv[1]);
    }

    DataType inDtype = DT_FLOAT;

    LOG_PRINT("inDtype: %d\n", static_cast<int>(inDtype));

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [PARALLEL_CONCAT_GE_IR]: Create ir session using build options failed\n",
                  GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> buildOptions = {};
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Start to create ir session using build options\n",
              GetTime().c_str());
    // Stack-allocated session: destructor-based cleanup on every exit path.
    ge::Session session(buildOptions);

    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Create ir session using build options success\n",
              GetTime().c_str());
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session.AddGraph(graphId, graph, graphOptions);

    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Session add ir compute graph to ir session success\n",
              GetTime().c_str());
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: dump graph to txt\n", GetTime().c_str());
    std::string filePath = "./dump";
    aclgrphDumpGraph(graph, filePath.c_str(), filePath.length());
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session.RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [PARALLEL_CONCAT_GE_IR]: Run graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Session run ir compute graph success\n", GetTime().c_str());

    int inputNum = input.size();
    for (int i = 0; i < inputNum; i++) {
        LOG_PRINT("input %d dtype: %d\n", i, static_cast<int>(input[i].GetTensorDesc().GetDataType()));
        string inputFile = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        const uint8_t* inputDataI = input[i].GetData();
        int64_t inputShape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("input %d shape size: %ld\n", i, inputShape);
        uint32_t dataSize = inputShape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(inputFile, dataSize, inputDataI);
    }

    int outputNum = output.size();
    for (int i = 0; i < outputNum; i++) {
        LOG_PRINT("output %d dtype: %d\n", i, static_cast<int>(output[i].GetTensorDesc().GetDataType()));
        string outputFile = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t* outputDataI = output[i].GetData();
        int64_t outputShape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("output %d shape size: %ld\n", i, outputShape);
        uint32_t dataSize = outputShape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(outputFile, dataSize, outputDataI);
    }

    ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
    std::string errorStr(errorMsg.GetString());
    LOG_PRINT("Error message: %s\n", errorStr.c_str());
    ge::AscendString warningMsg = ge::GEGetWarningMsgV2();
    std::string warningStr(warningMsg.GetString());
    LOG_PRINT("Warning message: %s\n", warningStr.c_str());
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Precision is ok\n", GetTime().c_str());
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [PARALLEL_CONCAT_GE_IR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [PARALLEL_CONCAT_GE_IR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
