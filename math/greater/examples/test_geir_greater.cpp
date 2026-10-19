/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/greater_proto.h"

#define FAILED -1
#define SUCCESS 0
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}
using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape, value)                                          \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesDataFloat32(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                     \
                             placeholder##intputIndex##_desc, value);                                               \
    if (ret != SUCCESS) {                                                                                           \
        LOG_PRINT("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                           \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    greater.set_input_##intputName(placeholder##intputIndex);                                                       \
    inputs.push_back(placeholder##intputIndex)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                                           \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        LOG_PRINT("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                           \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    greater.set_input_##intputName(placeholder##intputIndex);                                                       \
    greater.update_input_desc_##intputName(placeholder##intputIndex##_desc);                                        \
    inputs.push_back(placeholder##intputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    greater.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
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

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc);
    graphStatus ret = input_tensor.SetData(reinterpret_cast<uint8_t*>(pData), data_len,
                                           [](uint8_t* ptr) { delete[] reinterpret_cast<float*>(ptr); });
    if (ret != SUCCESS) {
        delete[] pData;
        return FAILED;
    }
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t* pData = new (std::nothrow) int32_t[size];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc);
    graphStatus ret = input_tensor.SetData(reinterpret_cast<uint8_t*>(pData), data_len,
                                           [](uint8_t* ptr) { delete[] reinterpret_cast<int32_t*>(ptr); });
    if (ret != SUCCESS) {
        delete[] pData;
        return FAILED;
    }
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
    // 自定义代码：添加单算子定义到图中
    auto greater = op::Greater("greater");

    std::vector<int64_t> x1Shape = {2, 2};
    std::vector<int64_t> x2Shape = {2};
    ADD_INPUT(1, x1, inDtype, x1Shape, 3);
    ADD_INPUT(2, x2, inDtype, x2Shape, 1);

    ADD_OUTPUT(1, y, DT_BOOL, x1Shape);

    outputs.push_back(greater);
    // 添加完毕
    return SUCCESS;
}

int InitializeAndSetupGraph(Graph& graph, std::vector<ge::Tensor>& input, DataType inDtype)
{
    LOG_PRINT("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    return SUCCESS;
}

int ExecuteGraph(Graph& graph, std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    std::map<AscendString, AscendString> build_options = {};
    LOG_PRINT("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    LOG_PRINT("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    Status ret = session->AddGraph(graph_id, graph, graph_options);

    LOG_PRINT("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    LOG_PRINT("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    LOG_PRINT("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());

    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    delete session;
    return SUCCESS;
}

int32_t ProcessIOData(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        LOG_PRINT("input %d dtype: %d\n", i, static_cast<int>(input[i].GetTensorDesc().GetDataType()));
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("input %d shape size: %ld\n", i, input_shape);
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        LOG_PRINT("output %d dtype: %d\n", i, static_cast<int>(output[i].GetTensorDesc().GetDataType()));
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT("output %d shape size: %ld\n", i, output_shape);
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        uint8_t* resultData = (uint8_t*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %u\n", j, resultData[j]);
            if (resultData[j] != 1) {
                LOG_PRINT("%s - ERROR - [XIR]: Precision check failed at output %d index %ld, expected 1 but got %u\n",
                          GetTime().c_str(), i, j, resultData[j]);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        LOG_PRINT("%s - ERROR - [XIR]: Missing soc version argument\n", GetTime().c_str());
        return FAILED;
    }

    // 1、创建图对象
    const char* graph_name = "tc_ge_irrun_greater_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    LOG_PRINT("argv[1] = %s\n", argv[1]);

    DataType inDtype = DT_FLOAT;

    LOG_PRINT("inDtype: %d\n", static_cast<int>(inDtype));

    // 初始化和设置图
    if (InitializeAndSetupGraph(graph, input, inDtype) != SUCCESS) {
        return FAILED;
    }

    // 执行图计算
    std::vector<ge::Tensor> output;
    if (ExecuteGraph(graph, input, output) != SUCCESS) {
        return FAILED;
    }

    // 处理输入输出数据
    if (ProcessIOData(input, output) != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Precision check failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    LOG_PRINT("Error message: %s\n", error_str.c_str());
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    LOG_PRINT("Warning message: %s\n", warning_str.c_str());
    LOG_PRINT("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    LOG_PRINT("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
