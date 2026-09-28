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
#include <stdlib.h>
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
#include "../op_graph/const_place_holder_proto.h"

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

// ConstPlaceHolder 的 addr 属性要求合法的设备内存地址（GE 强制校验 placement==1 且 addr 非空），
// 通过弱符号调用 rtMalloc/rtFree（libruntime.so 由 libge_runner 传递加载）分配设备内存
extern "C" {
int32_t rtMalloc(void** devPtr, uint64_t size, uint32_t type, const uint16_t moduleId) __attribute__((weak));
int32_t rtFree(void* devPtr) __attribute__((weak));
}

static void* MallocDeviceBuffer(int64_t bytes)
{
    void* dev_addr = nullptr;
    if (rtMalloc == nullptr || rtMalloc(&dev_addr, static_cast<uint64_t>(bytes), 0U, 1U) != 0) {
        printf("%s - ERROR - [XIR]: rtMalloc failed\n", GetTime().c_str());
        return nullptr;
    }
    return dev_addr;
}

static void FreeDeviceBuffer(void* dev_addr)
{
    if (dev_addr != nullptr && rtFree != nullptr) {
        (void)rtFree(dev_addr);
    }
}

ge::Graph BuildGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& shape_attr, void* dev_addr)
{
    ge::Graph graph("test_const_place_holder_ir");
    ge::DataType inDtype = ge::DT_FLOAT;

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

    auto const_place_holder_node = op::ConstPlaceHolder("const_place_holder_node");
    const_place_holder_node.set_attr_origin_shape(shape_attr);
    const_place_holder_node.set_attr_origin_format(0);
    const_place_holder_node.set_attr_storage_shape(shape_attr);
    const_place_holder_node.set_attr_storage_format(0);
    const_place_holder_node.set_attr_expand_dim_rules("");
    const_place_holder_node.set_attr_dtype(ge::DT_FLOAT);
    const_place_holder_node.set_attr_addr(reinterpret_cast<int64_t>(dev_addr));
    const_place_holder_node.set_attr_size(elem_num * static_cast<int64_t>(sizeof(float)));
    const_place_holder_node.set_attr_placement(1);
    ge::TensorDesc output_desc(ge::Shape(shape_attr), ge::FORMAT_ND, ge::DT_FLOAT);
    const_place_holder_node.update_output_desc_y(output_desc);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(data_node);
    add_node.set_input_x2(const_place_holder_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(const_place_holder_node);
    graph.AddOp(add_node);
    return graph;
}

// 校验 RunGraph 后用户侧 desc（已实证：ConstPlaceHolder 的输出 desc 会被编译期推导回写，
// 判别实验中设置错误初始 desc {7,7}+DT_INT32 后，RunGraph 成功时读回的是属性推导出的正确值）
bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expected)
{
    ge::Operator const_place_holder_op;
    graph.FindOpByName("const_place_holder_node", const_place_holder_op);
    auto output_desc = const_place_holder_op.GetOutputDesc(0);
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

bool RunScenario(const string& tag, const ExpectedResult& expected, RunContext& ctx, const string& om_name)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    void* dev_addr = MallocDeviceBuffer(expected.size * static_cast<int64_t>(sizeof(float)));
    if (dev_addr == nullptr) {
        return false;
    }

    vector<ge::Tensor> input;
    ge::Graph graph = BuildGraph(input, expected.shape, dev_addr);

    ge::Session* session = new ge::Session(ctx.session_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        FreeDeviceBuffer(dev_addr);
        return false;
    }
    ge::Status ret = session->AddGraph(ctx.graph_id, graph, ctx.session_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        FreeDeviceBuffer(dev_addr);
        return false;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(ctx.graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        FreeDeviceBuffer(dev_addr);
        return false;
    }
    printf("%s - INFO - [XIR]: RunGraph success, output count=%zu\n", GetTime().c_str(), output.size());
    for (size_t i = 0; i < output.size(); i++) {
        auto out_shape = output[i].GetTensorDesc().GetShape();
        printf("%s - INFO - [XIR]: output[%zu] dim_num=%zu", GetTime().c_str(), i, out_shape.GetDimNum());
        for (size_t d = 0; d < out_shape.GetDimNum(); d++) {
            printf(" dim%zu=%ld", d, out_shape.GetDim(d));
        }
        printf(" shape_size=%ld dtype=%d\n", out_shape.GetShapeSize(), output[i].GetTensorDesc().GetDataType());
    }

    if (!CheckInferShape(graph, expected.shape)) {
        delete session;
        FreeDeviceBuffer(dev_addr);
        return false;
    }

    if (output.empty() || output[0].GetTensorDesc().GetShape().GetShapeSize() != expected.size ||
        output[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: expected shape_size=%ld dtype=%d\n", GetTime().c_str(),
               expected.size, ge::DT_FLOAT);
        delete session;
        FreeDeviceBuffer(dev_addr);
        return false;
    }
    delete session;
    FreeDeviceBuffer(dev_addr);

    void* dev_addr_om = MallocDeviceBuffer(expected.size * static_cast<int64_t>(sizeof(float)));
    if (dev_addr_om == nullptr) {
        return false;
    }
    vector<ge::Tensor> input_om;
    ge::Graph graph_om = BuildGraph(input_om, expected.shape, dev_addr_om);
    ge::ModelBufferData model_buffer;
    ret = aclgrphBuildModel(graph_om, ctx.session_options, model_buffer);
    FreeDeviceBuffer(dev_addr_om);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: BuildModel failed\n", GetTime().c_str());
        return false;
    }
    aclgrphSaveModel(om_name.c_str(), model_buffer);
    printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), om_name.c_str());
    printf("%s - INFO - [XIR]: %s PASSED\n", GetTime().c_str(), tag.c_str());
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

    if (!RunScenario("shape(2,3)", shape_2_3, ctx, "./const_place_holder_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(4,5)", shape_4_5, ctx, "./const_place_holder_shape_4_5_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("shape(1,2,3)", shape_1_2_3, ctx, "./const_place_holder_shape_1_2_3_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
