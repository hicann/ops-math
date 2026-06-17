/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "math/reduce_max/op_graph/reduce_max_proto.h"

using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status GetInputTensorDimNum(const Operator& data_op, int64_t& dim_num)
{
    ge::TensorDesc input_desc = data_op.GetInputDesc(0);
    auto shape = input_desc.GetShape();
    if (shape.GetDimNum() <= 0) {
        OP_LOGE("GetInputTensorDimNum", "Get input shape is invalid.");
        return FAILED;
    }

    dim_num = shape.GetDimNum();
    OP_LOGI(GetOpName(data_op).c_str(), "GetInputTensorDimNum is: %ld", dim_num);
    return SUCCESS;
}

static Status ParseParamsReduceMax(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int> axes = {};
    bool keep_dims = true;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                axes.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        }
        if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            OP_LOGW(GetOpName(op_dest).c_str(), "Only support noop_with_empty_axes=0, but 1 is obtained now");
        }
    }
    int num = axes.size();
    std::vector<int64_t> dims = {};
    if (num != 0) {
        dims.push_back(num);
    } else {
        dims.push_back(0);
    }
    ge::Tensor tensor = Vec2Tensor(axes, dims, ge::DT_INT32, ge::FORMAT_NCHW);

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("axes", tensor);
    op_dest.SetAttr("keep_dims", keep_dims);
    const int input = 2;
    const int output = 1;
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceMax");

    return SUCCESS;
}

static Status ParseOpToGraphReduceMax(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

    ge::Tensor axes;
    if (op.GetAttr("axes", axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get axes from op failed");
        return FAILED;
    }

    if (axes.GetSize() == 0) {
        int64_t input_dim_num = 0;
        if (GetInputTensorDimNum(op, input_dim_num) != SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "Failed to get input tensor dimensions");
            return FAILED;
        }
        std::vector<int64_t> v_axes;
        for (int64_t i = 0; i < input_dim_num; ++i) {
            v_axes.push_back(i);
        }
        int num = v_axes.size();
        std::vector<int64_t> dims = {};
        if (num != 0) {
            dims.push_back(num);
        }
        axes = Vec2Tensor(v_axes, dims, ge::DT_INT64);
    }

    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(axes);
    auto reducemax = op::ReduceMax((ori_name + "_ReduceMax").c_str()).set_input_x(data0).set_input_axes(data1);

    bool keep_dims = false;
    if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }
    reducemax.set_attr_keep_dims(keep_dims);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(reducemax, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

static Status ParseParamsReduceMax13(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("ReduceMax13", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::13::ReduceMax");

    int input_size = node->input_size();
    bool keep_dims = true;
    int noop_with_empty_axes = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        } else if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT) {
            noop_with_empty_axes = attr.i();
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("input_size", input_size);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.SetAttr("noop_with_empty_axes", noop_with_empty_axes);
    return SUCCESS;
}

namespace {
struct ReduceMax13Prop {
    std::string ori_name;
    bool keep_dims = false;
    int input_num = 1;
    int empty_axes = 0;
};

Status GetProperty(const Operator& op, ReduceMax13Prop& prop)
{
    if (op.GetAttr("name", prop.ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    if (op.GetAttr("keep_dims", prop.keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }

    if (op.GetAttr("input_size", prop.input_num) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get input_num from op failed");
        return FAILED;
    }

    if (op.GetAttr("noop_with_empty_axes", prop.empty_axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attribute noop_with_empty_axes failed");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

static Status ParseOpToGraphReduceMax13(const Operator& op, Graph& graph)
{
    ReduceMax13Prop prop;
    if (GetProperty(op, prop) != SUCCESS) {
        return FAILED;
    }
    auto data0 = op::Data((prop.ori_name + "_data0").c_str()).set_attr_index(0);
    int num_input = 2;
    if (prop.input_num == 1 && prop.empty_axes == 0) {
        int64_t input_dim_num = 0;
        if (GetInputTensorDimNum(op, input_dim_num) != SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "Failed to get input tensor dimensions");
            return FAILED;
        }

        std::vector<int64_t> v_axes;
        for (int64_t i = 0; i < input_dim_num; ++i) {
            v_axes.push_back(i);
        }
        ge::TensorDesc tensorDesc;
        std::vector<int64_t> dims = {input_dim_num};
        ge::Shape shape(dims);
        tensorDesc.SetShape(shape);
        tensorDesc.SetDataType(DT_INT64);
        ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_axes.data()), v_axes.size() * sizeof(int64_t));
        auto axes = op::Const((prop.ori_name + "_axes").c_str()).set_attr_value(tensor);
        std::vector<Operator> inputs{data0, axes};
        std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
        auto reducemax = op::ReduceMax((prop.ori_name + "_ReduceMax").c_str())
                             .set_input_x(data0)
                             .set_input_axes(axes)
                             .set_attr_keep_dims(prop.keep_dims)
                             .set_attr_noop_with_empty_axes(prop.empty_axes);
        output_indexs.emplace_back(reducemax, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else if (prop.input_num == num_input) {
        auto data1 = op::Data((prop.ori_name + "_data1").c_str()).set_attr_index(1);
        auto reducemax13 = op::ReduceMax((prop.ori_name + "_ReduceMax").c_str())
                               .set_input_x(data0)
                               .set_input_axes(data1)
                               .set_attr_keep_dims(prop.keep_dims)
                               .set_attr_noop_with_empty_axes(prop.empty_axes);
        std::vector<Operator> inputs{data0, data1};
        std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
        output_indexs.emplace_back(reducemax13, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else {
        OP_LOGE(GetOpName(op).c_str(), "Input num or set attr is error");
        return FAILED;
    }
    return SUCCESS;
}

// register ReduceMax op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::ReduceMax"), ge::AscendString("ai.onnx::9::ReduceMax"),
         ge::AscendString("ai.onnx::10::ReduceMax"), ge::AscendString("ai.onnx::11::ReduceMax"),
         ge::AscendString("ai.onnx::12::ReduceMax")})
    .ParseParamsFn(ParseParamsReduceMax)
    .ParseOpToGraphFn(ParseOpToGraphReduceMax)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("ReduceMax")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::13::ReduceMax"), ge::AscendString("ai.onnx::14::ReduceMax"),
         ge::AscendString("ai.onnx::15::ReduceMax"), ge::AscendString("ai.onnx::16::ReduceMax"),
         ge::AscendString("ai.onnx::17::ReduceMax"), ge::AscendString("ai.onnx::18::ReduceMax")})
    .ParseParamsFn(ParseParamsReduceMax13)
    .ParseOpToGraphFn(ParseOpToGraphReduceMax13)
    .ImplyType(ImplyType::TVM);
} // namespace domi
