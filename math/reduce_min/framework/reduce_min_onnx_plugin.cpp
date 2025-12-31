/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "math/reduce_min/op_graph/reduce_min_proto.h"

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsReduceMin(const Message* op_src, ge::Operator& op_dest)
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
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceMin");

    return SUCCESS;
}

static Status ParseOpToGraphReduceMin(const ge::Operator& op, Graph& graph)
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
    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(axes);
    auto reducemin = op::ReduceMin((ori_name + "_ReduceMin").c_str()).set_input_x(data0).set_input_axes(data1);

    bool keep_dims = false;
    if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }
    reducemin.set_attr_keep_dims(keep_dims);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(reducemin, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

// register ReduceMin op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ReduceMin"),
                   ge::AscendString("ai.onnx::9::ReduceMin"),
                   ge::AscendString("ai.onnx::10::ReduceMin"),
                   ge::AscendString("ai.onnx::11::ReduceMin"),
                   ge::AscendString("ai.onnx::12::ReduceMin"),
                   ge::AscendString("ai.onnx::13::ReduceMin"),
                   ge::AscendString("ai.onnx::14::ReduceMin"),
                   ge::AscendString("ai.onnx::15::ReduceMin"),
                   ge::AscendString("ai.onnx::16::ReduceMin"),
                   ge::AscendString("ai.onnx::17::ReduceMin"),
                   ge::AscendString("ai.onnx::18::ReduceMin")})
    .ParseParamsFn(ParseParamsReduceMin)
    .ParseOpToGraphFn(ParseOpToGraphReduceMin)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
