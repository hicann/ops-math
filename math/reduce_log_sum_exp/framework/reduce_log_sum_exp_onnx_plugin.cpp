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
#include "math/reduce_log_sum_exp/op_graph/reduce_log_sum_exp_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
static Status ParseParamsReduceLogSumExp(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int op_input_size = node->input_size();
    int op_output_size = node->output_size();
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceLogSumExp");

    std::vector<int> v_axes = {};
    bool keep_dims = true;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                v_axes.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        } else if (
            attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            OP_LOGW(GetOpName(op_dest).c_str(), "Only support noop_with_empty_axes=0, but 1 is obtained now");
        }
    }

    int64_t len = v_axes.size();
    std::vector<int64_t> dims = {};
    if (len != 0) {
        dims.push_back(len);
    }
    ge::Tensor tensor1 = Vec2Tensor(v_axes, dims, DT_INT32, ge::FORMAT_ND);
    op_dest.SetAttr("axes", tensor1);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphReduceLogSumExp(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    bool flag = false;
    ge::Tensor const_value;
    if (op.GetAttr("axes", const_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get axes from op failed");
        return FAILED;
    }
    if (op.GetAttr("keep_dims", flag) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get value of keep_dims from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto const_op = op::Const((ori_name + "_const_data").c_str()).set_attr_value(const_value);
    auto reduce_log_sum_exp = op::ReduceLogSumExp((ori_name + "_ReduceLogSumExp").c_str())
                                  .set_input_x(data0)
                                  .set_input_axes(const_op)
                                  .set_attr_keep_dims(flag);
    std::vector<Operator> inputs{data0, const_op};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    output_indexs.emplace_back(reduce_log_sum_exp, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register ReduceLogSumExp op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::9::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::10::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::11::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::12::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::13::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::14::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::15::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::16::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::17::ReduceLogSumExp"),
                   ge::AscendString("ai.onnx::18::ReduceLogSumExp")})
    .ParseParamsFn(ParseParamsReduceLogSumExp)
    .ParseOpToGraphFn(ParseOpToGraphReduceLogSumExp)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
