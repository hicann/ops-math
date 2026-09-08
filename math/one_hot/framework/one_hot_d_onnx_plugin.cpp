/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator.h"
#include "graph/graph.h"
#include "stub_ops.h"
#include "math_onnx_plugin_util.h"
#include "log/log.h"
#include "register/register.h"
#include "nlohmann/json.hpp"
#include "op_math_proto_extend.h"
#include "math/cast/op_graph/cast_proto.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuOneHot(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // 1.add dynamic input and out
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);

    // 2.set original_type
    op_dest.SetAttr("original_type", "npu::1::NPUOneHot");

    // 3.set attr if needed
    bool has_depth = false;
    int depth = 1;
    int num_classes = -1;
    int on_value = 1;
    int off_value = 0;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            const json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (const json& attr : attrs["attribute"]) {
                    if (!attr.contains("i")) {
                        continue;
                    }
                    const std::string name = attr.value("name", "");
                    if (name == "num_classes") {
                        num_classes = attr["i"].get<int>();
                    } else if (name == "depth") {
                        depth = attr["i"].get<int>();
                        has_depth = true;
                    } else if (name == "on_value") {
                        on_value = attr["i"].get<int>();
                    } else if (name == "off_value") {
                        off_value = attr["i"].get<int>();
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    if (!has_depth) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr depth");
        return FAILED;
    }

    // The ONNX node name is no longer reachable through the protobuf node type; the parser
    // now exposes it as the source operator's own name. Fall back to the dest op name when
    // the source name is unavailable.
    const std::string op_name = GetOpName(op_dest);
    ge::AscendString source_name_string;
    const std::string source_name = op_src.GetName(source_name_string) == ge::GRAPH_SUCCESS ?
                                        source_name_string.GetString() :
                                        std::string();
    const std::string node_name = source_name.empty() ? op_name : source_name;

    ge::Tensor scalar_on_value = CreateScalar(on_value, ge::DT_INT32);
    ge::Tensor scalar_off_value = CreateScalar(off_value, ge::DT_INT32);
    op_dest.SetAttr("name", node_name);
    op_dest.SetAttr("num_classes", num_classes);
    op_dest.SetAttr("depth", depth);
    op_dest.SetAttr("on_value", scalar_on_value);
    op_dest.SetAttr("off_value", scalar_off_value);
    return SUCCESS;
}

static Status ParseOpToGraphOneHot(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    int axis = -1;
    op.GetAttr("num_classes", axis);
    int depth = 0;
    op.GetAttr("depth", depth);
    ge::Tensor on_value;
    op.GetAttr("on_value", on_value);
    ge::Tensor off_value;
    op.GetAttr("off_value", off_value);

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto const_on_value = ge::op::Const((ori_name + "data_on_value").c_str()).set_attr_value(on_value);
    auto const_off_value = ge::op::Const((ori_name + "data_off_value").c_str()).set_attr_value(off_value);

    auto cast_data0 = ge::op::Cast((ori_name + "_Cast_data0").c_str())
                          .set_input_x(data0)
                          .set_attr_dst_type(ge::DT_INT32);
    auto cast_on_value = ge::op::Cast((ori_name + "_Cast_on_value").c_str())
                             .set_input_x(const_on_value)
                             .set_attr_dst_type(ge::DT_FLOAT);
    auto cast_off_value = ge::op::Cast((ori_name + "_Cast_off_value").c_str())
                              .set_input_x(const_off_value)
                              .set_attr_dst_type(ge::DT_FLOAT);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    auto OneHotD = ge::op::OneHotD((ori_name + "_OneHotD").c_str())
                       .set_input_x(cast_data0)
                       .set_input_on_value(cast_on_value)
                       .set_input_off_value(cast_off_value)
                       .set_attr_depth(depth)
                       .set_attr_axis(axis);
    outputs.emplace_back(OneHotD, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::NPUOneHot"), ge::AscendString("ai.onnx::12::NPUOneHot"),
                   ge::AscendString("ai.onnx::13::NPUOneHot"), ge::AscendString("ai.onnx::14::NPUOneHot"),
                   ge::AscendString("ai.onnx::15::NPUOneHot"), ge::AscendString("ai.onnx::16::NPUOneHot"),
                   ge::AscendString("ai.onnx::17::NPUOneHot"), ge::AscendString("ai.onnx::18::NPUOneHot"),
                   ge::AscendString("npu::1::NPUOneHot")})
    .ParseParamsByOperatorFn(ParseParamsNpuOneHot)
    .ParseOpToGraphFn(ParseOpToGraphOneHot)
    .ImplyType(ImplyType::TVM);
} // namespace domi
