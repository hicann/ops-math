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

namespace domi {
using json = nlohmann::json;

static Status ParseParamsCorr(const ge::Operator& op_src, ge::Operator& op_dest)
{
    OP_LOGD(GetOpName(op_dest).c_str(), "start ParseParamsCorr");

    int64_t groups = 1;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            const json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (const json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "groups" && attr.contains("i")) {
                        groups = attr["i"].get<int64_t>();
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

    // The ONNX node name is no longer reachable through the protobuf node type; the parser now
    // exposes it as the source operator's own name. Fall back to the dest op name when the source
    // name is unavailable.
    const std::string op_name = GetOpName(op_dest);
    ge::AscendString source_name_string;
    const std::string source_name = op_src.GetName(source_name_string) == ge::GRAPH_SUCCESS ?
                                        source_name_string.GetString() :
                                        std::string();
    const std::string node_name = source_name.empty() ? op_name : source_name;

    op_dest.SetAttr("name", node_name);
    op_dest.SetAttr("groups", groups);
    const int input_number = 2;
    op_dest.DynamicInputRegister("x", input_number);
    op_dest.DynamicOutputRegister("y", 1);

    op_dest.SetAttr("original_type", "ai.onnx::11::Corr");
    return SUCCESS;
}

static Status ParseOpToGraphCorr(const ge::Operator& op, ge::Graph& graph)
{
    OP_LOGD(GetOpName(op).c_str(), "start ParseOpToGraphCorr");
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto correlation = ge::op::Correlation((ori_name + "_Correlation").c_str());
    // update input format
    ge::Operator& new_op = const_cast<ge::Operator&>(op);
    ge::TensorDesc orgTensorX = op.GetInputDescByName("x");
    orgTensorX.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorX.SetFormat(ge::FORMAT_NCHW);
    new_op.UpdateInputDesc("x", orgTensorX);
    ge::TensorDesc orgTensorFilter = op.GetInputDescByName("filter");
    orgTensorFilter.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorFilter.SetFormat(ge::FORMAT_NCHW);
    new_op.UpdateInputDesc("filter", orgTensorFilter);
    // update output format
    ge::TensorDesc orgTensorY = op.GetOutputDescByName("y");
    orgTensorY.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorY.SetFormat(ge::FORMAT_NCHW);
    new_op.UpdateOutputDesc("y", orgTensorY);

    // Impl Onnx Corr[data,kernel]->Tbe Correlation[kernel,data]
    correlation.set_input_x(data0);
    correlation.set_input_filter(data1);

    // update attr groups
    int groups = 1;
    if (op.GetAttr("groups", groups) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get groups from op failed");
        return FAILED;
    }
    correlation.set_attr_groups(groups);

    std::vector<ge::Operator> inputs{data0, data1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(correlation, std::vector<std::size_t>{0});

    graph.SetInputs(inputs);
    graph.SetOutputs(outputs);
    return SUCCESS;
}

// register Correlation op info to GE
// Customer customized Corr
// tbe Correlation -> onnx Corr
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Corr"), ge::AscendString("ai.onnx::9::Corr"),
                   ge::AscendString("ai.onnx::10::Corr"), ge::AscendString("ai.onnx::11::Corr"),
                   ge::AscendString("ai.onnx::12::Corr"), ge::AscendString("ai.onnx::13::Corr"),
                   ge::AscendString("ai.onnx::14::Corr"), ge::AscendString("ai.onnx::15::Corr"),
                   ge::AscendString("ai.onnx::16::Corr")})
    .ParseParamsByOperatorFn(ParseParamsCorr)
    .ParseOpToGraphFn(ParseOpToGraphCorr)
    .ImplyType(ImplyType::TVM);
} // namespace domi
