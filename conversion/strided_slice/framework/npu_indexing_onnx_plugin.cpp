/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_indexing_onnx_plugin.cpp
 * \brief onnx plugin for custom operator npu_indexing
 */

#include "graph/operator.h"
#include "graph/graph.h"
#include "stub_ops.h"
#include "conversion/strided_slice/op_graph/strided_slice_proto.h"
#include "math_onnx_plugin_util.h"
#include "register/register.h"
#include "nlohmann/json.hpp"
#include "log/log.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamIndexing(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::vector<int64_t> ends = {};
    std::vector<int64_t> begins = {};
    std::vector<int64_t> strides = {};
    int begin_mask = 0;
    int end_mask = 0;
    int ellipsis_mask = 0;
    int new_axis_mask = 0;
    int shrink_axis_mask = 0;

    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            const json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (const json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "ends" && attr.contains("ints")) {
                        ends = attr["ints"].get<std::vector<int64_t>>();
                    } else if (name == "begins" && attr.contains("ints")) {
                        begins = attr["ints"].get<std::vector<int64_t>>();
                    } else if (name == "strides" && attr.contains("ints")) {
                        strides = attr["ints"].get<std::vector<int64_t>>();
                    } else if (name == "begin_mask" && attr.contains("i")) {
                        begin_mask = attr["i"].get<int>();
                    } else if (name == "end_mask" && attr.contains("i")) {
                        end_mask = attr["i"].get<int>();
                    } else if (name == "ellipsis_mask" && attr.contains("i")) {
                        ellipsis_mask = attr["i"].get<int>();
                    } else if (name == "new_axis_mask" && attr.contains("i")) {
                        new_axis_mask = attr["i"].get<int>();
                    } else if (name == "shrink_axis_mask" && attr.contains("i")) {
                        shrink_axis_mask = attr["i"].get<int>();
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
    // exposes it as the source operator's own name. Fall back to the dest op
    // name when the source name is unavailable (matches the migrated
    // PartitionedCall plugins such as Shape).
    const std::string op_name = GetOpName(op_dest);
    ge::AscendString source_name_string;
    const std::string source_name = op_src.GetName(source_name_string) == ge::GRAPH_SUCCESS ?
                                        source_name_string.GetString() :
                                        std::string();
    const std::string node_name = source_name.empty() ? op_name : source_name;

    op_dest.SetAttr("name", node_name);
    op_dest.SetAttr("end", ends);
    op_dest.SetAttr("begin", begins);
    op_dest.SetAttr("strides", strides);
    op_dest.SetAttr("begin_mask", begin_mask);
    op_dest.SetAttr("end_mask", end_mask);
    op_dest.SetAttr("ellipsis_mask", ellipsis_mask);
    op_dest.SetAttr("new_axis_mask", new_axis_mask);
    op_dest.SetAttr("shrink_axis_mask", shrink_axis_mask);

    op_dest.SetAttr("original_type", "npu::1::NPUIndexing");
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static ge::Operator NPUMakeConstOp(const ge::Operator& op, const std::string& attr_name)
{
    std::vector<int64_t> val = {};
    op.GetAttr(attr_name.c_str(), val);
    std::vector<int64_t> dims(1, val.size());
    ge::Tensor tensor = Vec2Tensor(val, dims, ge::DT_INT64);
    ge::Operator const_op = ge::op::Const(attr_name.c_str()).set_attr_value(tensor);
    return const_op;
}

static Status ParseOpToGraphIndexing(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int begin_mask = 0;
    op.GetAttr("begin_mask", begin_mask);
    int end_mask = 0;
    op.GetAttr("end_mask", end_mask);
    int ellipsis_mask = 0;
    op.GetAttr("ellipsis_mask", ellipsis_mask);
    int new_axis_mask = 0;
    op.GetAttr("new_axis_mask", new_axis_mask);
    int shrink_axis_mask = 0;
    op.GetAttr("shrink_axis_mask", shrink_axis_mask);

    auto data_op = ge::op::Data((ori_name + "_input1").c_str()).set_attr_index(0);
    auto const_op = NPUMakeConstOp(op, "begin");
    auto const_op1 = NPUMakeConstOp(op, "end");
    auto const_op2 = NPUMakeConstOp(op, "strides");
    auto slice_op = ge::op::StridedSlice((ori_name + "_StridedSlice").c_str())
                        .set_input_x(data_op)
                        .set_input_begin(const_op)
                        .set_input_end(const_op1)
                        .set_input_strides(const_op2)
                        .set_attr_begin_mask(begin_mask)
                        .set_attr_end_mask(end_mask)
                        .set_attr_ellipsis_mask(ellipsis_mask)
                        .set_attr_new_axis_mask(new_axis_mask)
                        .set_attr_shrink_axis_mask(shrink_axis_mask);

    std::vector<ge::Operator> inputs{data_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(slice_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUIndexing"), ge::AscendString("ai.onnx::11::NPUIndexing"),
                   ge::AscendString("ai.onnx::12::NPUIndexing"), ge::AscendString("ai.onnx::13::NPUIndexing"),
                   ge::AscendString("ai.onnx::14::NPUIndexing"), ge::AscendString("ai.onnx::15::NPUIndexing"),
                   ge::AscendString("ai.onnx::16::NPUIndexing"), ge::AscendString("ai.onnx::17::NPUIndexing"),
                   ge::AscendString("ai.onnx::18::NPUIndexing")})
    .ParseParamsByOperatorFn(ParseParamIndexing)
    .ParseOpToGraphFn(ParseOpToGraphIndexing)
    .ImplyType(ImplyType::TVM);
} // namespace domi
