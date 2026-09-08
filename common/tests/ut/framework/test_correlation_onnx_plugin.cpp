/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../src/framework/correlation_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

// 全属性解析：groups (INT) 正确转写到 op_dest
TEST(OnnxCorrelationPluginTest, ParsesGroupsAttribute)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"groups","type":2,"i":32}]})");
    ge::Operator op_dest = CreateOperator("correlation");

    EXPECT_EQ(domi::ParseParamsCorr(op_src, op_dest), domi::SUCCESS);

    int64_t groups = -1;
    EXPECT_EQ(op_dest.GetAttr("groups", groups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(groups, 32);

    std::string name;
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_EQ(name, "src");
    std::string original_type;
    EXPECT_EQ(op_dest.GetAttr("original_type", original_type), ge::GRAPH_SUCCESS);
    EXPECT_EQ(original_type, "ai.onnx::11::Corr");
}

// 缺失 attribute 字段：groups 默认 1，name 取源算子名
TEST(OnnxCorrelationPluginTest, NoAttributeUsesDefaults)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("correlation");

    EXPECT_EQ(domi::ParseParamsCorr(op_src, op_dest), domi::SUCCESS);

    int64_t groups = -1;
    EXPECT_EQ(op_dest.GetAttr("groups", groups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(groups, 1);
}

// 合法 JSON 但 attribute 数组为空：保持默认 groups=1 并返回 SUCCESS
TEST(OnnxCorrelationPluginTest, EmptyAttributeArrayKeepsDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("correlation");

    EXPECT_EQ(domi::ParseParamsCorr(op_src, op_dest), domi::SUCCESS);

    int64_t groups = -1;
    EXPECT_EQ(op_dest.GetAttr("groups", groups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(groups, 1);
}

// 非法 JSON 字符串：应返回 FAILED
TEST(OnnxCorrelationPluginTest, MalformedJsonReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator("not-a-json");
    ge::Operator op_dest = CreateOperator("correlation");

    EXPECT_EQ(domi::ParseParamsCorr(op_src, op_dest), domi::FAILED);
}

// groups 的 i 字段类型不匹配（非整数）：应返回 FAILED
TEST(OnnxCorrelationPluginTest, WrongGroupsItypeReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"groups","type":2,"i":"not-int"}]})");
    ge::Operator op_dest = CreateOperator("correlation");

    EXPECT_EQ(domi::ParseParamsCorr(op_src, op_dest), domi::FAILED);
}
