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

#include "../../../framework/one_hot_d_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

// 全属性解析：depth/num_classes 正确转写，name/original_type 正确
TEST(OnnxOneHotDPluginTest, ParsesDepthAndNumClasses)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"depth","type":2,"i":3},{"name":"num_classes","type":2,"i":5}]})");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::SUCCESS);

    int depth = 0;
    EXPECT_EQ(op_dest.GetAttr("depth", depth), ge::GRAPH_SUCCESS);
    EXPECT_EQ(depth, 3);
    int num_classes = 0;
    EXPECT_EQ(op_dest.GetAttr("num_classes", num_classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_classes, 5);

    std::string name;
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_EQ(name, "src");
    std::string original_type;
    EXPECT_EQ(op_dest.GetAttr("original_type", original_type), ge::GRAPH_SUCCESS);
    EXPECT_EQ(original_type, "npu::1::NPUOneHot");
}

// 仅设置 depth：num_classes 用默认值 -1
TEST(OnnxOneHotDPluginTest, DepthOnlyUsesDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"depth","type":2,"i":4}]})");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::SUCCESS);

    int depth = 0;
    EXPECT_EQ(op_dest.GetAttr("depth", depth), ge::GRAPH_SUCCESS);
    EXPECT_EQ(depth, 4);
    int num_classes = 123;
    EXPECT_EQ(op_dest.GetAttr("num_classes", num_classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_classes, -1);
}

// 缺失必需的 depth 属性：应返回 FAILED
TEST(OnnxOneHotDPluginTest, MissingDepthReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"num_classes","type":2,"i":5}]})");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::FAILED);
}

// 无 attribute 字段：depth 缺失，应返回 FAILED
TEST(OnnxOneHotDPluginTest, NoAttributeReturnsFailed)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::FAILED);
}

// 合法 JSON 但 attribute 数组为空：depth 缺失，应返回 FAILED
TEST(OnnxOneHotDPluginTest, EmptyAttributeArrayReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::FAILED);
}

// 非法 JSON 字符串：应返回 FAILED
TEST(OnnxOneHotDPluginTest, MalformedJsonReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator("not-a-json");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::FAILED);
}

// depth 的 i 字段类型不匹配（非整数）：应返回 FAILED
TEST(OnnxOneHotDPluginTest, WrongDepthItypeReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"depth","type":2,"i":"not-int"}]})");
    ge::Operator op_dest = CreateOperator("one_hot_d");

    EXPECT_EQ(domi::ParseParamsNpuOneHot(op_src, op_dest), domi::FAILED);
}
