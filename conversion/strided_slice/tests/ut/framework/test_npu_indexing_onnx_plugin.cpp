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

#include "../../../framework/npu_indexing_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

// 全属性解析：begins/ends/strides (INTS) 与五个 mask (INT) 均正确转写到 op_dest
TEST(OnnxNpuIndexingPluginTest, ParsesFullAttributes)
{
    const std::string attrs = R"({"attribute":[
        {"name":"begins","type":7,"ints":[0,1]},
        {"name":"ends","type":7,"ints":[10,9]},
        {"name":"strides","type":7,"ints":[1,1]},
        {"name":"begin_mask","type":2,"i":1},
        {"name":"end_mask","type":2,"i":2},
        {"name":"ellipsis_mask","type":2,"i":4},
        {"name":"new_axis_mask","type":2,"i":8},
        {"name":"shrink_axis_mask","type":2,"i":16}
    ]})";
    ge::Operator op_src = CreateSourceOperator(attrs);
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::SUCCESS);

    std::vector<int64_t> begins;
    EXPECT_EQ(op_dest.GetAttr("begin", begins), ge::GRAPH_SUCCESS);
    EXPECT_EQ(begins, (std::vector<int64_t>{0, 1}));
    std::vector<int64_t> ends;
    EXPECT_EQ(op_dest.GetAttr("end", ends), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ends, (std::vector<int64_t>{10, 9}));
    std::vector<int64_t> strides;
    EXPECT_EQ(op_dest.GetAttr("strides", strides), ge::GRAPH_SUCCESS);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 1}));

    int begin_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("begin_mask", begin_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(begin_mask, 1);
    int end_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("end_mask", end_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(end_mask, 2);
    int ellipsis_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("ellipsis_mask", ellipsis_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ellipsis_mask, 4);
    int new_axis_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("new_axis_mask", new_axis_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(new_axis_mask, 8);
    int shrink_axis_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("shrink_axis_mask", shrink_axis_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(shrink_axis_mask, 16);

    std::string name;
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_EQ(name, "src");
    std::string original_type;
    EXPECT_EQ(op_dest.GetAttr("original_type", original_type), ge::GRAPH_SUCCESS);
    EXPECT_EQ(original_type, "npu::1::NPUIndexing");
}

// 缺失 attribute 字段：begins/ends/strides 为空，mask 默认 0，name 取源算子名
TEST(OnnxNpuIndexingPluginTest, NoAttributeUsesDefaults)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::SUCCESS);

    std::vector<int64_t> begins;
    EXPECT_EQ(op_dest.GetAttr("begin", begins), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(begins.empty());
    std::vector<int64_t> ends;
    EXPECT_EQ(op_dest.GetAttr("end", ends), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(ends.empty());
    std::vector<int64_t> strides;
    EXPECT_EQ(op_dest.GetAttr("strides", strides), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(strides.empty());

    int begin_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("begin_mask", begin_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(begin_mask, 0);

    std::string name;
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_EQ(name, "src");
    std::string original_type;
    EXPECT_EQ(op_dest.GetAttr("original_type", original_type), ge::GRAPH_SUCCESS);
    EXPECT_EQ(original_type, "npu::1::NPUIndexing");
}

// 合法 JSON 但 attribute 数组为空：保持默认值并返回 SUCCESS
TEST(OnnxNpuIndexingPluginTest, EmptyAttributeArrayKeepsDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::SUCCESS);

    std::vector<int64_t> begins;
    EXPECT_EQ(op_dest.GetAttr("begin", begins), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(begins.empty());
    int begin_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("begin_mask", begin_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(begin_mask, 0);
}

// 仅设置 begin_mask：向量保持空，其余 mask 保持默认 0
TEST(OnnxNpuIndexingPluginTest, PartialAttributesPreserveDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"begin_mask","type":2,"i":5}]})");
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::SUCCESS);

    std::vector<int64_t> begins;
    EXPECT_EQ(op_dest.GetAttr("begin", begins), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(begins.empty());
    int begin_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("begin_mask", begin_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(begin_mask, 5);
    int end_mask = -1;
    EXPECT_EQ(op_dest.GetAttr("end_mask", end_mask), ge::GRAPH_SUCCESS);
    EXPECT_EQ(end_mask, 0);
}

// 非法 JSON 字符串：应返回 FAILED
TEST(OnnxNpuIndexingPluginTest, MalformedJsonReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator("not-a-json");
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::FAILED);
}

// ints 字段类型不匹配（非数组）：get<vector> 抛异常，应返回 FAILED
TEST(OnnxNpuIndexingPluginTest, WrongIntsTypeReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"ends","type":7,"ints":"not-array"}]})");
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::FAILED);
}

// i 字段类型不匹配（非整数）：get<int> 抛异常，应返回 FAILED
TEST(OnnxNpuIndexingPluginTest, WrongMaskTypeReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"begin_mask","type":2,"i":"not-int"}]})");
    ge::Operator op_dest = CreateOperator("npu_indexing");

    EXPECT_EQ(domi::ParseParamIndexing(op_src, op_dest), domi::FAILED);
}
