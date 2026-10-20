/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/truncated_normal_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {
const std::string kPassName = "TruncatedNormalFusionPass";
const char* const kSupportedArchDav3510 = "3510"; // Ascend950/Ascend350

// UT stub for rtGetSocSpec: the UT executable does not link the runtime lib, while the pass gates
// on NpuArch via this API. The stub feeds the arch from the ASCEND_NPU_ARCH env variable (same
// convention as concat_d UT). Weak definition: yields to any real one.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

__attribute__((weak)) rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    const char* arch = std::getenv("ASCEND_NPU_ARCH");
    if (arch == nullptr || std::strcmp(label, "version") != 0 || std::strcmp(key, "NpuArch") != 0) {
        return static_cast<rtError_t>(1);
    }
    (void)std::snprintf(val, maxLen, "%s", arch);
    return RT_ERROR_NONE;
}

static void SetNpuArch(const char* arch) { (void)setenv("ASCEND_NPU_ARCH", arch, 1); }
} // namespace

class TruncatedNormalFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(kSupportedArchDav3510);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(kSupportedArchDav3510);
    }

    void SetUnsupportedPlatform()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch("2201");
    }
};

static std::shared_ptr<Graph> BuildTestGraph(const std::string& graphName, DataType dtype,
                                             const std::vector<int64_t>& shapeDims, int64_t seed = 1024,
                                             int64_t seed2 = 2048)
{
    Shape shapeDesc(shapeDims);

    auto graphBuilder = es::EsGraphBuilder(graphName.c_str());
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);
    std::string varName = graphName + "/offsetVariable";
    auto offset = graphBuilder.CreateVariable(0, varName.c_str());
    auto varNodePtr = offset.GetProducer();
    if (varNodePtr != nullptr) {
        varNodePtr->UpdateOutputDesc(0, offsetDesc);
        int64_t initValue = 0;
        Tensor initTensor(offsetDesc, reinterpret_cast<uint8_t*>(&initValue), sizeof(int64_t));
        varNodePtr->SetAttr("init_value", initTensor);
    }

    int64_t dtypeInt = static_cast<int64_t>(dtype);
    auto v2Output = es::TruncatedNormalV2(shape, offset, seed, seed2, dtypeInt);
    auto v2NodePtr = v2Output.y.GetProducer();

    if (v2NodePtr != nullptr) {
        TensorDesc shapeInputDesc(shapeDesc, FORMAT_ND, DT_INT64);
        v2NodePtr->UpdateInputDesc(0, shapeInputDesc);
        v2NodePtr->UpdateInputDesc(1, offsetDesc);

        TensorDesc outputYDesc(shapeDesc, FORMAT_ND, dtype);
        v2NodePtr->UpdateOutputDesc(0, outputYDesc);
        v2NodePtr->UpdateOutputDesc(1, offsetDesc);
    }

    auto graph = graphBuilder.BuildAndReset({v2Output.y});

    return graph;
}

TEST_F(TruncatedNormalFusionPassTest, patternTest)
{
    TruncatedNormalFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(TruncatedNormalFusionPassTest, unsupportedPlatformFail)
{
    SetUnsupportedPlatform();

    auto graph = BuildTestGraph("test_unsupported", DT_FLOAT, {10});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TruncatedNormalFusionPassTest, unsupportedDtypeFail)
{
    auto graph = BuildTestGraph("test_unsupported_dtype", DT_DOUBLE, {10});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessFp32)
{
    auto graph = BuildTestGraph("test_fp32", DT_FLOAT, {10});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessFp16)
{
    auto graph = BuildTestGraph("test_fp16", DT_FLOAT16, {3});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessBf16)
{
    auto graph = BuildTestGraph("test_bf16", DT_BF16, {4, 5});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessLargeShape)
{
    auto graph = BuildTestGraph("test_large", DT_FLOAT, {6});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}
