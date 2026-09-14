/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HANS_950_TILING_TEST_UTILS_H
#define HANS_950_TILING_TEST_UTILS_H

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"

namespace Hans950Test {

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = 0;
    uint32_t blockDim = 0;
    std::vector<uint8_t> data;
    std::vector<size_t> workspaces;

    int64_t Int64(size_t index) const
    {
        int64_t value = 0;
        const size_t offset = index * sizeof(value);
        EXPECT_LE(offset + sizeof(value), data.size());
        if (offset + sizeof(value) <= data.size()) {
            std::memcpy(&value, data.data() + offset, sizeof(value));
        }
        return value;
    }

    bool Bool(size_t offset) const
    {
        EXPECT_LT(offset, data.size());
        return offset < data.size() && data[offset] != 0;
    }
};

// Explicit platform resources ensure these tests enter the 950 branch even
// when the surrounding UT binary was built with a different default SoC.
inline TilingResult Run(const gert::TilingContextPara& parameters, bool ascend950 = true)
{
    TilingResult result;
    EXPECT_NE(parameters.compileInfo_, nullptr);
    if (parameters.compileInfo_ == nullptr) {
        return result;
    }
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> soc = {{"ai_core_cnt", std::to_string(parameters.coreNum_)},
                                              {"vector_core_cnt", std::to_string(parameters.coreNum_)},
                                              {"cube_core_cnt", std::to_string(parameters.coreNum_)},
                                              {"core_type_list", "AICore"}};
    std::map<std::string, std::string> spec = {{"ub_size", std::to_string(parameters.ubSize_)}};
    std::map<std::string, std::string> version = {{"Short_SoC_version", ascend950 ? "Ascend950" : "Ascend910B"},
                                                  {"NpuArch", ascend950 ? "3510" : "2201"}};
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");
    platform_ascendc::PlatformAscendC device(&platform);
    EXPECT_EQ(device.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950, ascend950);
    EXPECT_EQ(device.GetCoreNumAiv(), parameters.coreNum_);
    uint64_t ubSize = 0;
    device.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // PlatformAscendC reserves 256 UB bytes for KFC on 910B, but not on 950.
    const uint64_t reservedUbBytes = ascend950 ? 0 : 256;
    EXPECT_EQ(ubSize, parameters.ubSize_ - reservedUbBytes);

    std::vector<std::unique_ptr<gert::Tensor>> storage;
    auto makeTensors = [&storage](const std::vector<gert::TilingContextPara::TensorDescription>& descriptions) {
        std::vector<gert::Tensor*> tensors;
        for (const auto& desc : descriptions) {
            storage.push_back(std::make_unique<gert::Tensor>(
                desc.shape_, gert::StorageFormat(desc.format_, desc.format_, gert::ExpandDimsType()),
                gert::TensorPlacement::kOnHost, desc.dtype_, nullptr));
            tensors.push_back(storage.back().get());
        }
        return tensors;
    };
    auto inputs = makeTensors(parameters.inputTensorDesc_);
    auto outputs = makeTensors(parameters.outputTensorDesc_);
    gert::TilingContextFaker faker;
    faker.SetOpType(parameters.opName_).NodeIoNum(inputs.size(), outputs.size());
    faker.InputTensors(inputs).OutputTensors(outputs).CompileInfo(parameters.compileInfo_).PlatformInfo(&platform);
    for (const auto& attr : parameters.attrs_) {
        // HANS has only boolean attributes.
        faker.Attr(attr.attrName_, *reinterpret_cast<const bool*>(attr.attr_.valuePtr_.get()));
    }
    auto tilingData = gert::TilingData::CreateCap(parameters.tilingDataSize_);
    auto workspace = gert::ContinuousVector::Create<size_t>(16);
    auto holder = faker.TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto* context = holder.GetContext();
    EXPECT_NE(context, nullptr) << "Failed to build the HANS tiling context";
    if (context == nullptr) {
        return result;
    }
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    EXPECT_NE(registry, nullptr);
    if (registry == nullptr) {
        return result;
    }
    auto implementation = registry->GetOpImpl(parameters.opName_.c_str());
    EXPECT_NE(implementation, nullptr);
    if (implementation == nullptr || implementation->tiling == nullptr) {
        ADD_FAILURE() << "HANS tiling function is not registered";
        return result;
    }
    result.status = implementation->tiling(context);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.key = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    const auto* raw = context->GetRawTilingData();
    const auto* begin = static_cast<const uint8_t*>(raw->GetData());
    result.data.assign(begin, begin + raw->GetDataSize());
    const size_t count = context->GetWorkspaceNum();
    if (count != 0) {
        const auto* sizes = context->GetWorkspaceSizes(count);
        result.workspaces.assign(sizes, sizes + count);
    }
    return result;
}

inline gert::TilingContextPara::TensorDescription Tensor(int64_t elements, ge::DataType dtype)
{
    return {{{elements}, {elements}}, dtype, ge::FORMAT_ND};
}

inline gert::TilingContextPara::OpAttr BoolAttr(const std::string& name, bool value)
{
    return {name, Ops::Math::AnyValue::CreateFrom<bool>(value)};
}

} // namespace Hans950Test
#endif // HANS_950_TILING_TEST_UTILS_H
