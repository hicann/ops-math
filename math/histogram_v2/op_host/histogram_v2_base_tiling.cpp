/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_base_tiling.cpp
 * \brief
 */
#include "histogram_v2_tiling.h"
#include "log/log.h"

namespace optiling {
constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 0;

ge::graphStatus HistogramV2BaseClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        npuArch = ascendcPlatform.GetCurNpuArch();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        aicoreParams_.ubSize = ubSizePlatform;
    } else {
        auto compileInfoPtr = reinterpret_cast<const HistogramV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        npuArch = compileInfoPtr->npuArch;
        aicoreParams_.ubSize = compileInfoPtr->ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2BaseClass::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus HistogramV2BaseClass::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
