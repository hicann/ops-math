/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_tiling_arch35.cpp
\brief tiling implementation for grouped_bias_add_grad arch35
*/
#include "grouped_bias_add_grad_tiling_arch35.h"
#include "grouped_bias_add_grad_RA_tiling_arch35.h"
#include "grouped_bias_add_grad_ARA_tiling_arch35.h"
#include "log/log.h"

namespace optiling {
using namespace Ops::Base;
static ge::graphStatus TilingGetCompileInfo(
    gert::TilingContext* context, GroupedBiasAddGradCompileInfoArch35* compileInfo)
{
    OP_LOGD(context, "Enter TilingGetCompileInfo.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "core num is negative."), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0L), OP_LOGE(context, "fail to get ub size."), return ge::GRAPH_FAILED);

    compileInfo->clSize = CACHELINE_DEFINE;
    OP_CHECK_IF((compileInfo->clSize <= 0U), OP_LOGE(context, "fail to get cache line size."), return ge::GRAPH_FAILED);

    compileInfo->blockSize = GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->blockSize <= 0), OP_LOGE(context, "fail to get block size."), return ge::GRAPH_FAILED);

    compileInfo->vRegSize = GetVRegSize(context);
    OP_CHECK_IF((compileInfo->vRegSize <= 0U), OP_LOGE(context, "fail to get vReg size."), return ge::GRAPH_FAILED);

    OP_LOGD(context, "Exit TilingGetCompileInfo.");
    return ge::GRAPH_SUCCESS;
}
// External tiling entry functions
ge::graphStatus Tiling4GroupedBiasAddGradArch35(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Start tiling for GroupedBiasAddGrad Arch35.");
    GroupedBiasAddGradCompileInfoArch35 compileInfo;
    TilingGetCompileInfo(context, &compileInfo);
    if (IsGroupedBiasAddGradARA(context)) {
        return Tiling4GroupedBiasAddGradARA(context, &compileInfo);
    } else {
        GroupedBiasAddGradTilingArch35 tiling(context, &compileInfo);
        return tiling.DoTiling();
    }
}

ge::graphStatus TilingPrepare4GroupedBiasAddGradArch35(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "Start tiling prepare for GroupedBiasAddGrad Arch35.");
    auto compileInfo = context->GetCompiledInfo<GroupedBiasAddGradCompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum == 0), OP_LOGE(context->GetNodeName(), "Failed to get core number."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((ubSize == 0), OP_LOGE(context->GetNodeName(), "Failed to get UB size."), return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    OP_LOGD(context->GetNodeName(), "Compile info: coreNum=%u, ubSize=%lu", compileInfo->coreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

// Register tiling implementation
IMPL_OP_OPTILING(GroupedBiasAddGrad)
    .Tiling(Tiling4GroupedBiasAddGradArch35)
    .TilingParse<GroupedBiasAddGradCompileInfoArch35>(TilingPrepare4GroupedBiasAddGradArch35);

} // namespace optiling