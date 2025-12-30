/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file space_to_depth_tiling_arch35.cpp
 * \brief
 */

#include "util/platform_util.h"
#include "../../../transpose/op_host/arch35/transpose_tiling_base.h"
#include "../../../transpose/op_host/arch35/transpose_tiling_arch35.h"
#include "../../../transpose/op_host/arch35/transpose_tiling_with_gather_arch35.h"

namespace optiling {

namespace SpaceToDepth {

static ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareTransposeForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (ci->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Transpose Op get coreNum:%ld, ubSize:%ld.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForRelatedToTranspose(gert::TilingParseContext* context)
{
  OP_LOGD(context->GetNodeName(), "Enter TilingPrepareForRelatedToTranspose.");
  TilingPrepareTransposeForAscendC(context);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToDepthTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start SpaceToDepthTilingForAscendC.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForSpaceToDepth(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingForSpaceToDepth");
  return SpaceToDepthTilingForAscendC(context);
}

IMPL_OP_OPTILING(SpaceToDepth)
    .Tiling(TilingForSpaceToDepth)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareForRelatedToTranspose);

} // namespace SpaceToDepth
} // namespace optiling
