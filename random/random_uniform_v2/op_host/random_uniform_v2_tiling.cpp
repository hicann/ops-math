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
 * \file random_uniform_v2_tiling.cpp
 * \brief
 */
#include "random_uniform_v2_tiling.h"
#include "exe_graph/runtime/shape.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "random_uniform_v2_tiling_arch35.h"
using namespace ge;

namespace optiling {
static ge::graphStatus TilingPrepare4RandomUniformV2Tiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<RandomUniformV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "RandomUniformV2 GetHardwareInfo Failed, vectorCoreNum:%ld, ubSize:%ld.", compileInfo->totalCoreNum,
            compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingRandomUniformV2(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingRandomUniformV2");
    RandomUniformV2Tiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(RandomUniformV2)
    .Tiling(TilingRandomUniformV2)
    .TilingParse<RandomUniformV2CompileInfo>(TilingPrepare4RandomUniformV2Tiling)
    .TilingInputsDataDependency({0});

} // namespace optiling