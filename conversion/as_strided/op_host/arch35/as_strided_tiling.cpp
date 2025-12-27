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
 * \file as_strided_tiling.cpp
 * \brief as_strided
 */
#include "as_strided_tiling.h"
#include "as_strided_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr size_t IN_SIZE = 1;
constexpr size_t IN_STRIDE = 2;
constexpr size_t IN_OFFSET = 3;

static ge::graphStatus TilingForAsStrided(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "AsStrided tiling running begin");
    const AsStridedCompileInfo* compile_info = reinterpret_cast<const AsStridedCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "TilingForAsStrided dsl compile_info is Null! running simd tiling!");
    uint32_t maxCoreNum = compile_info->maxCoreNum;
    uint32_t ubSizePlatform = compile_info->ubSizePlatform;
    return TilingForAsStridedOfAsc(context, maxCoreNum, ubSizePlatform);
}

static ge::graphStatus TilingPrepareForAsStrided(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<AsStridedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "AscendC tiling is starting!");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compile_info->maxCoreNum <= 0), OP_LOGE(context->GetNodeName(), "The core num is invalid."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ubSizePlatform = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF(
        (compile_info->ubSizePlatform <= 0), OP_LOGE(context->GetNodeName(), "The ubSize is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsStrided)
    .Tiling(TilingForAsStrided)
    .TilingParse<AsStridedCompileInfo>(TilingPrepareForAsStrided)
    .TilingInputsDataDependency({IN_SIZE, IN_STRIDE, IN_OFFSET});
} // namespace optiling
