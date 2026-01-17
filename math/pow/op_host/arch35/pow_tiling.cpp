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
 * \file pow_tiling.cpp
 * \brief
 */

#include "pow_tiling_arch35.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/util/platform_util.h"

namespace optiling {
using namespace Ops::Math::OpTiling;

static ge::graphStatus TilingForPow(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const PowCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForPow(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4Pow enter.");
    auto compileInfo = context->GetCompiledInfo<PowCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get core num failed, core num: %u",
            static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);
    compileInfo->isRegBase = (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95 ||
                                ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::MC62CM12A) ?
                                    true :
                                    false;
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4Pow exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Pow).Tiling(TilingForPow).TilingParse<PowCompileInfo>(TilingPrepareForPow);
} // namespace optiling