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
 * \file    logical_or_tiling_arch35.cpp
 * \brief   logical_or_tiling source file
 */

#include "logical_or_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "math/logical_or/op_kernel/arch35/logical_or_dag.h"
#include "math/logical_or/op_kernel/arch35/logical_or_struct.h"
#include "op_host/tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace ge;
using namespace Ops::Math::OpTiling;

namespace optiling {

static constexpr uint64_t LOGICAL_OR_COMMON_TILING_PRIORITY = 0;

ge::graphStatus LogicalOrTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool LogicalOrTiling::IsCapable()
{
    return true;
}

ge::graphStatus LogicalOrTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto outputYDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    ge::DataType outputDtype = outputYDesc->GetDataType();
    if ((input0DType != ge::DT_BOOL) || (input1DType != ge::DT_BOOL) ||
        (outputDtype != ge::DT_BOOL)) {
        OP_LOGE(
           context_->GetNodeName(), "dtype of input0[%s], dtype of input1[%s], dtype of output[%s] are not bool.",
           ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
           ge::TypeUtils::DataTypeToSerialString(input1DType).c_str(),
           ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    
    BroadcastBaseTiling<LogicalOrOp::LogicalOrCompute<uint8_t>::OpDag> brcBaseTiling(context_);
    ge::graphStatus ret = brcBaseTiling.DoTiling();
    tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    return ret;
}

ge::graphStatus LogicalOrTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LogicalOrTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LogicalOrTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogicalOrTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogicalOrTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LogicalOr(gert::TilingContext* context)
{
    OP_LOGD("LogicalOrTiling", "Enter Tiling4LogicalOr");
    if (context == nullptr) {
        OP_LOGE("Tiling4LogicalOr", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc Tiling4LogicalOr");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForBroadcastLogicalOr(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogicalOr)
    .Tiling(Tiling4LogicalOr)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcastLogicalOr);

REGISTER_OPS_TILING_TEMPLATE(LogicalOr, LogicalOrTiling, LOGICAL_OR_COMMON_TILING_PRIORITY);
} // namespace optiling