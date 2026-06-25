/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trilu_tiling.cpp
 * \brief Tiling implementation for trilu operator
 *
 * v2.0: diagonal is read from OPTIONAL_INPUT("k") instead of ATTR
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../op_kernel/arch35/trilu_tiling_data.h"
#include "../../op_kernel/arch35/trilu_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

constexpr int64_t MIN_PER_CORE_ELEMENTS = 1024;
constexpr int64_t ALIGN_SIZE = 32;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr uint32_t WS_SYS_SIZE = 0U;

struct TriluCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndAttrsInfo(gert::TilingContext* context,
    int64_t& totalElements, int64_t& h, int64_t& w, int64_t& upper)
{
    auto xInput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    auto xShape = xInput->GetStorageShape();

    size_t dimNum = xShape.GetDimNum();
    totalElements = 1;
    for (size_t i = 0; i < dimNum; i++) {
        totalElements *= xShape.GetDim(i);
    }

    if (dimNum >= 2) {
        h = xShape.GetDim(dimNum - 2);
        w = xShape.GetDim(dimNum - 1);
    } else if (dimNum == 1) {
        h = 1;
        w = xShape.GetDim(0);
    } else {
        h = 1;
        w = 1;
    }

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* upperPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, upperPtr);
    upper = *upperPtr;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetLocalMemory(gert::TilingContext* context, uint64_t ubSize)
{
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
        OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize),
        return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(
        static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "SetLocalMemorySize failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TriluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalElements, h, w, upper;
    OP_CHECK_IF(
        GetShapeAndAttrsInfo(context, totalElements, h, w, upper) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndAttrsInfo error"), return ge::GRAPH_FAILED);

    context->GetWorkspaceSizes(1)[0] = WS_SYS_SIZE;

    TriluTilingData* tiling = context->GetTilingData<TriluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TriluTilingData), 0, sizeof(TriluTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalElements = totalElements;
    int64_t perCoreElements = totalElements > 0 ?
        Ops::Base::CeilAlign(Ops::Base::CeilDiv(totalElements, coreNum), ALIGN_SIZE) : 0;
    perCoreElements = std::max(MIN_PER_CORE_ELEMENTS, perCoreElements);
    tiling->perCoreElements = perCoreElements;
    tiling->needCoreNum = totalElements > 0 ?
        static_cast<int32_t>(Ops::Base::CeilDiv(totalElements, perCoreElements)) : 1;
    tiling->lastCoreElements = totalElements - perCoreElements * (tiling->needCoreNum - 1);
    tiling->upper = static_cast<int32_t>(upper);
    tiling->h = h;
    tiling->w = w;

    context->SetBlockDim(tiling->needCoreNum);
    OP_CHECK_IF(SetLocalMemory(context, ubSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetLocalMemory error"), return ge::GRAPH_FAILED);
    context->SetTilingKey(GET_TPL_TILING_KEY(TRILU_TPL_SCH_MODE_0));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrilu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Trilu)
    .Tiling(TriluTilingFunc)
    .TilingParse<TriluCompileInfo>(TilingParseForTrilu);
}  // namespace optiling
