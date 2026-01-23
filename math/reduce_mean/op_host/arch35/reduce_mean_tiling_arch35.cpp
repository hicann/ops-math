/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_mean_tiling_arch35.cpp
 * \brief
 */

#include <vector>
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "op_host/tiling_util.h"
#include "util/platform_util.h"
#include "math/reduce_mean/op_kernel/arch35/reduce_mean_dag.h"
#include "math/reduce_mean/op_kernel/arch35/reduce_mean_tiling_key.h"
#include "reduce_mean_tiling_arch35.h"

using namespace Ops::Base;

namespace optiling {
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;


static ge::graphStatus DoTilingAscendC(gert::TilingContext* context, const ReduceOpCompileInfo* compileInfo,
                                       ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<ReduceMean::ReduceMeanDag<float, float>::OpDag>(context, opInput, key,
                                                                                 compileInfo);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status =
            Tiling4ReduceOp<ReduceMean::ReduceMeanDag<half, float>::OpDag>(context, opInput, key, compileInfo);
    }

    OP_CHECK_IF((status == ge::GRAPH_FAILED),
                    OP_LOGE(
                        context->GetNodeName(), "ReduceOp Tiling failed, dtype shoude be in (bfloat16/float16/float)"),
                    return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4ReduceMean(gert::TilingContext* context) {
    auto compile_info = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(context, opInput, 0, 1, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"),
                    return ge::GRAPH_FAILED);

    if (opInput.axes.empty()) {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const bool isNoopWithEmpty = *(attrs->GetAttrPointer<bool>(1));
        if (!isNoopWithEmpty) {
            opInput.axes.resize(opInput.shape.size());
            for (size_t i = 0; i < opInput.shape.size(); i++) {
                opInput.axes[i] = i;
            }
        }
    }

    ReduceTilingKey key;
    OP_CHECK_IF((DoTilingAscendC(context, compile_info, opInput, key) == ge::GRAPH_FAILED),
                    OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceMean"),
                    return ge::GRAPH_FAILED);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(key.patternID, key.loopARCount, key.loopInnerARCount);
    OP_LOGI(context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
            key.patternID, key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceMean(gert::TilingParseContext* context) {
    auto compileInfo = context->GetCompiledInfo<ReduceOpCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum == 0UL),
        OP_LOGE(
            context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu", compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= CACHE_BUF_SIZE,
        OP_LOGE(
            context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lu.", compileInfo->ubSize,
            CACHE_BUF_SIZE),
        return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = GetCacheLineSize(context);
    OP_CHECK_IF(
        compileInfo->cacheLineSize == 0UL,
        OP_LOGE(
            context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, cacheLineSize:%lu.", compileInfo->cacheLineSize),
        return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0UL,
        OP_LOGE(
            context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubBlockSize:%lu.", compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = GetVRegSize(context);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0UL,
        OP_LOGE(
            context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vRegSize:%lu.", compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
        compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
        compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ReduceMean op.
IMPL_OP_OPTILING(ReduceMean)
    .Tiling(Tiling4ReduceMean)
    .TilingParse<ReduceOpCompileInfo>(TilingPrepare4ReduceMean);
}  // namespace optiling
