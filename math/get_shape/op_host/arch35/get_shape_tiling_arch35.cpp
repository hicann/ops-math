/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "math/get_shape/op_kernel/arch35/get_shape_tiling_data.h"
#include "get_shape_tiling_arch35.h"

namespace optiling {

static ge::graphStatus GetShapeTilingFunc(gert::TilingContext* context)
{
    context->SetBlockDim(1);
    context->SetTilingKey(0);

    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();

    auto tilingData = context->GetTilingData<GetShapeTilingData>();
    tilingData->inputNum = static_cast<int32_t>(inputNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGetShape([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GetShape).Tiling(GetShapeTilingFunc).TilingParse<GetShapeCompileInfo>(TilingParseForGetShape);

} // namespace optiling
