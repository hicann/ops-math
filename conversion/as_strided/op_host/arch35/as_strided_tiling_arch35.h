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
 * \file as_strided_tiling_arch35.h
 * \brief as_strided
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_AS_STRIDED_TILING_ARCH35_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_AS_STRIDED_TILING_ARCH35_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"
#include "conversion/as_strided/op_kernel/arch35/as_strided_struct.h"
#include "op_host/tiling_base.h"

namespace optiling {

struct AsStridedCompileInfo {
    uint32_t ubSizePlatform{0};
    uint32_t maxCoreNum{0};
};

struct AsStridedRunInfo {
    gert::Shape outputSize;
    gert::Shape outputStride;
    AsStridedRunInfo() {
        outputSize.SetDimNum(0);
        outputStride.SetDimNum(0);
    }
};

class AsStridedTilingClass
{
public:
    ge::graphStatus TilingForAsStridedOfAsc(gert::TilingContext *context, uint32_t maxCoreNum, uint32_t ubSizePlatform,
                                        AsStridedRunInfo& runInfo, int64_t storageOffset);
    ge::graphStatus NDDMAForAsStrided(gert::TilingContext* context, AsStridedTilingParam& tilingParam, gert::Shape outSize,
                                gert::Shape outStride, AsStridedTilingData& tiling);
    ge::graphStatus SingleCutOfNDDMAForAsStrided(gert::TilingContext* context, AsStridedTilingParam& tilingParam, gert::Shape outSize, 
                                gert::Shape outStride, AsStridedTilingData& tiling);
    ge::graphStatus SetTilingData(gert::TilingContext* context, AsStridedTilingData& tiling,
                                    AsStridedTilingParam& tilingParam);
    void NoTilingMergeAxis(gert::TilingContext* context, AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam, gert::Shape outSize);
    void MergeAxisAfterTiling([[maybe_unused]] const AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::TilingContext* context);
    ge::graphStatus AsStridedSetTilingData(gert::TilingContext* context, AsStridedTilingData& tilingData);
    void SetZeroStrideTilingData(gert::TilingContext* context, AsStridedTilingParam& tilingParam);
    void SetSimtTilingData(gert::TilingContext* context, AsStridedTilingParam& tilingParam);
    void SetWithGatherTilingData(gert::TilingContext* context, AsStridedUbGatherParam& ubGatherParam);

private:
    // 各模板tilingData
    AsStridedTilingData* tilingData_{nullptr};
    AsStridedSimtTilingData* simtTilingData_{nullptr};
    AsStridedZeroStrideTilingData* zeroStrideTilingData_{nullptr};
    AsStridedWithGatherTilingData* gatherTilingData_{nullptr};
};

} // namespace optiling

#endif