/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_tiling.h
 * \brief
 */
#ifndef MATH_HISTOGRAM_V2_H
#define MATH_HISTOGRAM_V2_H
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HistogramV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, bins);
TILING_DATA_FIELD_DEF(int64_t, ubBinsLength);

TILING_DATA_FIELD_DEF(int64_t, formerNum);
TILING_DATA_FIELD_DEF(int64_t, formerLength);
TILING_DATA_FIELD_DEF(int64_t, formerLengthAligned);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, tailLengthAligned);

TILING_DATA_FIELD_DEF(int64_t, formerTileNum);
TILING_DATA_FIELD_DEF(int64_t, formerTileDataLength);
TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLength);
TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLengthAligned);

TILING_DATA_FIELD_DEF(int64_t, tailTileNum);
TILING_DATA_FIELD_DEF(int64_t, tailTileDataLength);
TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLength);
TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLengthAligned);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HistogramV2, HistogramV2TilingData)

BEGIN_TILING_DATA_DEF(HistogramV2SimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, bins);
TILING_DATA_FIELD_DEF(int64_t, ubNumCanUse);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNum);
TILING_DATA_FIELD_DEF(int64_t, needXCoreNum);
TILING_DATA_FIELD_DEF(int64_t, formerLength);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, clearYCoreNum);
TILING_DATA_FIELD_DEF(int64_t, clearYFactor);
TILING_DATA_FIELD_DEF(int64_t, clearYTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HistogramV2_101, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_102, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_103, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_104, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_105, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_106, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_107, HistogramV2SimtTilingData)

REGISTER_TILING_DATA_CLASS(HistogramV2_201, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_202, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_203, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_204, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_205, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_206, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_207, HistogramV2SimtTilingData)

REGISTER_TILING_DATA_CLASS(HistogramV2_301, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_302, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_303, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_304, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_305, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_306, HistogramV2SimtTilingData)
REGISTER_TILING_DATA_CLASS(HistogramV2_307, HistogramV2SimtTilingData)

struct HistogramV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

class HistogramV2BaseClass : public Ops::Math::OpTiling::TilingBaseClass
{
public:
    explicit HistogramV2BaseClass(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;

    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};
} // namespace optiling
#endif // MATH_HISTOGRAM_V2_H