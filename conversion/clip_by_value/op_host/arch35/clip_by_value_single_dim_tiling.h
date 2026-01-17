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
 * \file clip_by_value_single_dim_tiling.h
 * \brief
 */
#ifndef CLIP_BY_VALUE_SINGLE_DIM_TILING_H
#define CLIP_BY_VALUE_SINGLE_DIM_TILING_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ClipByValueSigDimTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, xDim);   // 算子级合轴后的
TILING_DATA_FIELD_DEF(int64_t, minDim); // 算子级合轴后的
TILING_DATA_FIELD_DEF(int64_t, maxDim); // 算子级合轴后的
TILING_DATA_FIELD_DEF(int64_t, yDim);   // 算子级合轴后的，单轴目标shape
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClipByValue_2000100, ClipByValueSigDimTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValue_2001100, ClipByValueSigDimTilingData);

REGISTER_TILING_DATA_CLASS(ClipByValueV2_2000100, ClipByValueSigDimTilingData);
REGISTER_TILING_DATA_CLASS(ClipByValueV2_2001100, ClipByValueSigDimTilingData);

class ClipByValueTilingSingleDim : public Ops::Math::OpTiling::TilingBaseClass {
public:
    explicit ClipByValueTilingSingleDim(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    const char* opName = "ClipByValue";
    uint32_t coreNum{0};     // syscfg
    uint64_t ubSize{0};      // syscfg
    uint64_t cacheline{128}; // syscfg Byte

    ge::DataType xDtype{ge::DataType::DT_FLOAT};
    uint32_t dTypeSize{1};

    bool isSigDim{false};
    int64_t xDim{1};   // 合轴后
    int64_t minDim{1}; // 合轴后
    int64_t maxDim{1}; // 合轴后
    int64_t yDim{1};   // 合轴后
    bool isAss{false};

    ClipByValueSigDimTilingData tilingData;
};

} // namespace optiling
#endif