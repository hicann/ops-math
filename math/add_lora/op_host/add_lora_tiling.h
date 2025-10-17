/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_lora_tiling.h
 * \brief
 */
#ifndef ADD_LORA_TILING_H
#define ADD_LORA_TILING_H
#include "tiling_base/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {
enum class TilingKeyInfo : int
{
    KEY_DEFAULT_SCENE = 0,
    KEY_SPARSE_SCENE = 1,
    KEY_BGMV_SCENE = 10
};

enum class SocVersionKey : int
{
    KEY_SOC_VERSION_910 = 1,
    KEY_SOC_VERSION_310 = 2
};

BEGIN_TILING_DATA_DEF(AddLoraTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, layer);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, H1);
TILING_DATA_FIELD_DEF(uint32_t, H2);
TILING_DATA_FIELD_DEF(uint32_t, R);
TILING_DATA_FIELD_DEF(uint32_t, wBatch);
TILING_DATA_FIELD_DEF(uint32_t, layer_idx);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(uint32_t, y_offset);
TILING_DATA_FIELD_DEF(uint32_t, y_slice_size);
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, H2PerCore);
TILING_DATA_FIELD_DEF(uint32_t, addLoraFlag);
TILING_DATA_FIELD_DEF(uint32_t, y_column);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddLora, AddLoraTilingData)

class AddLoraTiling : public Ops::Math::OpTiling::TilingBaseClass {
public:
    explicit AddLoraTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~AddLoraTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    static constexpr uint32_t MAX_BATCH_SIZE = 65536;
    static constexpr uint32_t MAX_RANK_SIZE = 128;
    static constexpr uint32_t MAX_WEIGHT_NUM = 32;

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

    void Reset();

    void PrintTilingData();
    bool SetPlatformInfoForTiling();
    bool CheckTilingValidation();
    const gert::Shape GetShape(const size_t index);

    AddLoraTilingData tilingData_;
    bool compileInfoInit_ = false;
    bool addLoraFlag = 0;
    bool isAscend310P_;
    std::uint32_t coreNum_ = 1;
};

} // namespace optiling
#endif // ADD_LORA_TILING_H