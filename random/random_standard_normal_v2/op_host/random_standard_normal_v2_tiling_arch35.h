/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file random_standard_normal_v2_tiling_arch35.h
 * \brief
 */
#ifndef RANDOM_STANDARD_NORMAL_V2_TILING_ARCH35_H
#define RANDOM_STANDARD_NORMAL_V2_TILING_ARCH35_H

#include <string>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {
class RandomStandardNormalV2Tiling : public Ops::Math::OpTiling::TilingBaseClass
{
public:
    explicit RandomStandardNormalV2Tiling(gert::TilingContext* context)
        : TilingBaseClass(context), opName_(context->GetNodeName())
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

    void DumpTilingInfo() override;

private:
    const std::string opName_;
    int64_t ubSize_{0};
    int64_t totalCoreNum_{0};
    int64_t shapeSize_{1};
    uint32_t outputDtypeVal_{0};
    int64_t outputSize_{0};
    int64_t seed_{0};
    int64_t seed2_{0};
    ge::DataType shapeDtype_{ge::DT_INT32};
    ge::DataType outDtype_{ge::DT_FLOAT};
    int64_t blockFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t ubFactor_{0};
    int64_t tailUbFactor_{0};
    int64_t tailBlockTailUbFactor_{0};
    int64_t blockNum_{0};
    int64_t outputDtypeSize_{0};
    int64_t perCoreHandleRandom_{0};

    void SetTilingData() const;
    uint64_t New64() const;
    template <typename T>
    ge::graphStatus GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape);
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetOutputInfo();
    ge::graphStatus GetAttrInfo();
    void DoBlockTiling();
};

} // namespace optiling
#endif // Random_Standard_Normal_V2_TILING_ARCH35_H