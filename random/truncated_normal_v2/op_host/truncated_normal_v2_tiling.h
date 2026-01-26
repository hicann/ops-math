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
 * \file truncated_normal_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_Truncated_Normal_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_Truncated_Normal_V2_H_

#include <random>
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/truncated_normal_v2_tiling_data.h"

namespace optiling {
class TruncatedNormalV2Tiling : public Ops::Math::OpTiling::TilingBaseClass {
public:
    explicit TruncatedNormalV2Tiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    uint32_t maxThread_{512};
    uint64_t totalCoreNum_;
    uint64_t ubSize_;
    uint64_t workspaceSize_{0};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetOutputInfo();
    ge::graphStatus GetAttrInfo();
    template <typename T>
    ge::graphStatus GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape);

private:
    void SetTilingData();

private:
    int64_t seed{0};
    int64_t seed2{0};
    int64_t newSeed{0};
    int64_t newSeed2{0};
    int64_t outputNum{0};
    int64_t dtype{0};
    int64_t shapeSize{1};
    ge::DataType outDtype{ge::DT_FLOAT};

    const char* opName = "TruncatedNormalV2";
};

struct TruncatedNormalV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_
