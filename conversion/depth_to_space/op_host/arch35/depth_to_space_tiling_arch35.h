/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file depth_to_space_tiling_arch35.h
 * \brief tiling for DepthToSpace
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DEPTH_TO_SPACE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DEPTH_TO_SPACE_H

#include "conversion/transpose/op_host/arch35/transpose_tiling_arch35.h"

namespace optiling {

namespace DepthToSpace {

BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TransposeOpTilingData, transposeOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData);

struct DepthToSpaceCompileInfo {
    TransposeCompilerInfo transposeCompilerInfo;
};

ge::graphStatus DepthToSpaceTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo);

class DepthToSpaceTiling {
public:
    explicit DepthToSpaceTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus ParametersVerifying();
    void ProcessShapeInfo(ShapeInfo& shapeInfo);

private:
    ParamInfo paramInfo_;
    gert::TilingContext* tilingContext_ = nullptr;
    int64_t nhwcDcrPerm_[DIM_SIX] = {0, 1, 3, 2, 4, 5};
    int64_t nchwDcrPerm_[DIM_SIX] = {0, 3, 4, 1, 5, 2};
    int64_t crdPerm_[DIM_SIX] = {0, 1, 4, 2, 5, 3};
};
}  // namespace DepthToSpace
}  // namespace optiling
#endif