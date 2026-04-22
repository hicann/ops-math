/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file population_count_def.cpp
 * \brief PopulationCount operator definition
 *
 * PopulationCount counts the number of set bits (popcount) in each element:
 *   y[i] = popcount(x[i]), where y is UINT8 and x is INT16 or UINT16.
 *
 * Input:  x (Tensor, INT16 / UINT16)
 * Output: y (Tensor, UINT8, same shape as x)
 * Target: Ascend950 (arch35) only
 */
#include "register/op_def_registry.h"

namespace ops {
class PopulationCount : public OpDef {
public:
    explicit PopulationCount(const char* name) : OpDef(name)
    {
        // Input x: INT16 or UINT16; dtype index 0 -> INT16, index 1 -> UINT16
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_UINT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // Output y: UINT8 for both input dtypes
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Ascend950 (arch35) configuration only
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "population_count");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(PopulationCount);
} // namespace ops
