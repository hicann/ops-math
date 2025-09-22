/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file is_finite_regbase_optiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_IS_FINITE_REGBASE_OPTILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_IS_FINITE_REGBASE_OPTILING_H

#include "elewise/elewise_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {
using namespace Ops::Base;

class IsFiniteRegbaseTiling
{
public:
    explicit IsFiniteRegbaseTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();

private:
    EleBaseTilingDataV2* tiling_ = nullptr;
    uint64_t dType = 0;
    gert::TilingContext* tilingContext;
    ge::DataType inputDtype;
    ge::DataType outputDtype;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_IS_FINITE_REGBASE_OPTILING_H