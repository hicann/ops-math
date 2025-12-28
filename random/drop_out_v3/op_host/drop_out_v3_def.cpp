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
 * \file drop_out_v3_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class DropOutV3 : public OpDef {
public:
    const std::vector<ge::DataType> inOutType = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

    const std::vector<ge::DataType> probType = {
        ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16};

    const std::vector<ge::DataType> seedType = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
                                                ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

    const std::vector<ge::DataType> offsetType = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                  ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                  ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                  ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

    const std::vector<ge::DataType> maskType = {ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
                                                ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
                                                ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
                                                ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8};

    const std::vector<ge::Format> baseFormat = {
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

    explicit DropOutV3(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({inOutType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});
        this->Input("noise_shape")
            .ParamType(OPTIONAL)
            .DataType({offsetType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});
        this->Input("p")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({probType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});
        this->Input("seed")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({seedType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});
        this->Input("offset")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({offsetType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({inOutType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});
        this->Output("mask")
            .ParamType(REQUIRED)
            .DataType({maskType})
            .Format({baseFormat})
            .UnknownShapeFormat({baseFormat});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910_95", aicoreConfig);
    }
};

OP_ADD(DropOutV3);
} // namespace ops