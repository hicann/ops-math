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
 * \file fill_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class FillV2 : public OpDef {
public:
    const std::vector<ge::DataType> dimsDataType = {
        ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
    const std::vector<ge::DataType> yDataType = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64};
    const std::vector<ge::Format> ndFormat = {
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit FillV2(const char* name) : OpDef(name)
    {
        this->Input("dims")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(dimsDataType)
            .Format(ndFormat)
            .UnknownShapeFormat(ndFormat);
        this->Output("y").ParamType(REQUIRED).DataType(yDataType).Format(ndFormat).UnknownShapeFormat(ndFormat);
        this->Attr("value").AttrType(OPTIONAL).Float(0.0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "fill_v2_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};

OP_ADD(FillV2);
} // namespace ops
