/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confusion_matrix_def.cpp
 * \brief confusion_matrix def
 */

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> labelsPredictionsDtypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,  ge::DT_UINT8,
    ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8,   ge::DT_UINT8};

static const std::vector<ge::DataType> weightsOutputDtypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_INT32, ge::DT_INT8,    ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT,
    ge::DT_INT32,   ge::DT_INT8,    ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32,   ge::DT_INT8,
    ge::DT_UINT8,   ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32,   ge::DT_INT8,  ge::DT_UINT8,   ge::DT_FLOAT16,
    ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT8,  ge::DT_UINT8};

static const std::vector<ge::Format> kFormats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ConfusionMatrix : public OpDef {
public:
    explicit ConfusionMatrix(const char* name) : OpDef(name)
    {
        this->Input("labels")
            .ParamType(REQUIRED)
            .DataType(labelsPredictionsDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats);
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType(labelsPredictionsDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats);
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataType(weightsOutputDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(weightsOutputDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats);
        this->Attr("num_classes").AttrType(REQUIRED).Int();
        this->Attr("dtype").AttrType(REQUIRED).String("float32");
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "confusion_matrix");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(ConfusionMatrix);
} // namespace ops
