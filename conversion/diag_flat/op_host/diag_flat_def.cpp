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
 * \file diag_flat_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class DiagFlat : public OpDef {
public:
    DiagFlat(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
                 ge::DT_INT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8, ge::DT_COMPLEX64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
                 ge::DT_INT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8, ge::DT_COMPLEX64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND});
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config_310p_910;
        config_310p_910.Input("x")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT8,
                 ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8, ge::DT_COMPLEX64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_310p_910.Output("y")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT8,
                 ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8, ge::DT_COMPLEX64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_310p_910.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910", config_310p_910);
        this->AICore().AddConfig("ascend310p", config_310p_910);
    }
};

OP_ADD(DiagFlat);
} // namespace ops
