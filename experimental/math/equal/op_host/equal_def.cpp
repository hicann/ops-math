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
 * \file equal_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class Equal : public OpDef {
public:
    explicit Equal(const char* name) : OpDef(name)
    {
        // Direct Ascend C kernel dtypes. COMPLEX64/COMPLEX128 keep using the ACLNN AICPU-TF fallback.
        const std::initializer_list<ge::DataType> aicoreInputDtypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8,
            ge::DT_INT16,   ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_BOOL};
        const std::initializer_list<ge::Format> aicoreFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        const std::initializer_list<ge::DataType> aicoreOutputDtypes = {
            ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
            ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL};

        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(aicoreInputDtypes)
            .Format(aicoreFormats)
            .UnknownShapeFormat(aicoreFormats);
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(aicoreInputDtypes)
            .Format(aicoreFormats)
            .UnknownShapeFormat(aicoreFormats);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(aicoreOutputDtypes)
            .Format(aicoreFormats)
            .UnknownShapeFormat(aicoreFormats);

        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(Equal); // 添加算子信息库
} // namespace ops
