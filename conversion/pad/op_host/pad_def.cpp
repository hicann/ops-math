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
 * \file pad_def.cpp
 * \brief Pad ophost
 */
#include "register/op_def_registry.h"
#include "common/inc/op_host/math_def_util.h"
#include <array>

namespace {
// x/y 支持的数据类型
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,        ge::DT_UINT8,         ge::DT_INT16,       ge::DT_UINT16,     ge::DT_INT32,
    ge::DT_UINT32,      ge::DT_INT64,         ge::DT_UINT64,      ge::DT_BF16,       ge::DT_FLOAT16,
    ge::DT_FLOAT,       ge::DT_DOUBLE,        ge::DT_BOOL,        ge::DT_HIFLOAT8,   ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
// paddings 支持的数据类型（paddings 输入按 INT64 + INT32 顺序与 x 两两组合）
static constexpr std::array PADDING_DATA_TYPE_ALL{ge::DT_INT64, ge::DT_INT32};
static constexpr auto DATA_TYPE_LIST = Ops::Math::CombineDataTypes(VALUE_DATA_TYPE_ALL, PADDING_DATA_TYPE_ALL);
static constexpr auto& VALUE_DATA_TYPE = std::get<0>(DATA_TYPE_LIST);
static constexpr auto& PAD_DATA_TYPE = std::get<1>(DATA_TYPE_LIST);
static const auto FORMAT_LIST = std::vector<ge::Format>(VALUE_DATA_TYPE.size(), ge::FORMAT_ND);
static const auto VALUE_DATA_TYPE_VEC = std::vector<ge::DataType>(VALUE_DATA_TYPE.begin(), VALUE_DATA_TYPE.end());
static const auto PAD_DATA_TYPE_VEC = std::vector<ge::DataType>(PAD_DATA_TYPE.begin(), PAD_DATA_TYPE.end());
} // namespace

namespace ops {
class Pad : public OpDef {
public:
    explicit Pad(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(VALUE_DATA_TYPE_VEC).Format(FORMAT_LIST);
        this->Input("paddings")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(PAD_DATA_TYPE_VEC)
            .Format(FORMAT_LIST);
        this->Output("y").ParamType(REQUIRED).DataType(VALUE_DATA_TYPE_VEC).Format(FORMAT_LIST);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "pad_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
        this->AICore().AddConfig("ascend350", aicore_config);
    }
};

OP_ADD(Pad);
} // namespace ops
