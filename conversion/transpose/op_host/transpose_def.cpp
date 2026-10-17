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
 * \file transpose_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
namespace {
// 计算各输入的 data type 组合
template <size_t DT1_CNT, size_t DT2_CNT>
static constexpr std::array<std::array<ge::DataType, DT1_CNT * DT2_CNT>, 2> CombineDataTypes(
    std::array<ge::DataType, DT1_CNT> dt1, std::array<ge::DataType, DT2_CNT> dt2)
{
    std::array<ge::DataType, DT1_CNT * DT2_CNT> combinedDT1{}, combinedDT2{};

    std::size_t idx = 0;
    for (std::size_t i2 = 0; i2 < dt2.size(); ++i2) {
        for (std::size_t i1 = 0; i1 < dt1.size(); ++i1) {
            combinedDT1[idx] = dt1[i1];
            combinedDT2[idx] = dt2[i2];
            ++idx;
        }
    }
    std::array res{combinedDT1, combinedDT2};
    return res;
}

// x data type 所有取值
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,   ge::DT_UINT8, ge::DT_INT16,    ge::DT_UINT16,      ge::DT_INT32,
    ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,   ge::DT_FLOAT,       ge::DT_FLOAT16,
    ge::DT_BF16,   ge::DT_BOOL,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
};
// 索引类型 data type 所有取值
static constexpr std::array IDX_DATA_TYPE_ALL{ge::DT_INT32, ge::DT_INT64};
static constexpr auto DATA_TYPE_LIST = CombineDataTypes(VALUE_DATA_TYPE_ALL, IDX_DATA_TYPE_ALL);
static const std::vector<ge::DataType> VALUE_DTS = std::vector<ge::DataType>(std::get<0>(DATA_TYPE_LIST).begin(),
                                                                             std::get<0>(DATA_TYPE_LIST).end());
static const std::vector<ge::DataType> PERM_DTS = std::vector<ge::DataType>(std::get<1>(DATA_TYPE_LIST).begin(),
                                                                            std::get<1>(DATA_TYPE_LIST).end());
static const auto DATA_FORMATS = std::vector<ge::Format>(VALUE_DATA_TYPE_ALL.size() * IDX_DATA_TYPE_ALL.size(),
                                                         ge::FORMAT_ND);

// arch92
static constexpr ge::DataType DT_HIFLOAT4 = static_cast<ge::DataType>(42);
static constexpr ge::DataType DT_HIFLOAT4_SCALE = static_cast<ge::DataType>(43);
// x data type 所有取值
static constexpr std::array VALUE_DATA_TYPE_ALL_ARCH92{
    ge::DT_INT8,        ge::DT_UINT8,       ge::DT_INT16,    ge::DT_UINT16,      ge::DT_INT32,
    ge::DT_UINT32,      ge::DT_INT64,       ge::DT_UINT64,   ge::DT_FLOAT,       ge::DT_FLOAT16,
    ge::DT_BF16,        ge::DT_BOOL,        ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, DT_HIFLOAT4,     DT_HIFLOAT4_SCALE,
};
// 索引类型 data type 所有取值
static constexpr std::array IDX_DATA_TYPE_ALL_ARCH92{ge::DT_INT32, ge::DT_INT64};
static constexpr auto DATA_TYPE_LIST_ARCH92 = CombineDataTypes(VALUE_DATA_TYPE_ALL_ARCH92, IDX_DATA_TYPE_ALL_ARCH92);
static const std::vector<ge::DataType> VALUE_DTS_ARCH92 = std::vector<ge::DataType>(
    std::get<0>(DATA_TYPE_LIST_ARCH92).begin(), std::get<0>(DATA_TYPE_LIST_ARCH92).end());
static const std::vector<ge::DataType> PERM_DTS_ARCH92 = std::vector<ge::DataType>(
    std::get<1>(DATA_TYPE_LIST_ARCH92).begin(), std::get<1>(DATA_TYPE_LIST_ARCH92).end());
static const auto DATA_FORMATS_ARCH92 = std::vector<ge::Format>(
    VALUE_DATA_TYPE_ALL_ARCH92.size() * IDX_DATA_TYPE_ALL_ARCH92.size(), ge::FORMAT_ND);
} // namespace

class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(VALUE_DTS).Format(DATA_FORMATS).UnknownShapeFormat(DATA_FORMATS);
        this->Input("perm")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(PERM_DTS)
            .Format(DATA_FORMATS)
            .UnknownShapeFormat(DATA_FORMATS);
        this->Output("y").ParamType(REQUIRED).DataType(VALUE_DTS).Format(DATA_FORMATS).UnknownShapeFormat(DATA_FORMATS);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "transpose_apt");

        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);

        auto config92 = buildConfig(VALUE_DTS_ARCH92, PERM_DTS_ARCH92, DATA_FORMATS_ARCH92);
        this->AICore().AddConfig("ascend960dt", config92);
        this->AICore().AddConfig("ascend960pr", config92);
    }

private:
    inline OpAICoreConfig buildConfig(const std::vector<ge::DataType> valueDTs, const std::vector<ge::DataType> permDTs,
                                      std::vector<ge::Format> dataFmtS)
    {
        OpAICoreConfig config;
        config.Input("x").ParamType(REQUIRED).DataType(valueDTs).Format(dataFmtS).UnknownShapeFormat(dataFmtS);
        config.Input("perm")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(permDTs)
            .Format(dataFmtS)
            .UnknownShapeFormat(dataFmtS);
        config.Output("y").ParamType(REQUIRED).DataType(valueDTs).Format(dataFmtS).UnknownShapeFormat(dataFmtS);
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "transpose_apt");
        return config;
    }
};

OP_ADD(Transpose);
} // namespace ops
