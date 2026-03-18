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
 * \file dynamic_stitch_def.cpp
 * \brief op config of dynamic_stitch
 */
#include <iostream>
#include <sstream>
#include "register/op_def_registry.h"

namespace ops {
constexpr int32_t AICORE_CHECK_LIST_CNT = 64;

static const std::vector<ge::DataType> xDType = {
    ge::DT_INT8,   ge::DT_UINT8, ge::DT_INT16,   ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
    ge::DT_UINT64, ge::DT_BOOL,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64};
static const std::vector<ge::DataType> indicesDType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
static const std::vector<ge::Format> xFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static ge::graphStatus CheckSupported(const ge::Operator& op, ge::AscendString& result) {
    int32_t attrN = 0;
    std::string resultJsonStr;
    ge::graphStatus retStatus = op.GetAttr("N", attrN);
    if (retStatus != ge::GRAPH_SUCCESS) {
        resultJsonStr = R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "GetAttr N error"})";
        result = ge::AscendString(resultJsonStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (attrN < 1 || attrN > AICORE_CHECK_LIST_CNT) {
        resultJsonStr = R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "attr out of range."})";
        result = ge::AscendString(resultJsonStr.c_str());
        return ge::GRAPH_FAILED;
    }
    resultJsonStr = R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "CheckSupported success."})";
    result = ge::AscendString(resultJsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

class DynamicStitch : public OpDef
{
public:
    explicit DynamicStitch(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(DYNAMIC)
            .DataType(indicesDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(xDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Attr("N").AttrType(OPTIONAL).Int(1);
        this->AICore().SetCheckSupport(CheckSupported);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "dynamic_stitch_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(DynamicStitch);
} // namespace ops