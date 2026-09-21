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
 * \file fill_v2_tiling_arch35.cpp
 * \brief
 */
#include <cmath>
#include <limits>
#include <string>
#include "fill_v2_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "util/fp16.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "../../op_kernel/arch35/fill_v2_dag.h"
#include "../../op_kernel/arch35/fill_v2_tiling_key.h"
#include "../../op_kernel/arch35/fill_v2_tilingdata.h"

using namespace ge;
using namespace FillV2Op;
using namespace Ops::Base;

namespace optiling {
constexpr uint64_t FILL_V2_WORKSPACE_RESERVE_BYTE = 16777216; // 16 * 1024 * 1024
const std::string FILL_V2_TILING_OP_NAME = "FillV2Tiling";
constexpr int64_t MAX_DIM_NUM = 8;

union FillV2Value {
    int8_t vI8;
    int16_t vI16;
    int32_t vI32;
    int64_t vI64;
    float vF32;
    double vF64;
    Ops::Base::fp16_t vF16;

    explicit FillV2Value(int64_t v) : vI64(v) {}
};

ge::graphStatus FillV2CheckType(ge::DataType dtype, const std::initializer_list<ge::DataType>& supportList)
{
    for (auto supportDtype : supportList) {
        if (dtype == supportDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

// 按dims自身dtype逐维读取数值校验非负：dims原始值超出dtype表示范围时存储回绕为负数，
// golden侧报"trying to create tensor with negative dimension"，tiling侧以相同口径拦截
template <typename T>
static ge::graphStatus FillV2CheckDimsValueNonNegative(const char_t* nodeName, const gert::Tensor* tensorDims)
{
    const T* dimsData = tensorDims->GetData<T>();
    // dims数值不可见(如非常量输入且未下发数据依赖)时无法校验，跳过以对齐infershape的兜底行为
    if (dimsData == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t dimNum = tensorDims->GetShapeSize();
    for (int64_t i = 0; i < dimNum; ++i) {
        if (dimsData[i] < 0) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                nodeName, "dims(input)", std::to_string(dimsData[i]),
                "Dim values in dims must be non-negative. A negative value usually means the original dim "
                "exceeds the representable range of the dims dtype and wraps around");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillV2Tiling::SetTilingData(const ElewiseBaseTiling& elewiseBaseTiling)
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<uint64_t>(FILL_V2_WORKSPACE_RESERVE_BYTE);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(dType);
    OP_LOGD(FILL_V2_TILING_OP_NAME, "[TilingData] : tilingKey=%lu", tilingKey);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillV2Tiling::CheckInputDims()
{
    auto dimsStorageShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimsStorageShape);
    auto dimsShape = Ops::Base::EnsureNotScalar(dimsStorageShape->GetStorageShape());
    if (dimsShape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dims(input)", Ops::Base::ToString(dimsShape),
                                              "The shape of dims must be 1D");
        return ge::GRAPH_FAILED;
    }
    if (dimsShape.GetDim(0) > MAX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dims(input)", Ops::Base::ToString(dimsShape),
                                              "The dims size must be less than or equal to 8");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor* tensorDims = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorDims);
    int64_t dimNum = tensorDims->GetShapeSize();
    if (dimNum < 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "dims(input)", std::to_string(dimNum),
                                                  "The shape size of dims cannot be negative");
        return ge::GRAPH_FAILED;
    }
    auto dimsDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimsDesc);
    ge::DataType inputDimsDType = dimsDesc->GetDataType();
    if (inputDimsDType != ge::DT_INT16 && inputDimsDType != ge::DT_INT32 && inputDimsDType != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dims(input)", Ops::Base::ToString(inputDimsDType),
                                  "Int16, Int32 or Int64");
        return ge::GRAPH_FAILED;
    }
    switch (inputDimsDType) {
        case ge::DT_INT16:
            return FillV2CheckDimsValueNonNegative<int16_t>(context_->GetNodeName(), tensorDims);
        case ge::DT_INT32:
            return FillV2CheckDimsValueNonNegative<int32_t>(context_->GetNodeName(), tensorDims);
        case ge::DT_INT64:
            return FillV2CheckDimsValueNonNegative<int64_t>(context_->GetNodeName(), tensorDims);
        default:
            return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus FillV2Tiling::CalcOutputDtype()
{
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    this->outputDtype_ = outputDesc->GetDataType();

    static const std::initializer_list<ge::DataType> SUPPORT_LIST = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64};
    if (FillV2CheckType(this->outputDtype_, SUPPORT_LIST) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y(output)", Ops::Base::ToString(this->outputDtype_),
                                  "Float16, Float, Double, Int8, Int16, Int32 and Int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillV2Tiling::SetAttr()
{
    auto tiling = context_->GetTilingData<FillV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* valueAttr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueAttr);
    float value = *valueAttr;
    FillV2Value v(0);
    switch (this->outputDtype_) {
        case ge::DT_FLOAT16:
            // 对齐 torch.full：有限值超出 fp16 最大有限值 65504 时报错；inf/nan 原样填充
            if (std::isfinite(value) && std::fabs(value) > 65504.0f) {
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "value", std::to_string(value).c_str(),
                                          "in range [-65504, 65504] when y is float16");
                return ge::GRAPH_FAILED;
            }
            if (std::isnan(value)) {
                v.vF16 = Ops::Base::fp16_t(static_cast<uint16_t>(0x7E00u));
            } else if (std::isinf(value)) {
                v.vF16 = Ops::Base::fp16_t(static_cast<uint16_t>(value > 0 ? 0x7C00u : 0xFC00u));
            } else {
                v.vF16 = Ops::Base::fp16_t(value);
            }
            break;
        case ge::DT_FLOAT:
            v.vF32 = value;
            break;
        case ge::DT_DOUBLE:
            v.vF64 = static_cast<double>(value);
            break;
        case ge::DT_INT8:
            if (!(value >= -128.0f && value <= 127.0f)) {
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "value", std::to_string(value).c_str(),
                                          "in range [-128, 127] when y is int8");
                return ge::GRAPH_FAILED;
            }
            v.vI8 = static_cast<int8_t>(value);
            break;
        case ge::DT_INT16:
            if (!(value >= -32768.0f && value <= 32767.0f)) {
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "value", std::to_string(value).c_str(),
                                          "in range [-32768, 32767] when y is int16");
                return ge::GRAPH_FAILED;
            }
            v.vI16 = static_cast<int16_t>(value);
            break;
        case ge::DT_INT32:
            // INT32_MAX(2147483647) 在 fp32 不可精确表示，fp32 值不超过 INT32_MAX 等价于严格小于 2^31
            if (!(value >= -2147483648.0f && value < 2147483648.0f)) {
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "value", std::to_string(value).c_str(),
                                          "in range [-2147483648, 2147483647] when y is int32");
                return ge::GRAPH_FAILED;
            }
            v.vI32 = static_cast<int32_t>(value);
            break;
        case ge::DT_INT64:
            // INT64_MAX 在 fp32 不可精确表示，fp32 域上边界取 2^63：
            // 恰为 2^63 时与 torch.full 一致饱和为 INT64_MAX，超过 2^63 报错；-2^63 恰为 INT64_MIN 可精确表示
            if (!(value >= -9223372036854775808.0f && value <= 9223372036854775808.0f)) {
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "value", std::to_string(value).c_str(),
                                          "in range [-9223372036854775808, 9223372036854775807] when y is int64");
                return ge::GRAPH_FAILED;
            }
            if (value == 9223372036854775808.0f) {
                v.vI64 = std::numeric_limits<int64_t>::max();
            } else {
                v.vI64 = static_cast<int64_t>(value);
            }
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y(output)", Ops::Base::ToString(this->outputDtype_),
                                      "Float16, Float, Double, Int8, Int16, Int32 and Int64");
            return ge::GRAPH_FAILED;
    }
    tiling->value = v.vI64;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillV2Tiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(context_);
    OP_CHECK_IF(CheckInputDims() == ge::GRAPH_FAILED, OP_LOGE(context_, "check dims failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(context_, "get output dtype failed"),
                return ge::GRAPH_FAILED);

    auto tiling = context_->GetTilingData<FillV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);

    OP_CHECK_IF(SetAttr() == ge::GRAPH_FAILED, OP_LOGE(context_, "set Attr failed"), return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype_ == ge::DT_FLOAT16) {
        dType = FILL_V2_TPL_FP16;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<half>::OpDag, false>(tiling->baseTiling);
    } else if (this->outputDtype_ == ge::DT_FLOAT) {
        dType = FILL_V2_TPL_FP32;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<float>::OpDag, false>(tiling->baseTiling);
    } else if (this->outputDtype_ == ge::DT_DOUBLE) {
        dType = FILL_V2_TPL_DOUBLE;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<int64_t>::OpDag, false>(tiling->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT8) {
        dType = FILL_V2_TPL_INT8;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<int8_t>::OpDag, false>(tiling->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT16) {
        dType = FILL_V2_TPL_INT16;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<int16_t>::OpDag, false>(tiling->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT32) {
        dType = FILL_V2_TPL_INT32;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<int32_t>::OpDag, false>(tiling->baseTiling);
    } else if (this->outputDtype_ == ge::DT_INT64) {
        dType = FILL_V2_TPL_INT64;
        res = elewiseBaseTiling.DoTiling<FillV2DAG<int64_t>::OpDag, false>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y(output)", Ops::Base::ToString(this->outputDtype_),
                                  "Float16, Float, Double, Int8, Int16, Int32 and Int64");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData(elewiseBaseTiling);
    return result;
}

static ge::graphStatus Tiling4FillV2(gert::TilingContext* context)
{
    OP_LOGD(FILL_V2_TILING_OP_NAME, "Enter Tiling4FillV2");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const FillV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    FillV2Tiling fillV2Tiling(context);
    return fillV2Tiling.RunTiling();
}

ge::graphStatus TilingPrepareForFillV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<FillV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FillV2)
    .Tiling(Tiling4FillV2)
    .TilingParse<FillV2CompileInfo>(TilingPrepareForFillV2)
    .InputsDataDependency({0});
} // namespace optiling
