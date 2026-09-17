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
 * \file add_v2_tiling_arch35.cpp
 * \brief add_v2 tiling for ascend950 (arch35)
 */

#include <cstddef>
#include <string>
#include <type_traits>

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/add_v2_dag.h"
#include "../../op_kernel/arch35/add_v2_struct_arch35.h"
#include "add_v2_tiling_arch35.h"

using namespace ge;
using namespace AddV2Op;
using namespace Ops::Base;

namespace optiling {

// 自定义模板分支：空 Tensor 走 schMode 999 + userDef 1，见 add_v2_struct_arch35.h
constexpr uint64_t ADD_V2_SCH_MODE_EMPTY = 999;
constexpr uint64_t ADD_V2_USER_DEF_NORMAL = 0;
constexpr uint64_t ADD_V2_USER_DEF_EMPTY = 1;
constexpr size_t ADD_V2_MIN_RANK = 1;
constexpr size_t ADD_V2_MAX_RANK = 8;

static_assert(std::is_standard_layout<AddV2CompileInfoArch35>::value,
              "AddV2CompileInfoArch35 must remain standard-layout");
static_assert(offsetof(AddV2CompileInfoArch35, isAscendC) == offsetof(BroadcastCompileInfo, isAscendC),
              "AddV2 compile-info isAscendC offset must match BroadcastCompileInfo");
static_assert(offsetof(AddV2CompileInfoArch35, coreNum) == offsetof(BroadcastCompileInfo, coreNum),
              "AddV2 compile-info coreNum offset must match BroadcastCompileInfo");
static_assert(offsetof(AddV2CompileInfoArch35, ubSize) == offsetof(BroadcastCompileInfo, ubSize),
              "AddV2 compile-info ubSize offset must match BroadcastCompileInfo");
static_assert(sizeof(AddV2CompileInfoArch35) >= sizeof(BroadcastCompileInfo),
              "AddV2 compile-info must contain the BroadcastCompileInfo prefix");

static bool IsSupportedDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT64 ||
           dtype == ge::DT_COMPLEX64 || dtype == ge::DT_UINT8 || dtype == ge::DT_INT8 || dtype == ge::DT_INT32 ||
           dtype == ge::DT_INT16;
}

static ge::graphStatus CheckRank(const gert::TilingContext* context, const char* tensorName, const char* shapeKind,
                                 const gert::Shape& shape)
{
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(
        rank < ADD_V2_MIN_RANK || rank > ADD_V2_MAX_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), tensorName, std::to_string(rank).c_str(),
                                                 (std::string(shapeKind) + " rank must be within [1, 8]").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckConcreteShape(const gert::TilingContext* context, const char* tensorName,
                                          const gert::Shape& shape)
{
    if (CheckRank(context, tensorName, "storage", shape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool hasZeroDim = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), tensorName, std::to_string(dim).c_str(),
                        ("storage dim[" + std::to_string(i) + "] must be non-negative").c_str()),
                    return ge::GRAPH_FAILED);
        hasZeroDim = hasZeroDim || dim == 0;
    }

    // A zero-sized shape has numel 0 regardless of the other dimensions. Avoid
    // GetShapeSize()'s left-to-right multiplication reporting a false overflow
    // when a zero dimension appears after very large dimensions.
    OP_CHECK_IF(!hasZeroDim && shape.GetShapeSize() < 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), tensorName,
                                                          Ops::Base::ToString(shape).c_str(),
                                                          "shape element count overflows int64_t"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool HasZeroDim(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckBroadcastResult(const gert::TilingContext* context, const gert::Shape& x1Shape,
                                            const gert::Shape& x2Shape, const gert::Shape& yShape)
{
    const size_t x1Rank = x1Shape.GetDimNum();
    const size_t x2Rank = x2Shape.GetDimNum();
    const size_t expectedRank = x1Rank > x2Rank ? x1Rank : x2Rank;
    OP_CHECK_IF(yShape.GetDimNum() != expectedRank,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "y", std::to_string(yShape.GetDimNum()).c_str(),
                    ("rank must match broadcast rank " + std::to_string(expectedRank)).c_str()),
                return ge::GRAPH_FAILED);

    const size_t x1Offset = expectedRank - x1Rank;
    const size_t x2Offset = expectedRank - x2Rank;
    for (size_t i = 0; i < expectedRank; ++i) {
        const int64_t x1Dim = i < x1Offset ? 1 : x1Shape.GetDim(i - x1Offset);
        const int64_t x2Dim = i < x2Offset ? 1 : x2Shape.GetDim(i - x2Offset);
        OP_CHECK_IF(
            x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x1 and x2", (std::to_string(x1Dim) + " and " + std::to_string(x2Dim)).c_str(),
                ("dimensions cannot broadcast at output dim " + std::to_string(i)).c_str()),
            return ge::GRAPH_FAILED);

        // The non-one dimension is the result. This deliberately preserves 0
        // for the valid 0-vs-1 empty-tensor broadcast case.
        const int64_t expectedDim = x1Dim == 1 ? x2Dim : x1Dim;
        OP_CHECK_IF(
            yShape.GetDim(i) != expectedDim,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "y", std::to_string(yShape.GetDim(i)).c_str(),
                ("dim[" + std::to_string(i) + "] must equal broadcast result " + std::to_string(expectedDim)).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

class AddV2TilingArch35 {
public:
    explicit AddV2TilingArch35(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcDtype();
    ge::graphStatus CheckShape() const;
    ge::graphStatus CheckDtype() const;
    ge::graphStatus SetWorkspace() const;
    ge::graphStatus HandleEmptyTensor() const;

private:
    ge::DataType inputDtype_ = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext_;
};

ge::graphStatus AddV2TilingArch35::SetWorkspace() const
{
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    auto compileInfo = tilingContext_->GetCompileInfo<AddV2CompileInfoArch35>();
    if (compileInfo != nullptr && compileInfo->libApiWorkspaceSize > 0) {
        currentWorkspace[0] = static_cast<size_t>(compileInfo->libApiWorkspaceSize);
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfo = tilingContext_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    currentWorkspace[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

// 空 Tensor 早返回。ATVOSS 的 BroadcastBaseTiling 在合轴之后会显式拒绝 0 元素
// （broadcast_tiling.h: "tensor check is empty, check failed"），不能落到 DoTiling，
// 因此这里自己出一份 tiling：blockDim = 1，tilingKey 选自定义分支，kernel 侧直接返回。
ge::graphStatus AddV2TilingArch35::HandleEmptyTensor() const
{
    OP_LOGD(tilingContext_, "AddV2: empty tensor, skip kernel computation.");
    OP_CHECK_IF(SetWorkspace() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "workspace", "unavailable",
                                                      "set workspace failed for empty tensor"),
                return ge::GRAPH_FAILED);

    auto* tilingData = tilingContext_->GetTilingData<AddV2EmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tilingData);
    tilingData->numel = 0;

    tilingContext_->SetBlockDim(1);
    tilingContext_->SetTilingKey(GET_TPL_TILING_KEY(ADD_V2_SCH_MODE_EMPTY, ADD_V2_USER_DEF_EMPTY));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddV2TilingArch35::CalcDtype()
{
    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    this->inputDtype_ = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddV2TilingArch35::CheckShape() const
{
    auto inputX1 = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputX1);
    auto inputX2 = tilingContext_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputX2);
    auto outputY = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputY);

    if (CheckRank(tilingContext_, "x1", "origin", inputX1->GetOriginShape()) != ge::GRAPH_SUCCESS ||
        CheckRank(tilingContext_, "x2", "origin", inputX2->GetOriginShape()) != ge::GRAPH_SUCCESS ||
        CheckRank(tilingContext_, "y", "origin", outputY->GetOriginShape()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& x1Shape = inputX1->GetStorageShape();
    const gert::Shape& x2Shape = inputX2->GetStorageShape();
    const gert::Shape& yShape = outputY->GetStorageShape();
    if (CheckConcreteShape(tilingContext_, "x1", x1Shape) != ge::GRAPH_SUCCESS ||
        CheckConcreteShape(tilingContext_, "x2", x2Shape) != ge::GRAPH_SUCCESS ||
        CheckConcreteShape(tilingContext_, "y", yShape) != ge::GRAPH_SUCCESS ||
        CheckBroadcastResult(tilingContext_, x1Shape, x2Shape, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 仅注册同 dtype 组合（canonical AddV2 Verifier 亦要求 x1/x2 同 dtype）
ge::graphStatus AddV2TilingArch35::CheckDtype() const
{
    auto input1Desc = tilingContext_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, input1Desc);
    auto outputDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputDesc);
    const ge::DataType input1Dtype = input1Desc->GetDataType();
    const ge::DataType outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        !IsSupportedDtype(this->inputDtype_) || !IsSupportedDtype(input1Dtype) || !IsSupportedDtype(outputDtype),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext_->GetNodeName(), "x1, x2 and y",
                                               (ge::TypeUtils::DataTypeToSerialString(this->inputDtype_) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + " and " +
                                                ge::TypeUtils::DataTypeToSerialString(outputDtype))
                                                   .c_str(),
                                               "x1, x2 and y dtype must be supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(input1Dtype != this->inputDtype_ || outputDtype != this->inputDtype_,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext_->GetNodeName(), "x1, x2 and y",
                                                       (ge::TypeUtils::DataTypeToSerialString(this->inputDtype_) +
                                                        ", " + ge::TypeUtils::DataTypeToSerialString(input1Dtype) +
                                                        " and " + ge::TypeUtils::DataTypeToSerialString(outputDtype))
                                                           .c_str(),
                                                       "x1, x2 and y must have the same dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddV2TilingArch35::RunTiling()
{
    OP_CHECK_IF(tilingContext_ == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("AddV2", "tiling context", "must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_->GetNodeName(), "Enter AddV2TilingArch35::RunTiling.");
    OP_CHECK_IF(
        CalcDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext_->GetNodeName(), "x1", "unknown", "calc dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "input shape", "invalid",
                                                      "input shape check failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "input dtype", "invalid",
                                                      "input dtype check failed"),
                return ge::GRAPH_FAILED);

    // Shape and dtype validation must happen before this early return. Otherwise
    // a forged empty y could bypass invalid broadcast/rank/dim/dtype checks.
    auto outputY = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputY);
    if (HasZeroDim(outputY->GetStorageShape())) {
        return HandleEmptyTensor();
    }

    ge::graphStatus ret = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    if (this->inputDtype_ == ge::DT_FLOAT16) {
        BroadcastBaseTiling<AddWithCastCompute<half>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_BF16) {
        BroadcastBaseTiling<AddWithCastCompute<bfloat16_t>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_FLOAT) {
        BroadcastBaseTiling<AddWithCastCompute<float>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_INT64 || this->inputDtype_ == ge::DT_COMPLEX64) {
        BroadcastBaseTiling<AddWithoutCastCompute<int64_t>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_UINT8) {
        BroadcastBaseTiling<AddWithoutCastCompute<uint8_t>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_INT8) {
        BroadcastBaseTiling<AddWithoutCastCompute<int8_t>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_INT32) {
        BroadcastBaseTiling<AddWithoutCastCompute<int32_t>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype_ == ge::DT_INT16) {
        BroadcastBaseTiling<AddWithoutCastCompute<int16_t>::OpDag> brcBaseTiling(tilingContext_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str(),
            "supported dtypes are fp16, bf16, fp32, int64, int32, int16, uint8, int8 and complex64");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "tiling", "failed",
                                                      "broadcastBaseTiling failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspace() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "workspace", "unavailable",
                                                      "set workspace failed"),
                return ge::GRAPH_FAILED);

    OP_LOGD(tilingContext_, "[TilingData] : tilingKey=%llu", static_cast<unsigned long long>(tilingKey));
    tilingContext_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AddV2Arch35(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("AddV2", "tiling_context", "must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter Tiling4AddV2Arch35");
    AddV2TilingArch35 addV2Tiling(context);
    return addV2Tiling.RunTiling();
}

static ge::graphStatus TilingPrepare4AddV2Arch35(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("AddV2", "tiling_parse_context", "must not be nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<AddV2CompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->isAscendC = true;
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    compileInfo->libApiWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize == 0 || compileInfo->libApiWorkspaceSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum,ubSize,libApiWorkspaceSize",
                                                      std::to_string(compileInfo->coreNum) + ", " +
                                                          std::to_string(compileInfo->ubSize) + ", " +
                                                          std::to_string(compileInfo->libApiWorkspaceSize),
                                                      "The platform resource values must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddV2).Tiling(Tiling4AddV2Arch35).TilingParse<AddV2CompileInfoArch35>(TilingPrepare4AddV2Arch35);
} // namespace optiling
