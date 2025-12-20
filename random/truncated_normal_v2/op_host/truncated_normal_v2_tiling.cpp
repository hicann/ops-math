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
 * \file truncated_normal_v2_tiling.cpp
 * \brief
 */
#include "truncated_normal_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "util/math_util.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_OFFSET = 1;
static constexpr uint16_t OUTPUT_IDX_SHAPE = 0;
static constexpr uint16_t INDEX_0 = 0;
static constexpr uint16_t INDEX_1 = 1;
static constexpr uint16_t INDEX_2 = 2;
static constexpr int64_t DEFAULT_VALUE = ge::DataType::DT_FLOAT;
static constexpr ge::DataType DEFAULT_TYPE_VALUE = ge::DataType::DT_FLOAT;
static constexpr uint32_t DCACHE_SIZE = 32U * 1024U;
static constexpr uint32_t RESERVED_WORKSPACE_SIZE = 16U * 1024U * 1024U;
static constexpr uint32_t TILING_KEY_BASE = 10000;
static constexpr int64_t THREAD_DISPOSAL_NUM = 4;
static const std::set<ge::DataType> TRUNCATED_NORMAL_V2_SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::unordered_map<int64_t, ge::DataType> TRUNCATED_NORMAL_V2_INT_TO_TYPE{
    {1, ge::DataType::DT_FLOAT}, {2, ge::DataType::DT_FLOAT16}, {3, ge::DataType::DT_BF16}};

template <typename T>
ge::graphStatus TruncatedNormalV2Tiling::GetIntValue(const gert::Tensor* constTensor, gert::Shape& constShape)
{
    const T* constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constValue);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    OP_LOGD(opName, "TruncatedNormalV2Tiling::GetIntValue constNum=%d", constNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::GetShapeAttrsInfo()
{
    OP_LOGI(opName, "TruncatedNormalV2Tiling::GetShapeAttrsInfo begin.");
    OP_CHECK_IF(GetInputInfo(), 
        OP_LOGE(opName, "GetInputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetOutputInfo(), 
        OP_LOGE(opName, "GetOutputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(shapeSize != outputNum,
        OP_LOGE(opName, "shape size: %ld is not equal to out size: %ld.", shapeSize, outputNum), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrInfo(), 
        OP_LOGE(opName, "GetAttrInfo failed!"), return ge::GRAPH_FAILED);
    OP_LOGI(opName, "TruncatedNormalV2Tiling::GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::GetInputInfo()
{
    auto shapeDesc = context_->GetInputDesc(INPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeDesc);
    ge::DataType shapeDtype = shapeDesc->GetDataType();
    OP_CHECK_IF(
        (shapeDtype != ge::DataType::DT_INT32) && (shapeDtype != ge::DataType::DT_INT64),
        OP_LOGE(
            opName, "input shape dtype should be int32, int64, but got %s.", Ops::Base::ToString(shapeDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto input1_shape = context_->GetInputShape(INPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1_shape);
    uint32_t shapeDimNum = input1_shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(shapeDimNum != 1,
        OP_LOGE(opName, "input shape is 1D tensor, but got %u.",
        shapeDimNum), return ge::GRAPH_FAILED);
    auto shapeTensor = context_->GetRequiredInputTensor(INPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeTensor);

    gert::Shape constShape;
    if (shapeDtype == ge::DataType::DT_INT32) {
        GetIntValue<int32_t>(shapeTensor, constShape);
    } else {
        GetIntValue<int64_t>(shapeTensor, constShape);
    }

    uint32_t shapeRank = constShape.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        shapeSize *= static_cast<int64_t>(constShape.GetDim(idx));
    }

    auto offsetDesc = context_->GetInputDesc(INPUT_IDX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
    auto offsetDtype = offsetDesc->GetDataType();
    OP_CHECK_IF(
        offsetDtype != ge::DataType::DT_INT64,
        OP_LOGE(opName, "input offset Dtype should be int64, but got %s.", Ops::Base::ToString(offsetDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto offsetTensor = context_->GetInputTensor(INPUT_IDX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetTensor);
    auto offsetTensorSize = static_cast<int64_t>(offsetTensor->GetShapeSize());
    OP_CHECK_IF(
        offsetTensorSize != 1, OP_LOGE(opName, "input offset shape_size should be 1, but got %ld.", offsetTensorSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::GetOutputInfo()
{
    auto outDesc = context_->GetOutputDesc(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    outDtype = outDesc->GetDataType();
    OP_CHECK_IF(
        TRUNCATED_NORMAL_V2_SUPPORT_DTYPE.count(outDtype) == 0,
        OP_LOGE(
            opName, "out shape dtype should be float, float16, bfloat16 but got %s.",
            Ops::Base::ToString(outDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outTensor = outputShape->GetStorageShape();
    outputNum = outTensor.GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* seedPtr = attrs->GetAttrPointer<int64_t>(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedPtr);
    newSeed = *seedPtr;

    const int64_t* seed2Ptr = attrs->GetAttrPointer<int64_t>(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seed2Ptr);
    newSeed2 = *seed2Ptr;

    // 查看newSeed和newSeed2是否同时为0
    if (newSeed == static_cast<int64_t>(0) && newSeed2 == static_cast<int64_t>(0)) {
        newSeed = static_cast<int64_t>(New64());
        newSeed2 = static_cast<int64_t>(New64());
    }

    const int64_t* dtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dtypePtr);
     ge::DataType yDtype = static_cast<ge::DataType>(*dtypePtr);
    OP_LOGD(opName, "attrDtype = [%d, %s]", static_cast<int32_t>(yDtype), Ops::Base::ToString(yDtype).c_str());
    OP_CHECK_IF(
        yDtype != outDtype,
        OP_LOGE(
            opName, "attrDtype is %s, is not equal to out shape dtype %s.", Ops::Base::ToString(yDtype).c_str(),
            Ops::Base::ToString(outDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (aivNum <= 0), OP_LOGE(opName, "ScatterNdUpdateTiling fail to get totalCoreNum_."), return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(
        (ubSizePlatForm <= DCACHE_SIZE), OP_LOGE(opName, "ub size less than Dcache Size. please check"),
        return ge::GRAPH_FAILED);
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ = ubSizePlatForm - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS),
        OP_LOGE(opName, "SetLocalMemorySize ubSize = %ld failed.", static_cast<int64_t>(ubSize_)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool TruncatedNormalV2Tiling::IsCapable()
{
    return true;
}

ge::graphStatus TruncatedNormalV2Tiling::DoOpTiling()
{
    // 一个线程处理4个数
    int64_t threadNum = Ops::Base::CeilAlign(outputNum, THREAD_DISPOSAL_NUM);
    uint64_t coreNum = Ops::Base::CeilAlign(threadNum, static_cast<int64_t>(maxThread_));
    totalCoreNum_ = std::min(coreNum, totalCoreNum_);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncatedNormalV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = RESERVED_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

uint64_t TruncatedNormalV2Tiling::GetTilingKey() const
{
    uint64_t tilingKey = static_cast<uint64_t>(1);

    OP_LOGD(opName, "tilingKey = %ld.", static_cast<int64_t>(tilingKey));
    return tilingKey;
}

void TruncatedNormalV2Tiling::SetTilingData()
{
    TruncatedNormalV2TilingData* tilingData = context_->GetTilingData<TruncatedNormalV2TilingData>();

    tilingData->newSeed = newSeed;
    tilingData->newSeed2 = newSeed2;
    tilingData->outputNum = outputNum;
}

ge::graphStatus TruncatedNormalV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(totalCoreNum_);
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4TruncatedNormalV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4TruncatedNormalV2 running TruncatedNormalV2 tiling.");
    TruncatedNormalV2Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4TruncatedNormalV2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<TruncatedNormalV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4DiagV2 fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4DiagFlat fail to get ub size."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncatedNormalV2)
    .Tiling(Tiling4TruncatedNormalV2)
    .TilingParse<TruncatedNormalV2CompileInfo>(TilingPrepare4TruncatedNormalV2)
    .TilingInputsDataDependency({0});
} // namespace optiling