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
 * \file random_uniform_v2_tiling_arch35.cpp
 * \brief
 */
#include <random>
#include <iostream>
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "random_uniform_v2_tiling.h"
#include "random_uniform_v2_tiling_arch35.h"
#include  "../../common/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t IN_SHAPE_IDX = 0;
static constexpr uint16_t IN_OFFSET_IDX = 1;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t ATTR_TYPE_IDX = 0;
static constexpr uint16_t ATTR_SEED_IDX = 1;
static constexpr uint16_t ATTR_SEED2_IDX = 2;
static constexpr int64_t CORE_ALIGN_SIZE = 512;
static constexpr uint32_t MIN_TILING_SIZE = 256;
static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);
static constexpr int64_t DOUBLE_BUFFER = 2;

static const std::set<ge::DataType> RANDOM_STANDARD_NORMAL_V2_SUPPORT_DTYPE = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16};

template <typename T>
ge::graphStatus RandomUniformV2Tiling::GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape)
{
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetIntValue begin.");
    const T *constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constValue);
    OP_LOGD(opName_, "RandomUniformV2Tiling::GetIntValue constValue=%d.", constValue[0]);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetIntValue constNum=%d", constNum);
    return ge::GRAPH_SUCCESS;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus RandomUniformV2Tiling::GetPlatformInfo()
{
    OP_LOGI(opName_, "RandomUniformV2Tiling GetPlatformInfo.");
    auto compileInfo = static_cast<const RandomUniformV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetPlatformInfo ubSize_=%d, totalCoreNum_=%d", ubSize_, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus RandomUniformV2Tiling::GetShapeAttrsInfo()
{
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetShapeAttrsInfo begin.");
    OP_CHECK_IF(GetInputInfo(), 
        OP_LOGE(opName_, "GetInputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetOutputInfo(), 
        OP_LOGE(opName_, "GetOutputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(shapeSize_ != outputSize_,
        OP_LOGE(opName_, "shape size: %ld is not equal to out size: %ld.", shapeSize_, outputSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrInfo(), 
        OP_LOGE(opName_, "GetAttrInfo failed!"), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformV2Tiling::GetInputInfo()
{
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetInputInfo begin.");
    auto shapeDesc = context_->GetRequiredInputDesc(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeDesc);
    auto shapeDtype = shapeDesc->GetDataType();
    OP_CHECK_IF((shapeDtype != ge::DataType::DT_INT32) && (shapeDtype != ge::DataType::DT_INT64),
        OP_LOGE(opName_, "input shape dtype should be int32, int64, but got %s.",
        Ops::Base::ToString(shapeDtype).c_str()), return ge::GRAPH_FAILED);

    auto input1Shape = context_->GetInputShape(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Shape);
    uint32_t shapeDimNum = input1Shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(shapeDimNum != 1,
        OP_LOGE(opName_, "input shape is 1D tensor, but got %u.",
        shapeDimNum), return ge::GRAPH_FAILED);

    auto shapeTensor = context_->GetRequiredInputTensor(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeTensor);
    
    gert::Shape constShape;
    if (shapeDtype == ge::DataType::DT_INT32) {
        GetIntValue<int32_t>(shapeTensor, constShape);
    } else {
        GetIntValue<int64_t>(shapeTensor, constShape);
    }
    OP_LOGD(opName_, "RandomUniformV2Tiling::GetInputInfo get shapeTensor end.");

    uint32_t shapeRank = constShape.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        shapeSize_ *= static_cast<int64_t>(constShape.GetDim(idx));
    }

    auto offsetDesc = context_->GetInputDesc(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
    auto offsetDtype = offsetDesc->GetDataType();
    OP_CHECK_IF(offsetDtype != ge::DataType::DT_INT64,
        OP_LOGE(opName_, "input offset Dtype should be int64, but got %s.",
        Ops::Base::ToString(offsetDtype).c_str()), return ge::GRAPH_FAILED);
        
    auto offsetTensor = context_->GetInputTensor(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetTensor);
    auto offsetTensorSize = static_cast<int64_t>(offsetTensor->GetShapeSize());
    OP_CHECK_IF(offsetTensorSize != 1,
        OP_LOGE(opName_, "input offset shape_size should be 1, but got %ld.", offsetTensorSize), return ge::GRAPH_FAILED);

    OP_LOGI(opName_, "RandomUniformV2Tiling::GetInputInfo end.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformV2Tiling::GetOutputInfo()
{
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetOutputInfo begin.");
    auto outDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    outDtype_ = outDesc->GetDataType();
    OP_CHECK_IF(RANDOM_STANDARD_NORMAL_V2_SUPPORT_DTYPE.count(outDtype_) == 0,
        OP_LOGE(opName_, "out shape dtype should be float, float16, bfloat16 but got %s.",
        Ops::Base::ToString(outDtype_).c_str()), return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outTensor = outputShape->GetStorageShape();
    outputSize_ = outTensor.GetShapeSize();
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetOutputInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformV2Tiling::GetAttrInfo()
{
    OP_LOGI(opName_, "RandomUniformV2Tiling::GetAttrInfo begin.");
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto attrDtype = attrs->GetAttrPointer<int64_t>(ATTR_TYPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDtype);
    const auto* seedAttr = attrs->GetAttrPointer<int64_t>(ATTR_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedAttr);
    const auto* seed2Attr = attrs->GetAttrPointer<int64_t>(ATTR_SEED2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seed2Attr);
    
    seed_ = *seedAttr;
    seed2_ = *seed2Attr;
    if (seed_ == 0 && seed2_ == 0) {
        seed_ = static_cast<int64_t>(New64());
        seed2_ = static_cast<int64_t>(New64());
    }
    OP_LOGI(opName_, "RandomUniformV2Tiling seed_ is %ld, seed2_ is %ld", seed_, seed2_);
    return ge::GRAPH_SUCCESS;
}

bool RandomUniformV2Tiling::IsCapable()
{
    return true;
}

void RandomUniformV2Tiling::SetTilingData()
{
    RandomUniformV2TilingData4RegBase* tilingData = context_->GetTilingData<RandomUniformV2TilingData4RegBase>();
    tilingData->blockNum = blockNum_;
    tilingData->normalCoreProNum = normalCoreProNum_;
    tilingData->tailCoreProNum = tailCoreProNum_;
    tilingData->singleUbSize = singleUbSize_;
    tilingData->seed = seed_;
    tilingData->seed2 = seed2_;
    tilingData->outputSize = outputSize_;
}

void RandomUniformV2Tiling::DoBlockTiling() 
{
    outputDtypeSize_ = ge::GetSizeByDataType(outDtype_);
    if (outputDtypeSize_ == 0) {
        return;
    }

    auto coreAlignFactor = CORE_ALIGN_SIZE / outputDtypeSize_;
    auto blockFactor = Ops::Base::CeilDiv(outputSize_, totalCoreNum_);
    auto blockAlignFactor = Ops::Base::CeilDiv(blockFactor, coreAlignFactor) * coreAlignFactor;
    auto minTilingSize = MIN_TILING_SIZE;
    normalCoreProNum_ = std::max(static_cast<uint32_t>(blockAlignFactor), minTilingSize);
    blockNum_ = Ops::Base::CeilDiv(outputSize_, normalCoreProNum_);
    tailCoreProNum_ = outputSize_ - normalCoreProNum_ * (blockNum_ - 1);
    return;
}

void RandomUniformV2Tiling::UbTiling() 
{
  // quarterUbSize: 2 for double buffer; coefVal for temp RNG
  int64_t coefVal = 1;
  if (outDtype_ == ge::DataType::DT_FLOAT16 || outDtype_ == ge::DataType::DT_BF16) {
    coefVal = DOUBLE_BUFFER; // philox temp buff need int32 to float16/bfloat16
  }
  auto quarterUbSize = ubSize_ / (DOUBLE_BUFFER + coefVal);
  auto ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
  auto alignFactor = ubBlockSize / outputDtypeSize_;
  singleUbSize_ = quarterUbSize / outputDtypeSize_ / alignFactor * alignFactor;
}

// 3、计算数据切分TilingData
ge::graphStatus RandomUniformV2Tiling::DoOpTiling()
{
    OP_LOGD(opName_, "RandomUniformV2Tiling DoOpTiling.");
    DoBlockTiling();
    UbTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus RandomUniformV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t RandomUniformV2Tiling::GetTilingKey() const
{
    uint64_t tilingKey = 1;
    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus RandomUniformV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus RandomUniformV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(blockNum_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void RandomUniformV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << " ubSize: " << ubSize_;
    info << " totalCoreNum: " << totalCoreNum_;
    info << " blockNum: " << blockNum_;
    info << " normalCoreProNum: " << normalCoreProNum_;
    info << " tailCoreProNum: " << tailCoreProNum_;
    info << " singleUbSize: " << singleUbSize_;
    info << " seed: " << seed_;
    info << " seed2: " << seed2_;
    info << " outputSize: " << outputSize_;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

} // namespace optiling