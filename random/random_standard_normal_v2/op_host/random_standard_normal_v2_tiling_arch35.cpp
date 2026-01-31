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
 * \file random_standard_normal_v2_tiling_arch35.cpp
 * \brief
 */
#include <random>
#include <iostream>
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "random_standard_normal_v2_tiling.h"
#include "random_standard_normal_v2_tiling_arch35.h"
#include "../op_kernel/arch35/random_standard_normal_v2_struct.h"
#include  "../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t IN_SHAPE_IDX = 0;
static constexpr uint16_t IN_OFFSET_IDX = 1;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t ATTR_TYPE_IDX = 0;
static constexpr uint16_t ATTR_SEED_IDX = 1;
static constexpr uint16_t ATTR_SEED2_IDX = 2;
static constexpr uint16_t SIZE_OF_FLOAT = 4;
static constexpr uint16_t SPLIT_UB_NUM = 3;
static constexpr uint32_t CORE_ALIGN_SIZE = 512; // 性能考虑，512字节对齐性能好
static constexpr uint32_t MIN_TILING_SIZE = 256;
static constexpr uint32_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t DCACHE_SIZE = 128U * 1024U;

static const std::set<ge::DataType> RANDOM_STANDARD_NORMAL_V2_SUPPORT_DTYPE = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16};

template <typename T>
ge::graphStatus RandomStandardNormalV2Tiling::GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape)
{
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetIntValue begin.");
    const T *constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constValue);
    OP_LOGD(opName_, "RandomStandardNormalV2Tiling::GetIntValue constValue=%d.", constValue[0]);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetIntValue constNum=%d", constNum);
    return ge::GRAPH_SUCCESS;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus RandomStandardNormalV2Tiling::GetPlatformInfo()
{
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling GetPlatformInfo.");
    auto compileInfo = static_cast<const RandomStandardNormalV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSize;
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ -= DCACHE_SIZE;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size less than Dcache Size. please check."), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetPlatformInfo ubSize_=%d, totalCoreNum_=%d", ubSize_, totalCoreNum_);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS),
        OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", static_cast<int64_t>(ubSize_)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus RandomStandardNormalV2Tiling::GetShapeAttrsInfo()
{
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetShapeAttrsInfo begin.");
    OP_CHECK_IF(GetInputInfo(), 
        OP_LOGE(opName_, "GetInputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetOutputInfo(), 
        OP_LOGE(opName_, "GetOutputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(shapeSize_ != outputSize_,
        OP_LOGE(opName_, "shape size: %ld is not equal to out size: %ld.", shapeSize_, outputSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrInfo(), 
        OP_LOGE(opName_, "GetAttrInfo failed!"), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomStandardNormalV2Tiling::GetInputInfo()
{
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetInputInfo begin.");
    auto shapeDesc = context_->GetRequiredInputDesc(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeDesc);
    shapeDtype_ = shapeDesc->GetDataType();
    OP_CHECK_IF((shapeDtype_ != ge::DataType::DT_INT32) && (shapeDtype_ != ge::DataType::DT_INT64),
        OP_LOGE(opName_, "input shape dtype should be int32, int64, but got %s.",
        Ops::Base::ToString(shapeDtype_).c_str()), return ge::GRAPH_FAILED);

    auto input1_shape = context_->GetInputShape(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1_shape);
    uint32_t shapeDimNum = input1_shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(shapeDimNum != 1,
        OP_LOGE(opName_, "input shape is 1D tensor, but got %u.",
        shapeDimNum), return ge::GRAPH_FAILED);

    auto shapeTensor = context_->GetRequiredInputTensor(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeTensor);
    
    gert::Shape constShape;
    if (shapeDtype_ == ge::DataType::DT_INT32) {
        GetIntValue<int32_t>(shapeTensor, constShape);
    } else {
        GetIntValue<int64_t>(shapeTensor, constShape);
    }
    OP_LOGD(opName_, "RandomStandardNormalV2Tiling::GetInputInfo get shapeTensor end.");

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

    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetInputInfo end.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomStandardNormalV2Tiling::GetOutputInfo()
{
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetOutputInfo begin.");
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
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetOutputInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomStandardNormalV2Tiling::GetAttrInfo()
{
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling::GetAttrInfo begin.");
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
    OP_LOGI(opName_, "RandomStandardNormalV2Tiling seed_ is %ld, seed2_ is %ld", seed_, seed2_);
    return ge::GRAPH_SUCCESS;
}

bool RandomStandardNormalV2Tiling::IsCapable()
{
    return true;
}

void RandomStandardNormalV2Tiling::SetTilingData() const
{
    RandomStandardNormalV2TilingData4RegBase* tilingData =
        context_->GetTilingData<RandomStandardNormalV2TilingData4RegBase>();
    tilingData->perCoreHandleRandom = perCoreHandleRandom_;
    tilingData->blockFactor = blockFactor_;
    tilingData->tailBlockFactor = tailBlockFactor_;
    tilingData->ubFactor = ubFactor_;
    tilingData->tailUbFactor = tailUbFactor_;
    tilingData->tailBlockTailUbFactor = tailBlockTailUbFactor_;
    tilingData->seed = seed_;
    tilingData->seed2 = seed2_;
    tilingData->shapeSize = shapeSize_;
}

void RandomStandardNormalV2Tiling::DoBlockTiling() 
{
    outputDtypeSize_ = static_cast<int64_t>(SIZE_OF_FLOAT);
    // splitUbSize: 2 for double buffer; 1 for data converse
    auto splitUbSize = ubSize_ / static_cast<int64_t>(SPLIT_UB_NUM);
    auto alignFactor = Ops::Base::GetUbBlockSize(context_) / outputDtypeSize_;
    ubFactor_ = Ops::Base::FloorAlign(splitUbSize / outputDtypeSize_, alignFactor); // 一次UB容纳数据量

    auto coreAlignFactor = static_cast<int64_t>(CORE_ALIGN_SIZE) / outputDtypeSize_; 
    auto perCoreHandleRandom = Ops::Base::CeilDiv(outputSize_, totalCoreNum_);
    auto perCoreHandleRandomAlign = Ops::Base::CeilAlign(perCoreHandleRandom, coreAlignFactor);
    auto minTilingSize = MIN_TILING_SIZE;
    perCoreHandleRandom_ = std::max(static_cast<uint32_t>(perCoreHandleRandomAlign), minTilingSize);
    blockNum_ = Ops::Base::CeilDiv(outputSize_, perCoreHandleRandom_);
    auto tailCoreHandleRandom = outputSize_ - (blockNum_ - 1) * perCoreHandleRandom_;
    blockFactor_ = Ops::Base::CeilDiv(perCoreHandleRandom_, ubFactor_);
    tailUbFactor_ = perCoreHandleRandom_ - (blockFactor_ - 1) * ubFactor_;
    tailBlockFactor_ = Ops::Base::CeilDiv(tailCoreHandleRandom, ubFactor_);
    tailBlockTailUbFactor_ = tailCoreHandleRandom - (tailBlockFactor_ - 1) * ubFactor_;
}

// 3、计算数据切分TilingData
ge::graphStatus RandomStandardNormalV2Tiling::DoOpTiling()
{
    OP_LOGD(opName_, "RandomStandardNormalV2Tiling DoOpTiling.");
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus RandomStandardNormalV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t RandomStandardNormalV2Tiling::GetTilingKey() const
{
    uint64_t tilingKey = 1;

    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus RandomStandardNormalV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus RandomStandardNormalV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(blockNum_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void RandomStandardNormalV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    RandomStandardNormalV2TilingData4RegBase* tilingData =
        context_->GetTilingData<RandomStandardNormalV2TilingData4RegBase>();
    
    info << "ubSize: " << ubSize_;
    info << "totalCoreNum: " << totalCoreNum_;
    info << "blockNum: " << blockNum_;
    info << "perCoreHandleRandom:" << perCoreHandleRandom_;
    info << "BlockFactor: " << tilingData->blockFactor;
    info << "tailBlockFactor: " << tilingData->tailBlockFactor;
    info << "ubFactor: " << tilingData->ubFactor;
    info << "tailubFactor: " << tilingData->tailUbFactor;
    info << "tailBlockTailUbFactor: " << tilingData->tailBlockTailUbFactor;
    info << "seed: " << tilingData->seed;
    info << "seed2: " << tilingData->seed2;
    info << "shapeSize:" << tilingData->shapeSize;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

} // namespace optiling