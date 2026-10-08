/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confusion_matrix_tiling.cpp
 * \brief confusion_matrix_tiling file
 */

#include <iostream>
#include <cstring>
#include <string>
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"
#include "confusion_matrix_tiling.h"
#include "util/math_util.h"
#include "util/const_util.h"

namespace optiling {
ge::graphStatus ConfusionMatrixTiling::Init()
{
    OP_LOGD(context_->GetNodeName(), "ConfusionMatrixTiling init enter.");
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<ConfusionMatrixTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ConfusionMatrixTilingData), 0, sizeof(ConfusionMatrixTilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "ConfusionMatrixTiling init exit.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionMatrixTiling::ConfusionMatrixGetPlatformData(
    const AscendCConfusionMatrixCompileInfo* compileInfo)
{
    coreNum_ = compileInfo->totalCoreNum;
    ubSize_ = static_cast<int64_t>(compileInfo->totalUbSize - SIMD_SIMT_DCACHE_SIZE);
    isDetermine_ = context_->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGI(context_->GetNodeName(),
            "ConfusionMatrixGetPlatformData ubSize is %ld, coreNum_ is %ld, isDetermine is %ld", ubSize_, coreNum_,
            isDetermine_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionMatrixTiling::CheckShape()
{
    auto labelsShape = context_->GetInputShape(INPUT_IDX_LABELS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, labelsShape);
    OP_CHECK_IF(labelsShape->GetStorageShape().GetDimNum() != DIM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "labels",
                                             std::to_string(labelsShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);

    auto predictionsShape = context_->GetInputShape(INPUT_IDX_PREDICTIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, predictionsShape);
    OP_CHECK_IF(
        predictionsShape->GetStorageShape().GetDimNum() != DIM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "predictions",
                                     std::to_string(predictionsShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);

    auto labelsShapeSize = labelsShape->GetStorageShape().GetShapeSize();
    auto predictionsShapeSize = predictionsShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(labelsShapeSize != predictionsShapeSize,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "labels and predictions",
                    (std::to_string(labelsShapeSize) + " and " + std::to_string(predictionsShapeSize)).c_str(),
                    "The shapes of labels and predictions must be the same"),
                return ge::GRAPH_FAILED);

    auto weightsShape = context_->GetInputShape(INPUT_IDX_WEIGHTS);
    if (weightsShape != nullptr) {
        OP_CHECK_IF(
            weightsShape->GetStorageShape().GetDimNum() != DIM_1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "weights",
                                         std::to_string(weightsShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
            return ge::GRAPH_FAILED);
        auto weightsShapeSize = weightsShape->GetStorageShape().GetShapeSize();
        if (weightsShapeSize != 0) {
            OP_CHECK_IF(weightsShapeSize != labelsShapeSize,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            context_->GetNodeName(), "weights",
                            (std::to_string(weightsShapeSize) + " and " + std::to_string(labelsShapeSize)).c_str(),
                            "The shape of weights must be the same as labels when provided"),
                        return ge::GRAPH_FAILED);
        }
    }

    auto yShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    OP_CHECK_IF(yShape->GetStorageShape().GetDimNum() != DIM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y",
                                             std::to_string(yShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionMatrixTiling::CheckDtype()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const char* dtypePtr = attrs->GetAttrPointer<char>(ATTR_IDX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dtypePtr);
    std::string dtypeStr(dtypePtr);

    if (dtypeStr == "float32") {
        outputDtype_ = OUTPUT_DTYPE_FLOAT;
        sizeOfOutputDtype_ = static_cast<int64_t>(SIZE_DTYPE_FLOAT);
    } else if (dtypeStr == "int32") {
        outputDtype_ = OUTPUT_DTYPE_INT32;
        sizeOfOutputDtype_ = static_cast<int64_t>(SIZE_DTYPE_INT32);
    } else if (dtypeStr == "float16") {
        outputDtype_ = OUTPUT_DTYPE_FLOAT16;
        sizeOfOutputDtype_ = static_cast<int64_t>(SIZE_DTYPE_FLOAT16);
    } else if (dtypeStr == "int8") {
        outputDtype_ = OUTPUT_DTYPE_INT8;
        sizeOfOutputDtype_ = static_cast<int64_t>(SIZE_DTYPE_INT8);
    } else if (dtypeStr == "uint8") {
        outputDtype_ = OUTPUT_DTYPE_UINT8;
        sizeOfOutputDtype_ = static_cast<int64_t>(SIZE_DTYPE_UINT8);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dtype", dtypeStr.c_str(),
                                  "float32, int32, float16, int8 or uint8");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionMatrixTiling::CheckInputParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* numClassesPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_NUM_CLASSES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numClassesPtr);
    numClasses_ = *numClassesPtr;
    OP_CHECK_IF((numClasses_ < NUM_CLASSES_MIN || numClasses_ > NUM_CLASSES_MAX),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_classes",
                                                      std::to_string(numClasses_).c_str(),
                                                      "num_classes must be in range [1, 4096]."),
                return ge::GRAPH_FAILED);

    auto labelsShape = context_->GetInputShape(INPUT_IDX_LABELS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, labelsShape);
    inputSize_ = labelsShape->GetStorageShape().GetShapeSize();

    auto weightsShape = context_->GetInputShape(INPUT_IDX_WEIGHTS);
    if (weightsShape != nullptr) {
        int64_t weightsShapeSize = weightsShape->GetStorageShape().GetShapeSize();
        if (weightsShapeSize != 0) {
            isWeight_ = WEIGHT;
        }
    }
    OP_LOGI(context_->GetNodeName(), "inputSize is %ld, isWeight is %ld, numClasses is %ld", inputSize_, isWeight_,
            numClasses_);

    outputSize_ = numClasses_ * numClasses_;
    OP_LOGI(context_->GetNodeName(), "outputSize is %ld", outputSize_);

    return ge::GRAPH_SUCCESS;
}

inline bool ConfusionMatrixTiling::IsMatchSimtBatchLoadMode()
{
    return (2 * inputSize_) > outputSize_ / GM_ATOMIC_ADD_FACTOR;
}

ge::graphStatus ConfusionMatrixTiling::ComputeTilingStrategy()
{
    OP_LOGD(context_->GetNodeName(), "ComputeTilingStrategy enter.");
    ubNumCanUse_ = static_cast<int64_t>(ubSize_ / sizeOfOutputDtype_);

    // For int8/uint8 output, asc_atomic_add does not support these types.
    // Force DETERMINE mode which uses direct += on disjoint output ranges
    // (no atomic operations needed).
    if (isDetermine_ || outputDtype_ == OUTPUT_DTYPE_INT8 || outputDtype_ == OUTPUT_DTYPE_UINT8) {
        schId_ = SCH_ID_SIMT_DETERMIN;
        return ComputeTilingSimtDetermine();
    }

    if (outputSize_ < ubNumCanUse_) {
        schId_ = SCH_ID_SIMT_FULL_LOAD;
    } else if (IsMatchSimtBatchLoadMode()) {
        ubLoopNum_ = Ops::Base::CeilDiv(outputSize_, ubNumCanUse_);
        if (ubLoopNum_ <= MAX_UB_LOOP_FOR_BATCH) {
            schId_ = SCH_ID_SIMT_BATCH_LOAD;
        } else {
            schId_ = SCH_ID_SIMT_NOT_FULL_LOAD;
        }
    } else {
        schId_ = SCH_ID_SIMT_NOT_FULL_LOAD;
    }

    return ComputeTilingSimtNotDetermine();
}

ge::graphStatus ConfusionMatrixTiling::ComputeTilingSimtNotDetermine()
{
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtNotDetermine enter.");
    ubLoopNum_ = Ops::Base::CeilDiv(outputSize_, ubNumCanUse_);
    if (inputSize_ > 0) {
        formerLength_ = Ops::Base::CeilDiv(inputSize_, coreNum_);
        OP_CHECK_IF(formerLength_ == 0, OP_LOGE(context_->GetNodeName(), "formerLength_ must not be 0."),
                    return ge::GRAPH_FAILED);
        needXCoreNum_ = Ops::Base::CeilDiv(inputSize_, formerLength_);
        tailLength_ = inputSize_ - (needXCoreNum_ - 1) * formerLength_;
    } else {
        formerLength_ = 1;
        needXCoreNum_ = 1;
        tailLength_ = 0;
    }

    clearYFactor_ = Ops::Base::CeilDiv(outputSize_, coreNum_);
    OP_CHECK_IF(clearYFactor_ == 0, OP_LOGE(context_->GetNodeName(), "clearYFactor_ must not be 0."),
                return ge::GRAPH_FAILED);
    clearYCoreNum_ = Ops::Base::CeilDiv(outputSize_, clearYFactor_);
    clearYTail_ = outputSize_ - (clearYCoreNum_ - 1) * clearYFactor_;
    needCoreNum_ = std::max(needXCoreNum_, clearYCoreNum_);
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtNotDetermine end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConfusionMatrixTiling::ComputeTilingSimtDetermine()
{
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtDetermine enter.");
    binsFormerLength_ = Ops::Base::CeilDiv(outputSize_, coreNum_);
    OP_CHECK_IF(binsFormerLength_ == 0, OP_LOGE(context_->GetNodeName(), "binsFormerLength_ must not be 0."),
                return ge::GRAPH_FAILED);
    needBinsCoreNum_ = Ops::Base::CeilDiv(outputSize_, binsFormerLength_);
    binsTailLength_ = outputSize_ - (needBinsCoreNum_ - 1) * binsFormerLength_;
    needCoreNum_ = needBinsCoreNum_;
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtDetermine end.");
    return ge::GRAPH_SUCCESS;
}

void ConfusionMatrixTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "ConfusionMatrix tilingData needCoreNum_ is %ld, numClasses is %ld,"
            "ubNumCanUse is %ld, ubLoopNum is %ld, needXCoreNum is %ld, formerLength is %ld, tailLength is %ld,"
            "clearYCoreNum is %ld, clearYFactor is %ld, clearYTail is %ld, binsFormerLength is %ld"
            "needBinsCoreNum is %ld, binsTailLength is %ld",
            needCoreNum_, tilingData_->numClasses, tilingData_->ubNumCanUse, tilingData_->ubLoopNum,
            tilingData_->needXCoreNum, tilingData_->formerLength, tilingData_->tailLength, tilingData_->clearYCoreNum,
            tilingData_->clearYFactor, tilingData_->clearYTail, tilingData_->binsFormerLength,
            tilingData_->needBinsCoreNum, tilingData_->binsTailLength);
    return;
}

ge::graphStatus ConfusionMatrixTiling::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "SetTilingData enter.");
    tilingData_->numClasses = numClasses_;
    tilingData_->inputSize = inputSize_;
    tilingData_->ubNumCanUse = ubNumCanUse_;
    tilingData_->ubLoopNum = ubLoopNum_;
    tilingData_->needXCoreNum = needXCoreNum_;
    tilingData_->formerLength = formerLength_;
    tilingData_->tailLength = tailLength_;
    tilingData_->clearYCoreNum = clearYCoreNum_;
    tilingData_->clearYFactor = clearYFactor_;
    tilingData_->clearYTail = clearYTail_;
    tilingData_->binsFormerLength = binsFormerLength_;
    tilingData_->needBinsCoreNum = needBinsCoreNum_;
    tilingData_->binsTailLength = binsTailLength_;
    // UB buffer 字节数由 host 一次算准，kernel 只透传（避免 host/内核两套算法）
    tilingData_->fullLoadBufSize = outputSize_ * sizeOfOutputDtype_;
    tilingData_->batchUbBufSize = ubNumCanUse_ * sizeOfOutputDtype_;
    OP_LOGI(context_->GetNodeName(), "schId is %ld, outputDtype is %ld, isWeight is %ld", schId_, outputDtype_,
            isWeight_);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId_, outputDtype_, isWeight_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is %ld", tilingKey);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(needCoreNum_);
    context_->SetLocalMemorySize(ubSize_);
    context_->SetScheduleMode(1); // SyncAll need set
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ConfusionMatrix(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const AscendCConfusionMatrixCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    ConfusionMatrixTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.ConfusionMatrixGetPlatformData(compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "ConfusionMatrixGetPlatformData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.CheckShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CheckShape return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.CheckDtype() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CheckDtype return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.CheckInputParams() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CheckInputParams return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.ComputeTilingStrategy() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "ComputeTilingStrategy return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.SetTilingData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "SetTilingData return failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ConfusionMatrix(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4ConfusionMatrix running.");
    auto compileInfo = context->GetCompiledInfo<AscendCConfusionMatrixCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0),
                OP_LOGE(context->GetNodeName(), "coreNum is invalid, must greater than zero"), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->totalUbSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->totalUbSize <= 0),
                OP_LOGE(context->GetNodeName(), "ubSize is invalid, must greater than zero"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "totalUbSize is %lu.", compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ConfusionMatrix)
    .Tiling(Tiling4ConfusionMatrix)
    .TilingParse<AscendCConfusionMatrixCompileInfo>(TilingPrepare4ConfusionMatrix);

} // namespace optiling
