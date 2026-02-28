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
 * \file stateless_bernoulli_tiling_arch35.cpp
 * \brief
 */

#include "stateless_bernoulli_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"

namespace optiling {
static constexpr int64_t IN_SHAPE_IDX = 0;
static constexpr int64_t IN_PROB_IDX = 1;
static constexpr int64_t IN_SEED_IDX = 2;
static constexpr int64_t IN_OFFSET_IDX = 3;
static constexpr int64_t OUT_Y_IDX = 0;
static constexpr int64_t ATTR_TYPE_IDX = 0;

static constexpr int64_t ALG_OTHERS_NUM = 0;
static constexpr int64_t ALG_THREEFRY_NUM = 1;
static constexpr int64_t ALG_PHILOX_NUM = 2;

static constexpr uint64_t TILINGKEY_BASE_VALUE = 1000;
static constexpr uint64_t TILINGKEY_ADDITION1 = 1;
static constexpr uint64_t TILINGKEY_ADDITION2 = 2;
static constexpr uint64_t TILINGKEY_ADDITION3 = 3;

static constexpr uint32_t MIN_DIM_NUM_BOUND = 0;
static constexpr uint32_t MAX_DIM_NUM_BOUND = 8;
static constexpr uint32_t RIGHT_SHIFT_NUM = 32;

static constexpr uint64_t COUNTER_IDX_0 = 0;
static constexpr uint64_t COUNTER_IDX_1 = 1;
static constexpr uint64_t COUNTER_IDX_2 = 2;
static constexpr uint64_t COUNTER_IDX_3 = 3;
static constexpr uint64_t DIVISOR = 2;

static constexpr int64_t CORE_MINIEST_NUM = 256;
static constexpr int64_t OFFSET_MULTIPLE = 4;
static constexpr uint32_t DCACHE_SIZE = 32 * 1024;

void StatelessBernoulliTiling::Reset()
{
    opName_ = nullptr;
}

template <typename T>
ge::graphStatus StatelessBernoulliTiling::GetIntValue(const gert::Tensor* constTensor, gert::Shape& constShape)
{
    const T* constTensorValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constTensorValue);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constTensorValue[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetIntValueFromProb(
    const gert::Shape& originShape, gert::Shape& constShape, size_t shapeSize)
{
    constShape.SetDimNum(shapeSize);
    for (size_t i = 0U; i < shapeSize; i++) {
        constShape.SetDim(i, originShape.GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetIntToShape(const int64_t constIdx, gert::Shape& constShape)
{
    auto constTensor = context_->GetRequiredInputTensor(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, constTensor);

    auto inputDescPtr = context_->GetRequiredInputDesc(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDescPtr);
    auto constTensorDtype = inputDescPtr->GetDataType();

    auto ret = ge::GRAPH_FAILED;
    switch (constTensorDtype) {
        case ge::DT_INT32:
            ret = GetIntValue<int32_t>(constTensor, constShape);
            break;
        case ge::DT_INT64:
            ret = GetIntValue<int64_t>(constTensor, constShape);
            break;
        default:
            OP_LOGW(
                opName_, "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
                Ops::Base::ToString(constTensorDtype).c_str());
            return ge::GRAPH_FAILED;
    }

    auto probTensor = context_->GetRequiredInputTensor(IN_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probTensor);
    auto probOriginSize = probTensor->GetShapeSize();
    OP_CHECK_IF(
        (probOriginSize < 0),
        OP_LOGE(opName_, "prob tensor size should not smaller than 0, but got %ld.", probOriginSize),
        return ge::GRAPH_FAILED);

    probTensorSize_ = static_cast<uint64_t>(probOriginSize);
    isProbScalar_ = probTensorSize_ == 1 ? static_cast<uint64_t>(1) : static_cast<uint64_t>(0);

    if (constIdx == IN_SHAPE_IDX && !isProbScalar_) {
        const size_t constNum = constTensor->GetShapeSize();
        auto probShape = context_->GetInputShape(IN_PROB_IDX);
        auto originShape = probShape->GetOriginShape();
        ret = GetIntValueFromProb(originShape, constShape, static_cast<size_t>(constNum));
    }

    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get const value failed, please check."), return ge::GRAPH_FAILED);

    OP_LOGI(opName_, "current const value is %s", Ops::Base::ToString(constShape).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const StatelessBernoulliCompileInfoArch35*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = static_cast<int64_t>(compileInfoPtr->aivNum);
        ubSize_ = static_cast<int64_t>(compileInfoPtr->ubSize);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
        OP_LOGD(opName_, "Get aivNum form ascendcPlatform is: %ld", coreNum_);
    }
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            opName_,
            "coreNum and ubSize should not be samller than 0, but got coreNum [%lu] and ubSize [%lu], please check.",
            coreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    aicoreParams_.numBlocks = coreNum_;
    aicoreParams_.ubSize = static_cast<int64_t>(ubSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto resIn = GetInputInfo();
    if (resIn != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resOut = GetOutputInfo();
    if (resOut != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resAttr = GetAttrInfo();
    if (resAttr != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetInputKeyCounter()
{
    // check seed
    auto seedDesc = context_->GetRequiredInputDesc(IN_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedDesc);
    auto seedDtype = seedDesc->GetDataType();
    OP_CHECK_IF(
        seedDtype != ge::DataType::DT_INT64,
        OP_LOGE(opName_, "input seed dtype should be int64, but got %s.", Ops::Base::ToString(seedDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto seedTensor = context_->GetRequiredInputTensor(IN_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedTensor);
    auto seedTensorSize = static_cast<int64_t>(seedTensor->GetShapeSize());
    OP_CHECK_IF(
        seedTensorSize != 1, OP_LOGE(opName_, "input seed shape_size should be 1, but got %ld.", seedTensorSize),
        return ge::GRAPH_FAILED);

    // check offset
    auto offsetDesc = context_->GetRequiredInputDesc(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
    auto offsetDtype = offsetDesc->GetDataType();
    OP_CHECK_IF(
        offsetDtype != ge::DataType::DT_INT64,
        OP_LOGE(opName_, "input offset Dtype should be int64, but got %s.", Ops::Base::ToString(offsetDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto offsetTensor = context_->GetRequiredInputTensor(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetTensor);
    auto offsetTensorSize = static_cast<int64_t>(offsetTensor->GetShapeSize());
    OP_CHECK_IF(
        offsetTensorSize != 1, OP_LOGE(opName_, "input offset shape_size should be 1, but got %ld.", offsetTensorSize),
        return ge::GRAPH_FAILED);

    // get input value of seed & offset.
    OP_CHECK_IF(
        GetIntToShape(IN_SEED_IDX, inputSeed_) != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "get const shape of seed failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetIntToShape(IN_OFFSET_IDX, inputOffset_) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "get const shape of offset failed"), return ge::GRAPH_FAILED);
    OP_LOGD(
        opName_, "const seed = %s, const offset = %s.", Ops::Base::ToString(inputSeed_).c_str(),
        Ops::Base::ToString(inputOffset_).c_str());

    seed_ = inputSeed_[0];
    philoxOffset_ = inputOffset_[0];
    OP_CHECK_IF(
        philoxOffset_ % OFFSET_MULTIPLE != 0, OP_LOGE(opName_, "the offset value %ld must be a multiple of 4.", philoxOffset_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetInputInfo()
{
    // check shape
    auto inShapeOri = context_->GetInputShape(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShapeOri);
    const gert::Shape inShape = inShapeOri->GetStorageShape();
    OP_CHECK_IF(
        inShape.GetDimNum() != 1, OP_LOGE(opName_, "the rank of shape should be 1, but got %lu.", inShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    // check dtype
    auto shapeDesc = context_->GetRequiredInputDesc(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeDesc);
    inputDtype_ = shapeDesc->GetDataType();
    OP_CHECK_IF(
        (inputDtype_ != ge::DataType::DT_INT32) && (inputDtype_ != ge::DataType::DT_INT64),
        OP_LOGE(
            opName_, "input shape dtype should be int32, int64, but got %s.", Ops::Base::ToString(inputDtype_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetIntToShape(IN_SHAPE_IDX, inputShape_) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "get const shape of shape failed, please check."), return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "got const input shape = %s.", Ops::Base::ToString(inputShape_).c_str());

    uint32_t shapeRank = inputShape_.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        inputSize_ *= inputShape_.GetDim(idx);
    }
    OP_CHECK_IF(
        shapeRank > MAX_DIM_NUM_BOUND,
        OP_LOGE(opName_, "the rank of shape should bewteen [0-8], but got %u.", shapeRank), return ge::GRAPH_FAILED);

    // check prob
    auto probDesc = context_->GetRequiredInputDesc(IN_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probDesc);
    probDtype_ = probDesc->GetDataType();
    std::set<ge::DataType> probSupportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(
        probSupportedDtype.count(probDtype_) == 0,
        OP_LOGE(
            opName_, "prob dtype should be float, float16, bf16, but got %s.", Ops::Base::ToString(probDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto probTensor = context_->GetRequiredInputTensor(IN_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probTensor);
    auto probOriginSize = probTensor->GetShapeSize();
    OP_CHECK_IF(
        (probOriginSize < 0),
        OP_LOGE(opName_, "prob tensor size should not smaller than 0, but got %ld.", probOriginSize),
        return ge::GRAPH_FAILED);

    probTensorSize_ = static_cast<uint64_t>(probOriginSize);
    isProbScalar_ = probTensorSize_ == 1 ? static_cast<uint64_t>(1) : static_cast<uint64_t>(0);
    OP_LOGD(opName_, "probTensorSize = %lu, isProbScalar = %lu", probTensorSize_, isProbScalar_);

    // special branch
    outputSize_ = inputSize_;
    if (!isProbScalar_ && inputSize_ > probTensorSize_) {
        inputSize_ = probTensorSize_;
    }

    // check key & counter
    OP_CHECK_IF(
        GetInputKeyCounter() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "get value of seed & offset failed, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto outDtype = attrs->GetAttrPointer<ge::DataType>(ATTR_TYPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDtype);
    OP_LOGD(
        opName_, "attrDtype = [%d, %s], yDtype = [%d, %s]", static_cast<int32_t>(*outDtype),
        ge::TypeUtils::DataTypeToSerialString(*outDtype).c_str(), static_cast<int32_t>(outputDtype_),
        ge::TypeUtils::DataTypeToSerialString(outputDtype_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::GetOutputInfo()
{
    auto outputDesc = context_->GetOutputDesc(OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    outputDtype_ = outputDesc->GetDataType();
    std::set<ge::DataType> outputSupportedDtype = {ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16,
                                                   ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT64, ge::DT_UINT64,
                                                   ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL};
    OP_CHECK_IF(
        outputSupportedDtype.count(outputDtype_) == 0,
        OP_LOGE(
            opName_,
            "output dtype should be uint8, int8, uint16, int16, uint32, int32, uint64, int64, float16, float, "
            "bfloat16, bool, bug got %s.",
            Ops::Base::ToString(outputDtype_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline uint64_t StatelessBernoulliTiling::GetBytePerData(const ge::DataType& dtype)
{
    return static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

static inline int64_t Align256CeilSize(int64_t value)
{
    return static_cast<int64_t>((value + CORE_MINIEST_NUM - 1) / CORE_MINIEST_NUM * CORE_MINIEST_NUM);
}

void StatelessBernoulliTiling::BlockTiling()
{
    int64_t numOfPerCore = inputSize_;
    if (inputSize_ > CORE_MINIEST_NUM) {
        numOfPerCore = Align256CeilSize((inputSize_ + coreNum_ - 1) / coreNum_);
        blockNum_ = std::min((inputSize_ + numOfPerCore - 1) / numOfPerCore, coreNum_);
    }

    OP_LOGD(opName_, "blockNum = %lu", blockNum_);
}

void StatelessBernoulliTiling::SetTilingData()
{
    m_tilingData_.set_blockNum(blockNum_);
    m_tilingData_.set_probTensorSize(probTensorSize_);
    m_tilingData_.set_outputSize(outputSize_);
    m_tilingData_.set_isProbScalar(isProbScalar_);
    m_tilingData_.set_seed(seed_);
    m_tilingData_.set_philoxOffset(philoxOffset_);
}

ge::graphStatus StatelessBernoulliTiling::DoOpTiling()
{
    BlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StatelessBernoulliTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILINGKEY_BASE_VALUE;
    if (probDtype_ == ge::DT_FLOAT) {
        tilingKey += TILINGKEY_ADDITION1;
    } else if (probDtype_ == ge::DT_FLOAT16) {
        tilingKey += TILINGKEY_ADDITION2;
    } else if (probDtype_ == ge::DT_BF16) {
        tilingKey += TILINGKEY_ADDITION3;
    }
    OP_LOGD(opName_, "tilingKey = %lu.", tilingKey);
    return tilingKey;
}

void StatelessBernoulliTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "blockNum: " << m_tilingData_.get_blockNum() << ", ";
    info << "probTensorSize: " << m_tilingData_.get_probTensorSize() << ", ";
    info << "outputSize: " << m_tilingData_.get_outputSize() << ", ";
    info << "isProbScalar: " << m_tilingData_.get_isProbScalar() << ", ";
    info << "seed: " << m_tilingData_.get_seed() << ", ";
    info << "philoxOffset: " << m_tilingData_.get_philoxOffset() << ", ";
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus StatelessBernoulliTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessBernoulliTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(blockNum_);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    context_->SetTilingKey(GetTilingKey());

    if (m_tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4StatelessBernoulli(gert::TilingContext* context)
{
    StatelessBernoulliTiling tilingObj(context);
    return tilingObj.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessBernoulli([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StatelessBernoulli)
    .Tiling(Tiling4StatelessBernoulli)
    .TilingParse<StatelessBernoulliCompileInfoArch35>(TilingPrepare4StatelessBernoulli)
    .TilingInputsDataDependency({IN_SHAPE_IDX, IN_SEED_IDX, IN_OFFSET_IDX});
} // namespace optiling
