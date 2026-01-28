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
 * \file stateless_drop_out_gen_mask_tiling_arch35.cpp
 * \brief
 */
#include "stateless_drop_out_gen_mask_tiling_arch35.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {
static constexpr int64_t IN_SHAPE_IDX = 0;
static constexpr int64_t IN_PROB_IDX = 1;
static constexpr int64_t IN_SEED_IDX = 2;
static constexpr int64_t IN_SEED1_IDX = 3;
static constexpr int64_t IN_OFFSET_IDX = 4;
static constexpr int64_t OUT_Y_IDX = 0;

static constexpr int64_t ALG_OTHERS_NUM = 0;
static constexpr int64_t ALG_THREEFRY_NUM = 1;
static constexpr int64_t ALG_PHILOX_NUM = 2;

static constexpr uint64_t TILINGKEY_INIT_VALUE = 1000;
static constexpr uint64_t TILINGKEY_ADDITION1 = 1;
static constexpr uint64_t TILINGKEY_ADDITION2 = 2;
static constexpr uint64_t TILINGKEY_ADDITION3 = 3;

static constexpr uint32_t MIN_DIM_NUM_BOUND = 0;
static constexpr uint32_t MAX_DIM_NUM_BOUND = 8;
static constexpr uint32_t RIGHT_SHIFT_NUM = 32;

template <typename T1, typename T2>
inline T1 StatelessDropOutGenMaskTiling::CeilDiv(const T1 x, const T2 y) const
{
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename T1, typename T2>
inline T1 StatelessDropOutGenMaskTiling::FloorDiv(const T1 x, const T2 y) const
{
    return y == 0 ? x : x / y;
}

template <typename T1, typename T2>
inline T1 StatelessDropOutGenMaskTiling::CeilAlign(const T1 x, const T2 align) const
{
    return CeilDiv(x, align) * align;
}

template <typename T1, typename T2>
inline T1 StatelessDropOutGenMaskTiling::FloorAlign(const T1 x, const T2 align) const
{
    return align == 0 ? 0 : x / align * align;
}

template <typename T>
ge::graphStatus StatelessDropOutGenMaskTiling::GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape)
{
    const T *constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constValue);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetIntToShape(const int64_t constIdx, gert::Shape &constShape)
{
    auto constTensor = context_->GetRequiredInputTensor(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, constTensor);

    auto inputDescPtr = context_->GetRequiredInputDesc(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDescPtr);
    auto constDtype = inputDescPtr->GetDataType();

    auto ret = ge::GRAPH_FAILED;
    switch (constDtype) {
        case ge::DT_INT32:
            ret = GetIntValue<int32_t>(constTensor, constShape);
            break;
        case ge::DT_INT64:
            ret = GetIntValue<int64_t>(constTensor, constShape);
            break;
        case ge::DT_UINT32:
            ret = GetIntValue<uint32_t>(constTensor, constShape);
            break;
        case ge::DT_UINT64:
            ret = GetIntValue<uint64_t>(constTensor, constShape);
            break;
        default:
            OP_LOGW(opName, "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
                Ops::Base::ToString(constDtype).c_str());
            return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "get const value failed, please check."),
                    return ge::GRAPH_FAILED);
    OP_LOGI(opName, "current const value is %s", Ops::Base::ToString(constShape).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetPlatformInfo()
{
    auto compileInfoPtr = reinterpret_cast<const StatelessDropOutGenMaskCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfoPtr->aivNum <= 0),
        OP_LOGE(opName, "statelessDropOutGenMaskTiling fail to get coreNum."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfoPtr->ubSize <= REGBASE_CCEC_CACHE_SIZE),
        OP_LOGE(opName, "ub size less than REGBASE_CCEC_CACHE_SIZE Size, please check."),
        return ge::GRAPH_FAILED);
    coreNum_ = compileInfoPtr->aivNum;
    ubSize_ = compileInfoPtr->ubSize - REGBASE_CCEC_CACHE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetShapeAttrsInfo()
{
    auto resIn = GetInputInfo();
    if (resIn != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto resOut = GetOutputInfo();
    if (resOut != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

int64_t StatelessDropOutGenMaskTiling::GetCounterSize(Algorithm alg) const
{
    if (alg == Algorithm::RNG_ALG_PHILOX) {
        return ALG_PHILOX_NUM;
    } else if (alg == Algorithm::RNG_ALG_THREEFRY) {
        return ALG_THREEFRY_NUM;
    }
    OP_LOGD(opName, "Current alg not support, please check.");
    return ALG_OTHERS_NUM;
}

void StatelessDropOutGenMaskTiling::GetKeyFromMem(const int64_t key)
{
    key_[0] = static_cast<int32_t>(key);
    key_[1] = static_cast<int32_t>(key >> RIGHT_SHIFT_NUM); // 32 for lower 32 bits
}
void StatelessDropOutGenMaskTiling::GetCounterFromMem(const std::vector<int64_t> &counter)
{
    counter_[0] = static_cast<int32_t>(counter[0]);
    counter_[1] = static_cast<int32_t>(counter[0] >> RIGHT_SHIFT_NUM); // 32 for lower 32 bits
    counter_[2] = static_cast<int32_t>(counter[1]);
    counter_[3] = static_cast<int32_t>(counter[1] >> RIGHT_SHIFT_NUM); // 32 for lower 32 bits
}

ge::graphStatus StatelessDropOutGenMaskTiling::CheckSeedAndSeed1()
{
    // check seed
    auto seedDesc = context_->GetRequiredInputDesc(IN_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedDesc);
    auto seedDtype = seedDesc->GetDataType();
    OP_CHECK_IF((seedDtype != ge::DataType::DT_INT32) && (seedDtype != ge::DataType::DT_INT64),
        OP_LOGE(opName, "input seed dtype should be int32、int64, but got %d", seedDtype),
        return ge::GRAPH_FAILED);
    auto seedTensor = context_->GetRequiredInputTensor(IN_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedTensor);
    auto seedTensorSize = static_cast<int64_t>(seedTensor->GetShapeSize());
    OP_CHECK_IF(seedTensorSize != 1,
        OP_LOGE(opName, "input seed shape_size should be 1, but got %ld.", seedTensorSize),
        return ge::GRAPH_FAILED);

    // check seed1
    auto seed1Desc = context_->GetRequiredInputDesc(IN_SEED1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seed1Desc);
    auto seed1Dtype = seed1Desc->GetDataType();
    OP_CHECK_IF((seed1Dtype != ge::DataType::DT_INT32) && (seed1Dtype != ge::DataType::DT_INT64),
        OP_LOGE(opName, "input seed1 Dtype should be int32、int64, but got %d", seed1Dtype),
        return ge::GRAPH_FAILED);
    auto seed1Tensor = context_->GetRequiredInputTensor(IN_SEED1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seed1Tensor);
    auto seed1TensorSize = static_cast<int64_t>(seed1Tensor->GetShapeSize());
    OP_CHECK_IF(seed1TensorSize != 1,
        OP_LOGE(opName, "input seed1 shape_size should be 1, but got %ld.", seed1TensorSize),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetInputKeyCounter()
{
    // check seed & seed1
    OP_CHECK_IF(CheckSeedAndSeed1() != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "input seed or seed1 param check failed, please check."),
        return ge::GRAPH_FAILED);

    // check offset
    auto offsetDesc = context_->GetRequiredInputDesc(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
    auto offsetDtype = offsetDesc->GetDataType();
    OP_CHECK_IF(offsetDtype != ge::DataType::DT_INT64,
        OP_LOGE(opName, "input offset Dtype should be int64, but got %d.", offsetDtype),
        return ge::GRAPH_FAILED);
    auto offsetTensor = context_->GetRequiredInputTensor(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetTensor);
    auto offsetTensorSize = static_cast<int64_t>(offsetTensor->GetShapeSize());
    OP_CHECK_IF((offsetTensorSize != 1) && (offsetTensorSize != 2),
        OP_LOGE(opName, "input offset shape_size should be 1 or 2, but got %ld.", offsetTensorSize),
        return ge::GRAPH_FAILED);

    // get input value of seed & offset.
    OP_CHECK_IF(GetIntToShape(IN_SEED_IDX, inputSeed_) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "get const shape of seed failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetIntToShape(IN_OFFSET_IDX, inputOffset_) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "get const shape of offset failed"), return ge::GRAPH_FAILED);
    OP_LOGD(opName, "const seed = %s, const offset = %s.", Ops::Base::ToString(inputSeed_).c_str(),
        Ops::Base::ToString(inputOffset_).c_str());

    int64_t key = static_cast<int64_t>(inputSeed_[0]);
    std::vector<int64_t> counter;
    if (offsetTensorSize == 1) {
        counter = { 0, inputOffset_[0] };
    } else {
        counter = { inputOffset_[0], inputOffset_[1] };
    }
    OP_CHECK_IF(static_cast<int64_t>(counter.size()) < GetCounterSize(Algorithm(alg_)),
        OP_LOGE(opName, "counter tensor elements number at least %ld.",
        GetCounterSize(Algorithm(alg_))),return ge::GRAPH_FAILED);

    GetKeyFromMem(key);
    GetCounterFromMem(counter);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetInputInfo()
{
    // check shape
    auto shapeDesc = context_->GetRequiredInputDesc(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeDesc);
    inputDtype_ = shapeDesc->GetDataType();
    OP_CHECK_IF((inputDtype_ != ge::DataType::DT_INT32) && (inputDtype_ != ge::DataType::DT_INT64),
        OP_LOGE(opName, "input shape dtype should be int32、int64, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetIntToShape(IN_SHAPE_IDX, inputShape_) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "get const shape of shape failed, please check."),
        return ge::GRAPH_FAILED);
    OP_LOGD(opName, "got const input shape = %s.", Ops::Base::ToString(inputShape_).c_str());

    uint32_t shapeRank = inputShape_.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        inputSize_ *= inputShape_.GetDim(idx);
    }
    OP_CHECK_IF(shapeRank > MAX_DIM_NUM_BOUND,
        OP_LOGE(opName, "the rank of shape should between [0-8], please check."),
        return ge::GRAPH_FAILED);

    // check prob
    auto probDesc = context_->GetRequiredInputDesc(IN_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probDesc);
    probDtype_ = probDesc->GetDataType();
    std::set<ge::DataType> probSupportedDtype = { ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16 };
    OP_CHECK_IF(probSupportedDtype.count(probDtype_) == 0,
        OP_LOGE(opName, "prob dtype should be float、float16、bf16, but got %d.", probDtype_),
        return ge::GRAPH_FAILED);

    auto probTensor = context_->GetRequiredInputTensor(IN_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probTensor);
    auto probTensorSize = static_cast<int64_t>(probTensor->GetShapeSize());
    OP_CHECK_IF(probTensorSize != 1,
        OP_LOGE(opName, "prob data shape_size should be 1, but got %ld.", probTensorSize),
        return ge::GRAPH_FAILED);

    // check key & counter
    OP_CHECK_IF(GetInputKeyCounter() != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "get value of seed & offset failed, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetOutputInfo()
{
    auto outputDesc = context_->GetOutputDesc(OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(outputDtype != ge::DataType::DT_UINT8,
        OP_LOGE(opName, "input shape dtype should be uint8, but got %d.", outputDtype),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline uint64_t StatelessDropOutGenMaskTiling::GetBytePerData(const ge::DataType &dtype)
{
    return static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

void StatelessDropOutGenMaskTiling::BlockTiling()
{
    inputDtypeSize_ = GetBytePerData(inputDtype_);
    auto coreAlignFactor = FloorDiv(CORE_ALIGN_SIZE, inputDtypeSize_);
    auto blockFactor = CeilDiv(inputSize_, coreNum_);
    auto blockAlignFactor = CeilAlign(CeilAlign(blockFactor, coreAlignFactor), MIN_TILING_SIZE);
    auto minTilingSize = MIN_TILING_SIZE;
    blockTilingSize_ = std::max(static_cast<uint64_t>(blockAlignFactor), minTilingSize);
    blockNum_ = CeilDiv(inputSize_, blockTilingSize_);
    tailBlockTilingSize_ = inputSize_ - blockTilingSize_ * (blockNum_ - 1);

    OP_LOGD(opName,
        "inputSize = %lu, blockFactor = %lu, blockAlignFactor = %lu, blockTilingSize = %lu, \
tailBlockTilingSize = %lu, blockNum = %lu, coreNum = %lu, ubSize = %lu",
        inputSize_, blockFactor, blockAlignFactor, blockTilingSize_, tailBlockTilingSize_, blockNum_, coreNum_,
        ubSize_);
}

ge::graphStatus StatelessDropOutGenMaskTiling::UbTiling()
{
    // ub size
    auto quarterUbSize = ubSize_ / BUFFER_NUM / EXIST_NODE_NUM;
    OP_LOGD(opName, "quarterUbSize = %lu", quarterUbSize);

    // loop count
    auto ubTilingSize = FloorAlign(FloorDiv(quarterUbSize, inputDtypeSize_), MIN_TILING_SIZE);
    OP_CHECK_IF((ubTilingSize == 0),
    OP_LOGE(opName, "the divisor is %lu.",
    ubTilingSize), return ge::GRAPH_FAILED);
    ubTilingSize_ = (blockTilingSize_ % ubTilingSize) ? MIN_TILING_SIZE : ubTilingSize;
    
    blockLoopCount_ = CeilDiv(blockTilingSize_, ubTilingSize_);
    tailBlockLoopCount_ = CeilDiv(tailBlockTilingSize_, ubTilingSize_);
    OP_LOGD(opName, "ubTilingSize_ = %lu, blockLoopCount_ = %lu, tailBlockLoopCount_ = %lu",
        ubTilingSize_, blockLoopCount_, tailBlockLoopCount_);

    return ge::GRAPH_SUCCESS;
}

void StatelessDropOutGenMaskTiling::SetTilingData()
{
    m_tilingData_.set_blockNum(blockNum_);
    m_tilingData_.set_blockTilingSize(blockTilingSize_);
    m_tilingData_.set_tailBlockTilingSize(tailBlockTilingSize_);
    m_tilingData_.set_blockLoopCount(blockLoopCount_);
    m_tilingData_.set_tailBlockLoopCount(tailBlockLoopCount_);
    m_tilingData_.set_ubTilingSize(ubTilingSize_);
    m_tilingData_.set_key(key_);
    m_tilingData_.set_counter(counter_);
}

ge::graphStatus StatelessDropOutGenMaskTiling::DoOpTiling()
{
    BlockTiling();
    if (UbTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StatelessDropOutGenMaskTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILINGKEY_INIT_VALUE;
    if (probDtype_ == ge::DT_FLOAT) {
        tilingKey += TILINGKEY_ADDITION1;
    } else if (probDtype_ == ge::DT_FLOAT16) {
        tilingKey += TILINGKEY_ADDITION2;
    } else if (probDtype_ == ge::DT_BF16) {
        tilingKey += TILINGKEY_ADDITION3;
    }
    OP_LOGD(opName, "tilingKey = %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus StatelessDropOutGenMaskTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessDropOutGenMaskTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum_);

    if (m_tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4StatelessDropOutGenMask(gert::TilingContext *context)
{
    StatelessDropOutGenMaskTiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4StatelessDropOutGenMask([[maybe_unused]] gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "op [StatelessDropOutGenMask] tiling start.");
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context,platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<StatelessDropOutGenMaskCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context,compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StatelessDropOutGenMask)
    .Tiling(Tiling4StatelessDropOutGenMask)
    .TilingParse<StatelessDropOutGenMaskCompileInfo>(TilingPrepare4StatelessDropOutGenMask)
    .TilingInputsDataDependency({ IN_SHAPE_IDX, IN_PROB_IDX, IN_SEED_IDX, IN_SEED1_IDX, IN_OFFSET_IDX });
} // namespace optiling
