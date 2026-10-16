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
 * \file radix_top_k_tiling.cpp
 * \brief
 */

#include <cstdint>
#include <limits>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "radix_top_k_tiling.h"
#include "../op_kernel/radix_top_k_struct.h"
#include "../op_kernel/radix_top_k_tiling_key.h"

namespace optiling {
namespace RadixTopK {
static constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t CACHE_LINE = 512;
static constexpr uint64_t MAX_TILE_LEN = 12032;
static constexpr uint64_t LARGE_SORT_LEN = 1440000000;
static constexpr uint64_t LARGE_TILE_LEN = 10496;
static constexpr uint64_t MAX_TILE_NUM_IN_UB = 6144;
// sortLen 超过 int32 上限时切换 int64 模板（索引值域与跨核计数可达 sortLen）
static constexpr uint64_t INT32_SORT_LEN_MAX = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
// isInt64Shape_（sortLen > INT32_SORT_LEN_MAX）必须蕴含 isLargeShape_（sortLen > LARGE_SORT_LEN）：
// needWorkspace / GetWorkspaceSize 两处都写成 `isLargeShape_ || isInt64Shape_`（后者冗余）。这里用
// static_assert 把两个常量的相对大小钉死，避免以后调大 LARGE_SORT_LEN 时 i64 分支被漏掉。
static_assert(LARGE_SORT_LEN < INT32_SORT_LEN_MAX,
              "isInt64Shape_ must imply isLargeShape_: LARGE_SORT_LEN must stay below INT32_SORT_LEN_MAX");
// checkUb 中 UB 占用按「每 tile 元素数 × 系数」估算：
//   int32 索引：xBuf ping/pong(fp32) + outValue(fp32) + outIndex(int32) = 16 元素/tile；
//   int64 索引：outIndexBuf 翻倍（int64），系数 16 -> 20。
static constexpr uint64_t UB_COEF_PER_TILE = 16;
static constexpr uint64_t UB_COEF_PER_TILE_I64 = 20;
static constexpr uint64_t INPUT_X = 0;
static constexpr uint64_t NUM_VALUE_2BIT = 4;
static constexpr uint64_t BIT_PER_BYTE = 8;
static constexpr uint64_t BUFFER_NUM_IN_OUT = 4;
static constexpr uint64_t ATTR_INDEX_SORTED = 0;
static constexpr uint64_t ATTR_INDEX_LARGEST = 2;
static constexpr uint64_t WS_INT32_PER_TILE = 5;
// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus RadixTopKTiling::GetPlatformInfo()
{
    OP_LOGD(opName_, "RadixTopKTiling GetPlatformInfo.");
    auto compileInfo = static_cast<const RadixTopKCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<uint64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus RadixTopKTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "RadixTopKTiling GetShapeAttrsInfo.");
    // 获取输入shape和dtype
    const gert::StorageShape* xShape = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    uint64_t batch = 1;
    uint64_t dimNum = xShape->GetShape().GetDimNum();
    for (uint64_t i = 0; i < dimNum - 1; i++) {
        batch *= xShape->GetShape().GetDim(i);
    }
    tilingData_.batch = batch;
    tilingData_.sortLen = xShape->GetShape().GetDim(dimNum - 1);

    auto tensorK = context_->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorK);
    const int32_t* constDataPtr = tensorK->GetData<int32_t>();
    OP_CHECK_IF(constDataPtr == nullptr, OP_LOGE(opName_, "Get const data k failed."), return ge::GRAPH_FAILED);
    int32_t kValue = static_cast<int32_t>(*constDataPtr);
    OP_CHECK_IF(kValue == 0, OP_LOGE(opName_, "Get k equals to zero."), return ge::GRAPH_FAILED);
    tilingData_.kValue = kValue;

    auto attrs = context_->GetAttrs();
    tilingData_.sorted = *attrs->GetAttrPointer<bool>(ATTR_INDEX_SORTED);
    tilingData_.largest = *attrs->GetAttrPointer<bool>(ATTR_INDEX_LARGEST);
    isLargeShape_ = tilingData_.sortLen > LARGE_SORT_LEN;
    isInt64Shape_ = tilingData_.sortLen > INT32_SORT_LEN_MAX;
    // int64 模板输出 int64 索引，防御性校验输出 dtype 匹配（int32 组合不允许超 int32 shape）
    if (isInt64Shape_) {
        auto indicesDesc = context_->GetOutputDesc(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
        OP_CHECK_IF((indicesDesc->GetDataType() != ge::DT_INT64),
                    OP_LOGE(opName_, "RadixTopK sortLen [%lu] exceeds int32, indices output must be int64.",
                            tilingData_.sortLen),
                    return ge::GRAPH_FAILED);
    }

    xDType_ = context_->GetInputDesc(INPUT_X)->GetDataType();
    xDtypeSize_ = ge::GetSizeByDataType(xDType_);

    return ge::GRAPH_SUCCESS;
}

// 3、计算数据切分TilingData
ge::graphStatus RadixTopKTiling::DoOpTiling()
{
    OP_LOGD(opName_, "RadixTopKTiling DoOpTiling.");
    bool isCalcSuccess = false;
    if (isInt64Shape_) {
        isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen, UB_COEF_PER_TILE_I64);
    } else if (!isLargeShape_) {
        isCalcSuccess = CalcTilingParams(tilingData_.sortLen);
        if (!isCalcSuccess) {
            isLargeShape_ = true;
            isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen, UB_COEF_PER_TILE);
        }
    } else {
        isCalcSuccess = CalcLargeTilingParams(tilingData_.sortLen, UB_COEF_PER_TILE);
    }
    OP_CHECK_IF((!isCalcSuccess), OP_LOGE(opName_, "RadixTopK Tiling failed."), return ge::GRAPH_FAILED);
    // int64 模板的 coreTopK/边界计数等 int32 通道要求 kValue 不超 int32 上限
    OP_CHECK_IF((isInt64Shape_ && tilingData_.kValue > INT32_SORT_LEN_MAX),
                OP_LOGE(opName_, "RadixTopK kValue [%lu] exceeds int32 limit for int64 shape.", tilingData_.kValue),
                return ge::GRAPH_FAILED);
    // per-core 直方图计数保持 int32：per-core 元素数须 < 2^31（S < 2^31*coreNum，物理内存内恒成立，防御性校验）
    OP_CHECK_IF((isInt64Shape_ && tilingData_.formerTileNum * tilingData_.formerTileLen > INT32_SORT_LEN_MAX),
                OP_LOGE(opName_, "RadixTopK per-core element num exceeds int32 limit."), return ge::GRAPH_FAILED);

    blockDim_ = tilingData_.formerCoreNum + tilingData_.tailCoreNum;
    tilingData_.coreNum = blockDim_;
    // 仅 UB 变体可复用 indices 内存（WS/INT64 变体的 tileTopK/tileHist 与 CopyOutResult 跨 core 冲突）
    // 注：isInt64Shape_ ⇒ isLargeShape_（见文件头 static_assert），这里的 `|| isInt64Shape_` 是防御性冗余
    if (isLargeShape_ || isInt64Shape_) {
        tilingData_.needWorkspace = true;
    } else {
        uint64_t wsInt32Cnt = tilingData_.coreNum * (NUM_VALUE_2BIT + 2);
        tilingData_.needWorkspace = (tilingData_.kValue <= wsInt32Cnt);
    }
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus RadixTopKTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

// 5、计算TilingKey
uint64_t RadixTopKTiling::GetTilingKey() const
{
    OP_LOGD(opName_, "RadixTopKTiling GetTilingKey.");

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tilingData_.sorted, tilingData_.largest, isLargeShape_,
                                                  isInt64Shape_);
    OP_LOGD(opName_, "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(opName_, "sorted, largest, isLargeShape, isInt64Shape is: [%d, %d, %d, %d]", tilingData_.sorted,
            tilingData_.largest, isLargeShape_, isInt64Shape_);

    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus RadixTopKTiling::GetWorkspaceSize()
{
    if (!tilingData_.needWorkspace) {
        workspaceSize_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    // 注：同上，isInt64Shape_ 已被 isLargeShape_ 蕴含
    if (isLargeShape_ || isInt64Shape_) {
        workspaceSize_ = (NUM_VALUE_2BIT * tilingData_.coreNum + tilingData_.coreNum + tilingData_.coreNum +
                          WS_INT32_PER_TILE * tilingData_.totalTileNum) *
                         sizeof(int32_t);
    } else {
        workspaceSize_ = (NUM_VALUE_2BIT * tilingData_.coreNum + tilingData_.coreNum + tilingData_.coreNum) *
                         sizeof(int32_t);
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize_ += sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus RadixTopKTiling::PostTiling()
{
    OP_LOGD(opName_, "RadixTopKTiling PostTiling.");

    // 设置workspace大小
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    // 用到SyncAll，需要设置为batch mode模式
    context_->SetScheduleMode(1);

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(RadixTopKTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(RadixTopKTilingData));

    return ge::GRAPH_SUCCESS;
}

bool RadixTopKTiling::IsCapable() { return true; }

void RadixTopKTiling::PrintTilingData()
{
    OP_LOGD(opName_, "formerCoreNum:    %lu.", tilingData_.formerCoreNum);
    OP_LOGD(opName_, "tailCoreNum:      %lu.", tilingData_.tailCoreNum);
    OP_LOGD(opName_, "totalTileNum:     %lu.", tilingData_.totalTileNum);
    OP_LOGD(opName_, "formerTileNum:    %lu.", tilingData_.formerTileNum);
    OP_LOGD(opName_, "tailTileNum:      %lu.", tilingData_.tailTileNum);
    OP_LOGD(opName_, "formerTileLen:    %lu.", tilingData_.formerTileLen);
    OP_LOGD(opName_, "tailTileLen:      %lu.", tilingData_.tailTileLen);
    OP_LOGD(opName_, "batch:            %lu.", tilingData_.batch);
    OP_LOGD(opName_, "sortLen:          %lu.", tilingData_.sortLen);
    OP_LOGD(opName_, "kValue:           %lu.", tilingData_.kValue);
    OP_LOGD(opName_, "largest:          %d.", tilingData_.largest);
    OP_LOGD(opName_, "sorted:           %d.", tilingData_.sorted);
    OP_LOGD(opName_, "isInt64Shape:     %d.", isInt64Shape_);
    OP_LOGD(opName_, "needWorkspace:    %d.", tilingData_.needWorkspace);
}

void RadixTopKTiling::CalcTileDistribution(const uint64_t& dataNum, const uint64_t& formerTileLen)
{
    uint64_t totalTileNum = Ops::Base::CeilDiv(dataNum, formerTileLen);
    uint64_t formerTileNum = Ops::Base::CeilDiv(totalTileNum, totalCoreNum_);
    uint64_t tailTileNum = totalTileNum / totalCoreNum_;
    uint64_t formerCoreNum = totalTileNum % totalCoreNum_;
    uint64_t tailCoreNum = tailTileNum ? (totalCoreNum_ - formerCoreNum) : 0;
    uint64_t totalProcessedTiles = formerTileNum * formerCoreNum + tailTileNum * tailCoreNum;
    uint64_t tailTileLen = dataNum - formerTileLen * (totalProcessedTiles - 1);

    tilingData_.formerCoreNum = formerCoreNum;
    tilingData_.tailCoreNum = tailCoreNum;
    tilingData_.totalTileNum = totalTileNum;
    tilingData_.formerTileNum = formerTileNum;
    tilingData_.tailTileNum = tailTileNum;
    tilingData_.formerTileLen = formerTileLen;
    tilingData_.tailTileLen = tailTileLen;
}

template <typename CheckUbFn>
bool RadixTopKTiling::TryCalcTileDistribution(const uint64_t& dataNum, uint64_t startTileLen, uint64_t minTileLen,
                                              uint64_t step, CheckUbFn&& checkUb)
{
    for (uint64_t tileLen = startTileLen; tileLen >= minTileLen; tileLen -= step) {
        if (checkUb(tileLen)) {
            CalcTileDistribution(dataNum, tileLen);
            return true;
        }
    }
    OP_LOGD(opName_, "Cannot find suitable tileLen to do RadixTopK.");
    return false;
}

/**
 * @brief 大 shape / int64 shape 的 tile 切分（两者仅 UB 系数不同，合并为一个带参函数）
 * @param dataNum 单 core 处理的数据量
 * @param ubCoefPerTile UB 占用中「每 tile 元素数」的系数（int32 索引 16、int64 索引 20）
 */
bool RadixTopKTiling::CalcLargeTilingParams(const uint64_t& dataNum, const uint64_t& ubCoefPerTile)
{
    uint64_t dataAlign = CACHE_LINE / xDtypeSize_;
    uint64_t minTileLen = dataAlign;

    auto checkUb = [this, ubCoefPerTile](uint64_t tileLen) -> bool {
        uint64_t cmpMaskSize = tileLen / 8;
        uint64_t ubUsed = ubCoefPerTile * tileLen + cmpMaskSize + MAX_TILE_NUM_IN_UB * sizeof(int32_t) + BLOCK_SIZE;
        return ubUsed <= ubSize_;
    };

    return TryCalcTileDistribution(dataNum, LARGE_TILE_LEN, minTileLen, dataAlign, checkUb);
}

bool RadixTopKTiling::CalcTilingParams(const uint64_t& dataNum)
{
    uint64_t coreNum = totalCoreNum_;
    uint64_t dataAlign = CACHE_LINE / xDtypeSize_;
    uint64_t minTileLen = dataAlign;

    auto checkUb = [this, dataNum, coreNum](uint64_t tileLen) -> bool {
        if (tileLen == 0)
            return false;
        uint64_t totalTn = (dataNum + tileLen - 1) / tileLen;
        if (totalTn == 0)
            totalTn = 1;
        uint64_t tn = (totalTn + coreNum - 1) / coreNum;
        uint64_t tnAlign = Ops::Base::CeilAlign(tn, static_cast<uint64_t>(8));
        uint64_t tileHistSize = NUM_VALUE_2BIT * tnAlign * sizeof(int32_t);
        uint64_t cmpMaskSize = tileLen / 8;
        uint64_t tempBuf = (tileHistSize > cmpMaskSize) ? tileHistSize : cmpMaskSize;
        uint64_t ubUsed = BUFFER_NUM_IN_OUT * sizeof(int32_t) * tileLen + NUM_VALUE_2BIT * tnAlign + tempBuf +
                          BLOCK_SIZE;
        return ubUsed <= ubSize_;
    };

    uint64_t startTileLen = MAX_TILE_LEN;
    uint64_t utilCap = Ops::Base::CeilAlign(Ops::Base::CeilDiv(dataNum, coreNum), dataAlign);
    if (utilCap < startTileLen) {
        startTileLen = utilCap;
    }

    return TryCalcTileDistribution(dataNum, startTileLen, minTileLen, dataAlign, checkUb);
}

void RadixTopKTiling::SetTilingData()
{
    OP_LOGD(opName_, "RadixTopKTiling SetTilingData.");
    RadixTopKTilingData* tilingData = context_->GetTilingData<RadixTopKTilingData>();
    tilingData->formerCoreNum = tilingData_.formerCoreNum;
    tilingData->tailCoreNum = tilingData_.tailCoreNum;
    tilingData->totalTileNum = tilingData_.totalTileNum;
    tilingData->formerTileNum = tilingData_.formerTileNum;
    tilingData->tailTileNum = tilingData_.tailTileNum;
    tilingData->formerTileLen = tilingData_.formerTileLen;
    tilingData->tailTileLen = tilingData_.tailTileLen;
    tilingData->coreNum = tilingData_.coreNum;
    tilingData->batch = tilingData_.batch;
    tilingData->sortLen = tilingData_.sortLen;
    tilingData->kValue = tilingData_.kValue;
    tilingData->sorted = tilingData_.sorted;
    tilingData->largest = tilingData_.largest;
    tilingData->needWorkspace = tilingData_.needWorkspace;
}
} // namespace RadixTopK

static ge::graphStatus TilingRadixTopK(gert::TilingContext* context)
{
    OP_LOGD(context, "RadixTopKTiling start.");
    RadixTopK::RadixTopKTiling tilingOp(context);
    auto ret = tilingOp.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(context, "RadixTopKTiling tiling failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "RadixTopKTiling end.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForRadixTopK(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For RadixTopK start.");
    auto compileInfo = context->GetCompiledInfo<RadixTopKCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum == 0) {
        OP_LOGE(context, "coreNum %d", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For RadixTopK end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RadixTopK)
    .Tiling(TilingRadixTopK)
    .TilingParse<RadixTopKCompileInfo>(TilingPrepareForRadixTopK)
    .TilingInputsDataDependency({1});
} // namespace optiling
