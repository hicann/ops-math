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
 * \file tensor_redirect_tiling_arch35.cpp
 * \brief
 */

#include "tensor_redirect_tiling_arch35.h"

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <type_traits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/tensor_redirect_tiling_data.h"
#include "../../op_kernel/arch35/tensor_redirect_tiling_key.h"

using namespace ge;

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;

constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_OUTPUT_X = 0;
constexpr int64_t UB_FACTOR_MIN_BYTES = 2048; // UB 单块性能下界（字节）
constexpr int64_t N_BUFFER = 2;               // double buffer
constexpr int64_t ONE_BLK_BYTE = 32;          // ubblock_size
constexpr int64_t DATA_COPY_MAX_BLOCK_BYTES = 2097151;
constexpr size_t MIN_RANK = 1; // spec inputs[0].rank_range
constexpr size_t MAX_RANK = 8;

static_assert(std::is_trivial<TensorRedirectTilingData>::value,
              "TensorRedirectTilingData must remain trivial for byte initialization and serialization");
static_assert(std::is_standard_layout<TensorRedirectTilingData>::value,
              "TensorRedirectTilingData must remain standard-layout");

static int64_t GetRemainder(int64_t uValue, int64_t dValue)
{
    if (dValue == 0) {
        return uValue;
    }
    return uValue % dValue;
}

// 调用方已保证 dividend/divisor 均为正数；商加非零余数避免 dividend + divisor - 1 溢出。
static int64_t CeilDivPositive(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

// dtype 校验
static ge::graphStatus CheckTensorRedirectDtype(const gert::TilingContext* context)
{
    static const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT8,  ge::DT_INT32,
                                                          ge::DT_UINT8,   ge::DT_INT64,  ge::DT_INT16, ge::DT_UINT16,
                                                          ge::DT_UINT64,  ge::DT_UINT32, ge::DT_BF16};

    auto inputXPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(supportedDtype.count(xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    "TensorRedirect", "x", Ops::Base::ToString(xDtype).c_str(),
                    "dtype must be one of [DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, "
                    "DT_INT16, DT_UINT16, DT_UINT64, DT_UINT32, DT_BF16]"),
                return ge::GRAPH_FAILED);

    auto outputXPtr = context->GetOutputDesc(INDEX_OUTPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputXPtr);
    auto yDtype = outputXPtr->GetDataType();
    OP_CHECK_IF(yDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    "TensorRedirect", "x and output_x",
                    (Ops::Base::ToString(xDtype) + " and " + Ops::Base::ToString(yDtype)).c_str(),
                    "x and output_x must have the same dtype"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRank(const gert::TilingContext* context, const char* tensorName, const char* shapeKind,
                                 const gert::Shape& shape)
{
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(
        rank < MIN_RANK || rank > MAX_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), tensorName, std::to_string(rank).c_str(),
                                                 (std::string(shapeKind) + " rank must be within [1, 8]").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// rank + shape 校验
static ge::graphStatus CheckTensorRedirectShape(const gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xOriginShape = xShapePtr->GetOriginShape();
    const gert::Shape& xShape = xShapePtr->GetStorageShape();

    auto yShapePtr = context->GetOutputShape(INDEX_OUTPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    const gert::Shape& yOriginShape = yShapePtr->GetOriginShape();
    const gert::Shape& yShape = yShapePtr->GetStorageShape();

    if (CheckRank(context, "x", "origin", xOriginShape) != ge::GRAPH_SUCCESS ||
        CheckRank(context, "x", "storage", xShape) != ge::GRAPH_SUCCESS ||
        CheckRank(context, "output_x", "origin", yOriginShape) != ge::GRAPH_SUCCESS ||
        CheckRank(context, "output_x", "storage", yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // CheckDim: Tiling 是外部输入边界，concrete shape 的每一维必须非负；
    // -1/-2 等动态占位符或非法负值不能进入切分计算
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(
            xShape.GetDim(i) < 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("TensorRedirect", "x", std::to_string(xShape.GetDim(i)).c_str(),
                                                     "every dim of the concrete shape must be non-negative"),
            return ge::GRAPH_FAILED);
    }

    // CheckShape: output_x.shape == x.shape（逐维严格相等）
    OP_CHECK_IF(xShape != yShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "TensorRedirect", "x and output_x",
                    (Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(yShape)).c_str(),
                    "x and output_x must have the same shape"),
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

// 多核切分：由 ubFactor 反推 uo/usedCoreNum/blockFactor
static void CalcBlockFactor(TensorRedirectTilingParam& tilingParam, int64_t numel)
{
    tilingParam.uo = CeilDivPositive(numel, tilingParam.ubFactor);
    tilingParam.tailBlockTailUbFactor = GetRemainder(numel, tilingParam.ubFactor);

    // 先按总核数求每核块数，再反推实际使用核数。尾核块数用余数表达，
    // 避免 (usedCoreNum - 1) * blockFactor 在极大 shape 下产生有符号乘法溢出。
    tilingParam.blockFactor = CeilDivPositive(tilingParam.uo, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = CeilDivPositive(tilingParam.uo, tilingParam.blockFactor);
    tilingParam.tailBlockFactor = GetRemainder(tilingParam.uo, tilingParam.blockFactor);
    if (tilingParam.tailBlockFactor == 0) {
        tilingParam.tailBlockFactor = tilingParam.blockFactor;
    }
    if (tilingParam.tailBlockTailUbFactor == 0) {
        tilingParam.tailBlockTailUbFactor = tilingParam.ubFactor;
    }
}

static ge::graphStatus ValidateTilingResult(const gert::TilingContext* context,
                                            const TensorRedirectTilingParam& tilingParam, int64_t maxUbAvailable)
{
    // 对合法正数输入，以下界限由 CeilDiv/余数公式必然成立；该分支仅用于阻断算法回归或损坏的派生值。
    const bool invalidCore = tilingParam.usedCoreNum <= 0 || tilingParam.usedCoreNum > tilingParam.totalCoreNum ||
                             tilingParam.usedCoreNum > static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    const bool invalidBlock = tilingParam.uo <= 0 || tilingParam.blockFactor <= 0 || tilingParam.tailBlockFactor <= 0 ||
                              tilingParam.tailBlockFactor > tilingParam.blockFactor;
    const bool invalidUb = tilingParam.ubFactor <= 0 || tilingParam.ubFactor > maxUbAvailable ||
                           tilingParam.tailBlockTailUbFactor <= 0 ||
                           tilingParam.tailBlockTailUbFactor > tilingParam.ubFactor;
    OP_CHECK_IF(
        invalidCore || invalidBlock || invalidUb,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "derived tiling parameters", "out of range",
                                              "core, block and UB factors must satisfy their safety bounds"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, TensorRedirectTilingParam& tilingParam,
                                int64_t numel)
{
    OP_CHECK_IF(
        numel <= 0 || tilingParam.totalCoreNum <= 0 || tilingParam.bytesForOneData <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "numel,totalCoreNum,bytesForOneData",
                                              (std::to_string(numel) + "," + std::to_string(tilingParam.totalCoreNum) +
                                               "," + std::to_string(tilingParam.bytesForOneData))
                                                  .c_str(),
                                              "all non-empty tiling inputs must be positive"),
        return ge::GRAPH_FAILED);

    // InitBuffer 会将每个槽位向上对齐到 32B，因此先将单槽可用 UB 向下对齐。
    // DataCopyPad::blockLen 在 Ascend950PR 上最大为 2097151B，再取 32B 对齐上界，
    // 使任何满块/尾块都同时满足 UB 容量和搬运 API 参数范围。
    const int64_t slotUbBytes = tilingParam.ubSize / N_BUFFER / ONE_BLK_BYTE * ONE_BLK_BYTE;
    const int64_t maxDataCopyBytes = DATA_COPY_MAX_BLOCK_BYTES / ONE_BLK_BYTE * ONE_BLK_BYTE;
    const int64_t maxBufferBytes = std::min(slotUbBytes, maxDataCopyBytes);
    const int64_t maxUbAvailable = maxBufferBytes / tilingParam.bytesForOneData;
    OP_CHECK_IF(maxUbAvailable <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "maxUbAvailable", "0",
                                                      "ubSize too small for N_BUFFER * bytesForOneData"),
                return ge::GRAPH_FAILED);

    tilingParam.ubFactor = (numel >= maxUbAvailable) ? maxUbAvailable : numel;
    CalcBlockFactor(tilingParam, numel);

    // 核已用满，或每核多次循环 -> 无需提核优化
    if (tilingParam.usedCoreNum == tilingParam.totalCoreNum || tilingParam.blockFactor > 1) {
        return ValidateTilingResult(context, tilingParam, maxUbAvailable);
    }

    // 单核守卫：totalCoreNum==1 时跳过提核优化
    if (tilingParam.totalCoreNum <= 1) {
        return ValidateTilingResult(context, tilingParam, maxUbAvailable);
    }

    // 小 shape 提核优化：核未用满且每核仅一次循环时，缩小 ubFactor 以摊到更多核
    if (GetRemainder(numel, tilingParam.totalCoreNum) == 0) {
        tilingParam.ubFactor = FloorDiv(numel, tilingParam.totalCoreNum);
    } else {
        tilingParam.ubFactor = FloorDiv(numel, tilingParam.totalCoreNum - 1);
    }
    // 32B 对齐
    tilingParam.ubFactor = CeilAlign(tilingParam.ubFactor, ONE_BLK_BYTE / tilingParam.bytesForOneData);
    // 先钳制上界，避免向上对齐越过 UB/DataCopy 容量；性能下界不得反向突破该安全上界。
    tilingParam.ubFactor = std::min(tilingParam.ubFactor, maxUbAvailable);
    int64_t ubFactorMin = std::min(UB_FACTOR_MIN_BYTES / tilingParam.bytesForOneData, maxUbAvailable);
    tilingParam.ubFactor = tilingParam.ubFactor < ubFactorMin ? ubFactorMin : tilingParam.ubFactor;
    CalcBlockFactor(tilingParam, numel);
    return ValidateTilingResult(context, tilingParam, maxUbAvailable);
}

// 获取平台参数（核数/UB 容量），CompileInfo 优先，缺失时回退查 platform
static ge::graphStatus GetPlatformParams(const gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    auto compileInfo = context->GetCompileInfo<TensorRedirectCompileInfo>();
    if (compileInfo != nullptr && compileInfo->coreNum > 0 && compileInfo->ubSize > 0) {
        coreNum = compileInfo->coreNum; // GE 图路径
        ubSize = compileInfo->ubSize;
        return ge::GRAPH_SUCCESS;
    }

    auto platformInfoPtr = context->GetPlatformInfo(); // ACLNN 单算子路径回退
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv(); // AIV_ONLY -> 取 AIV 核数
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "platform vector core count must be greater than 0"),
        return ge::GRAPH_FAILED);

    uint64_t ubSizeTmp = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTmp);
    ubSize = static_cast<int64_t>(ubSizeTmp);
    OP_CHECK_IF(ubSize <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "platform UB size must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    auto compileInfo = context->GetCompileInfo<TensorRedirectCompileInfo>();
    if (compileInfo != nullptr && compileInfo->libApiWorkspaceSize > 0) {
        workspaces[0] = static_cast<size_t>(compileInfo->libApiWorkspaceSize);
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaces[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

// 空 Tensor 防护：在 GetShapeSize() 之前按维检查，避免 [INT64_MAX, 2, 0]
// 这类合法空 shape 因前缀乘法溢出而被误拒。
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context)
{
    TensorRedirectTilingData* tiling = context->GetTilingData<TensorRedirectTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TensorRedirectTilingData), 0, sizeof(TensorRedirectTilingData)) != EOK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tiling data", "not initialized",
                                                      "memset_s failed for empty tensor tiling data"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", "unavailable",
                                                      "set workspace failed for empty tensor"),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(1);
    ASCENDC_TPL_SEL_PARAM(context, TPL_SCH_MODE_0);
    return ge::GRAPH_SUCCESS; // 不下发有效计算
}

static ge::graphStatus Tiling4TensorRedirect(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("TensorRedirect", "tiling context", "must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter Tiling4TensorRedirect.");

    OP_CHECK_IF(CheckTensorRedirectDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input/output dtype", "invalid",
                                                      "dtype validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorRedirectShape(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input/output shape", "invalid",
                                                      "shape validation failed"),
                return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();

    // GetShapeSize() 逐维相乘并在溢出时立即返回，末尾的 0 无法消除此前的溢出。
    // shape/dtype 已完整校验，因此先识别任意位置的 0，再进入非空 numel 计算。
    if (HasZeroDim(xShape)) {
        OP_LOGD(context, "TensorRedirect: empty tensor, skip kernel computation.");
        return HandleEmptyTensor(context);
    }

    // 1D 线性展平，不解释 stride/rank
    int64_t numel = xShape.GetShapeSize();

    // 溢出防护：GetShapeSize() 在维度乘积溢出 int64_t 时返回 kInvalidDimValue，不会自行报错。
    // 必须在 numel == 0 判断之前拦截，否则负的 numel 会穿透到 DoTiling 产生 usedCoreNum == 0。
    OP_CHECK_IF(numel < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("TensorRedirect", "x",
                                                      Ops::Base::ToString(xShapePtr->GetStorageShape()).c_str(),
                                                      "the product of all dims overflows int64_t"),
                return ge::GRAPH_FAILED);

    TensorRedirectTilingParam tilingParam;
    OP_CHECK_IF(GetPlatformParams(context, tilingParam.totalCoreNum, tilingParam.ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform parameters", "unavailable",
                                                      "failed to get core count or UB size"),
                return ge::GRAPH_FAILED);

    auto inputXPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    tilingParam.bytesForOneData = ge::GetSizeByDataType(inputXPtr->GetDataType());
    OP_CHECK_IF(tilingParam.bytesForOneData <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "bytesForOneData", "0",
                                                      "failed to get the size of dtype"),
                return ge::GRAPH_FAILED);
    // Kernel 的 GM 偏移与 TilingData 均使用 int64_t。先按除法判界，不实际执行 numel * bytes，
    // 防止总字节数溢出后指针偏移回绕。
    OP_CHECK_IF(numel > std::numeric_limits<int64_t>::max() / tilingParam.bytesForOneData,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("TensorRedirect", "x", Ops::Base::ToString(xShape).c_str(),
                                                      "the tensor byte size overflows int64_t"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoTiling(context, tilingParam, numel) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tiling parameters", "invalid",
                                                      "tiling calculation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", "unavailable",
                                                      "set workspace failed"),
                return ge::GRAPH_FAILED);

    // TilingData 缓冲区容量由 kernel 侧 REGISTER_TILING_DEFAULT 决定；
    // GetTilingData<T>() 内部已调用 SetDataSize(sizeof(T))，此处无需手工设置。
    TensorRedirectTilingData* tiling = context->GetTilingData<TensorRedirectTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TensorRedirectTilingData), 0, sizeof(TensorRedirectTilingData)) != EOK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tiling data", "not initialized",
                                                      "memset_s failed for tiling data"),
                return ge::GRAPH_FAILED);

    tiling->usedCoreNum = tilingParam.usedCoreNum;
    tiling->blockFactor = tilingParam.blockFactor;
    tiling->tailBlockFactor = tilingParam.tailBlockFactor;
    tiling->ubFactor = tilingParam.ubFactor;
    tiling->tailBlockTailUbFactor = tilingParam.tailBlockTailUbFactor;

    context->SetBlockDim(tilingParam.usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, TPL_SCH_MODE_0);

    OP_LOGD(context,
            "TensorRedirect tilingData: usedCoreNum:%lld, ubFactor:%lld, tailBlockTailUbFactor:%lld, "
            "blockFactor:%lld, tailBlockFactor:%lld",
            static_cast<long long>(tiling->usedCoreNum), static_cast<long long>(tiling->ubFactor),
            static_cast<long long>(tiling->tailBlockTailUbFactor), static_cast<long long>(tiling->blockFactor),
            static_cast<long long>(tiling->tailBlockFactor));
    return ge::GRAPH_SUCCESS;
}

// CompileInfo 来自 platform
static ge::graphStatus TilingPrepare4TensorRedirect(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("TensorRedirect", "tiling parse context", "must not be nullptr"),
        return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<TensorRedirectCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv(); // AIV_ONLY -> 取 AIV 核数
    OP_CHECK_IF(compileInfo->coreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum",
                                                      std::to_string(compileInfo->coreNum).c_str(),
                                                      "platform vector core count must be greater than 0"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(compileInfo->ubSize <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
                                                      std::to_string(compileInfo->ubSize).c_str(),
                                                      "platform UB size must be greater than 0"),
                return ge::GRAPH_FAILED);
    compileInfo->libApiWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TensorRedirect)
    .Tiling(Tiling4TensorRedirect)
    .TilingParse<TensorRedirectCompileInfo>(TilingPrepare4TensorRedirect);

} // namespace optiling
