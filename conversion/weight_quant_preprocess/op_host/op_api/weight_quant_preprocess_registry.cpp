/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "weight_quant_preprocess_registry.h"

#include <inttypes.h>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

#define LOGE_WITH_SCENARIO(err, fmt, ...)                                              \
    OP_LOGE(err, "[NpuArch=%u, DataFlow=%s] " fmt, static_cast<uint32_t>(ctx.npuArch), \
            QuantDataFlowToString(ctx.dataFlow), ##__VA_ARGS__)

namespace {

static constexpr size_t DIMS_1 = 1;
static constexpr size_t DIMS_2 = 2;
static constexpr size_t DIMS_3 = 3;
static constexpr size_t DIMS_4 = 4;
static constexpr size_t IDX_0 = 0;
static constexpr size_t IDX_1 = 1;
static constexpr size_t IDX_2 = 2;
static constexpr size_t IDX_3 = 3;
static constexpr size_t IDX_4 = 4;
static constexpr size_t DOUBLE = 2;
static constexpr int64_t KGROUP_SIZE_MX = 32;
static constexpr int64_t NZ_16 = 16;
static constexpr int64_t NZ_C0_16 = 16;
static constexpr int64_t NZ_C0_32 = 32;
static constexpr int64_t B4_NUMS_PER_BYTE = 2; // 1 字节打包 2 个 4-bit 值（INT4/FP4 通用）

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CeilDiv divisor b must not be zero.");
        return INT64_MIN;
    }
    return (a + b - 1) / b;
}

static bool IsMMMxA8W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();

    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        weightViewShape.GetDimNum() == DIMS_2) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A8W4;
        return true;
    }
    return false;
}

static bool IsGMMMxA8W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();

    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        weightViewShape.GetDimNum() == DIMS_3) {
        ctx.dataFlow = QuantDataFlow::GMM_MX_A8W4;
        return true;
    }
    return false;
}

// ===== A16S4/A16F4 judge 判定件 =====
// judge 按 scale 模式 × 转置 × 出 format 拆分，以下判定件负责组装；dtype 通过模板参数传入
// x dtype fp16/bf16 + weight 2D；A16W4 无 xScale 语义，xScale 由各 judge 以 IsXScaleUndefined 统一排除
template <op::DataType WType>
static bool IsA16W4Base(const QuantContext& ctx)
{
    bool xDtypeMatch = (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16);
    return ctx.weight->GetDataType() == WType && xDtypeMatch && ctx.weight->GetViewShape().GetDimNum() == DIMS_2;
}

static bool IsXScaleUndefined(const QuantContext& ctx) { return ctx.xScaleDtype == op::DataType::DT_UNDEFINED; }

// 转置探测：最后两维 strides [1, size(-2)] 即转置（与 CheckWeightTrans 同一判定）
static bool IsWeightLastTwoDimsTrans(const QuantContext& ctx)
{
    const auto& viewShape = ctx.weight->GetViewShape();
    const auto& viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    return viewStrides[dimNum - IDX_2] == 1 && viewStrides[dimNum - IDX_1] == viewShape.GetDim(dimNum - IDX_2);
}

// scale 模式判定：per-tensor 单元素 / per-channel 1D 或 [1,N] 且 numel>1 / per-group 2D 且 G>1
static bool IsScalePerTensor(const QuantContext& ctx) { return ctx.weightScale->GetViewShape().GetShapeSize() == 1; }

static bool IsScalePerChannel(const QuantContext& ctx)
{
    const auto& scaleViewShape = ctx.weightScale->GetViewShape();
    int64_t scaleDim = static_cast<int64_t>(scaleViewShape.GetDimNum());
    return (scaleDim == DIMS_1 || (scaleDim == DIMS_2 && scaleViewShape.GetDim(IDX_0) == 1)) &&
           scaleViewShape.GetShapeSize() > 1;
}

static bool IsScalePerGroup(const QuantContext& ctx)
{
    const auto& scaleViewShape = ctx.weightScale->GetViewShape();
    return scaleViewShape.GetDimNum() == DIMS_2 && scaleViewShape.GetDim(IDX_0) > 1;
}

// out format 判定：NZ_C0_16 分形转换；ND/NCL 出（直拷）由转置 judge 命中，无需单独判定
static bool IsOutWeightNzC016(const QuantContext& ctx)
{
    return ctx.outWeight != nullptr && ctx.outWeight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ_C0_16;
}

// A16S4 per-tensor：转置不验证（ND 直拷物理透传与转置无关），仅 ND 出
static bool IsMMA16S4PerTensorDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerTensor(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERTENSOR;
        return true;
    }
    return false;
}

// A16S4 per-channel：转置 → ND 直拷；非转置仅支持 NZ_C0_16 出（ND 出不支持，与 A16F4 对齐）
static bool IsMMA16S4PerChannelTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerChannelNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

// A16S4 per-group：同 per-channel 的两路拆分（转置 ND 直拷 / 非转置 NZ_C0_16 转换）
static bool IsMMA16S4PerGroupTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerGroupNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

// A16F4 per-group：FP4 weight + per-group scale [G, N]（G > 1）；NZ only，转置由 checks 的
// CheckWeightNotTrans 拒绝（保留明确报错）
static bool IsMMA16F4PerGroupDataFlow(QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    bool scaleDtypeMatch = (scaleDtype == op::DataType::DT_FLOAT16 || scaleDtype == op::DataType::DT_BF16);

    if (IsA16W4Base<op::DataType::DT_FLOAT4_E2M1>(ctx) && scaleDtypeMatch && IsXScaleUndefined(ctx) &&
        IsScalePerGroup(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16F4_PERGROUP;
        return true;
    }
    return false;
}

// A16MXFP4 公共判定：FP4 weight + MX scale（E8M0，2D [K/32, N]，与 wqbmmv2 MX kernel 约定一致）
static bool IsMMA16MXF4Base(const QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();

    return IsA16W4Base<op::DataType::DT_FLOAT4_E2M1>(ctx) && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
           IsXScaleUndefined(ctx) && ctx.weightScale->GetViewShape().GetDimNum() == DIMS_2;
}

// A16MXFP4 转置（末两维严格转置）→ ND 直拷（wqbmmv2 MX kernel 支持 ND 转置输入）
static bool IsMMA16MXF4TransDataFlow(QuantContext& ctx)
{
    if (IsMMA16MXF4Base(ctx) && IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

// A16MXFP4 非转置 + NZ_C0_16 出 → ND→NZ 分形转换（NZ 出仅支持非转置，转置由转置条目的
// OUT_WEIGHT_ND_CHECKS 拒绝）
static bool IsMMA16MXF4NonTransNzDataFlow(QuantContext& ctx)
{
    if (IsMMA16MXF4Base(ctx) && !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

static aclnnStatus CheckWeightNotEmpty(const QuantContext& ctx)
{
    OP_CHECK(
        !ctx.weight->IsEmpty(),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "weight_quant_preprocess", "weight", std::to_string(ctx.weight->GetViewShape().GetShapeSize()).c_str(),
            (std::string("weight must not be empty tensor when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightFormatND(const QuantContext& ctx)
{
    auto weightFormat = ctx.weight->GetStorageFormat();
    OP_CHECK(weightFormat == op::Format::FORMAT_ND || weightFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "weight", op::ToString(weightFormat).GetString(),
                                        "ND or NCL"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(
        viewStrides[secondLastIdx] == 1 && viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx),
        OP_LOGE_FOR_INVALID_STRIDE(
            "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
            (std::string("last two dims stride [1, ") + std::to_string(viewShape.GetDim(secondLastIdx)) + "]").c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// A16S4 NZ 路径（per-channel / per-group）仅支持非转置 weight，转置返回错误
static aclnnStatus CheckWeightNotTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(viewStrides[lastIdx] == 1 && viewStrides[secondLastIdx] == viewShape.GetDim(lastIdx),
             OP_LOGE_FOR_INVALID_STRIDE(
                 "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
                 (std::string("transposed weight is not supported for ") + QuantDataFlowToString(ctx.dataFlow) +
                  ", last two dims stride should be [" + std::to_string(viewShape.GetDim(lastIdx)) + ", 1]")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 紧凑 4-bit（INT4/FP4）每字节打包 2 个值：打包维（连续维）必须为偶数
// 非转置 [K,N] 沿 N 打包 -> N 为偶数；转置 [K,N] strides [1,K] 沿 K 打包 -> K 为偶数
static aclnnStatus CheckWeightPackingDimEven(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;
    bool isTransposed = (viewStrides[secondLastIdx] == 1 && viewShape.GetDim(secondLastIdx) > 1);
    int64_t packingDim = isTransposed ? viewShape.GetDim(secondLastIdx) : viewShape.GetDim(lastIdx);

    OP_CHECK(packingDim % B4_NUMS_PER_BYTE == 0,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "weight", op::ToString(viewShape).GetString(),
                 (std::string("the packing dim of 4-bit weight must be even when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 4-bit 直拷（ProcessWeightDirectCopy）以 GetViewOffset()/2 建 UINT8 打包视图按字节物理透传：
// weight 视图必须为连续或末两维严格转置（其余 strides 模式会按错误的打包维寻址），
// 且 weight/outWeight 的 viewOffset 须为偶数（奇数偏移除以 2 截断后错位半字节）
static aclnnStatus CheckWeight4BitDirectCopyView(const QuantContext& ctx)
{
    if (ctx.weight->GetDataType() != op::DataType::DT_INT4 &&
        ctx.weight->GetDataType() != op::DataType::DT_FLOAT4_E2M1) {
        return ACLNN_SUCCESS;
    }
    // outWeight 为空属参数错误，由后续 CheckOutWeightSameAsInput/CheckOutWeightSameBase
    // 报告；此处提前返回避免空指针
    if (ctx.outWeight == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;
    bool isStrictTransposed = (viewStrides[secondLastIdx] == 1 && viewShape.GetDim(secondLastIdx) > 1 &&
                               viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx));
    OP_CHECK(IsContiguous(ctx.weight) || isStrictTransposed,
             OP_LOGE_FOR_INVALID_STRIDE(
                 "weight_quant_preprocess", "weight", op::ToString(viewStrides).GetString(),
                 (std::string("contiguous or strictly transposed at last two dims when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        ctx.weight->GetViewOffset() % B4_NUMS_PER_BYTE == 0 && ctx.outWeight->GetViewOffset() % B4_NUMS_PER_BYTE == 0,
        OP_LOGE_FOR_INVALID_VALUE("weight_quant_preprocess", "weight/outWeight viewOffset",
                                  (std::to_string(ctx.weight->GetViewOffset()) + std::string("/") +
                                   std::to_string(ctx.outWeight->GetViewOffset()))
                                      .c_str(),
                                  "even (4-bit packs 2 values per byte, odd offset misaligns the nibble)"),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalNull(const QuantContext& ctx)
{
    OP_CHECK(ctx.weightOffsetOptional == nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "weightOffsetOptional must be nullptr."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(ctx.outWeightOffsetOptional == nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "outWeightOffsetOptional must be nullptr."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNzStorageDim(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto outStorageShape = ctx.outWeight->GetStorageShape();
    OP_CHECK(outStorageShape.GetDimNum() == weightViewShape.GetDimNum() + 2,
             OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "outWeight",
                                          std::to_string(outStorageShape.GetDimNum()).c_str(),
                                          std::to_string(weightViewShape.GetDimNum() + 2).c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <int64_t nzC0, op::Format outWeightFormat>
static aclnnStatus CheckOutWeightTransNz(const QuantContext& ctx)
{
    auto outFormat = ctx.outWeight->GetStorageFormat();
    OP_CHECK(outFormat == outWeightFormat,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeight", op::ToString(outFormat).GetString(),
                                        op::ToString(outWeightFormat).GetString()),
             return ACLNN_ERR_PARAM_INVALID);

    auto outStorageShape = ctx.outWeight->GetStorageShape();
    auto weightViewShape = ctx.weight->GetViewShape();
    size_t outStorageDim = outStorageShape.GetDimNum();
    size_t viewDim = weightViewShape.GetDimNum();

    int64_t k = weightViewShape.GetDim(viewDim - IDX_2);
    int64_t n = weightViewShape.GetDim(viewDim - IDX_1);
    int64_t expectedNBlocks = CeilDiv(n, NZ_16);
    int64_t expectedKBlocks = CeilDiv(k, nzC0);

    // NZ_C0_16（A16S4/A16F4 紧凑 4-bit）物理布局为 [N/16, K/nzC0, 16, nzC0]（N 块在前）；
    // NZ_C0_32（A8W4）保持 master 的 [K/nzC0, N/16, 16, nzC0]（K 块在前）布局约定
    bool nFirst = (outWeightFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16);
    int64_t expectedBlocks4 = nFirst ? expectedNBlocks : expectedKBlocks;
    int64_t expectedBlocks3 = nFirst ? expectedKBlocks : expectedNBlocks;
    const char* layoutDesc = nFirst ? "{ceildiv(N, 16), ceildiv(K, nzC0), 16, nzC0}" :
                                      "{ceildiv(K, nzC0), ceildiv(N, 16), 16, nzC0}";
    OP_CHECK(
        outStorageShape.GetDim(outStorageDim - IDX_4) == expectedBlocks4 &&
            outStorageShape.GetDim(outStorageDim - IDX_3) == expectedBlocks3 &&
            outStorageShape.GetDim(outStorageDim - IDX_2) == NZ_16 &&
            outStorageShape.GetDim(outStorageDim - IDX_1) == nzC0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeight, weight",
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight storage shape last four dims must be ") + layoutDesc +
             std::string(" when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// ===== out 与 input 一致性归一校验 =====
// 轻量一致性：nullptr 一致（都空或都不空）→ 都非空时 out 非 empty → viewShape ==。
// 单独用于 NZ 出 weight 等 format/storageShape/dtype 合法异于输入的场景
static aclnnStatus CheckOutSameBase(const QuantContext& ctx, const aclTensor* input, const aclTensor* output,
                                    const char* inputName, const char* outputName)
{
    OP_CHECK((input == nullptr) == (output == nullptr),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_NULLPTR, "%s and %s must be both nullptr or both not nullptr.",
                                outputName, inputName),
             return ACLNN_ERR_PARAM_NULLPTR);
    if (input == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(!output->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", outputName, std::to_string(output->GetViewShape().GetShapeSize()).c_str(),
                 (std::string(outputName) + " must not be empty tensor when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inViewShape = input->GetViewShape();
    auto outViewShape = output->GetViewShape();
    OP_CHECK(outViewShape == inViewShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outViewShape).GetString() + std::string(", ") + op::ToString(inViewShape).GetString())
                     .c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same viewShape when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// 完整一致性（直拷透传）：Base 之上再要求 format/dtype/storageShape 全 ==、连续性一致
static aclnnStatus CheckOutSameAsInput(const QuantContext& ctx, const aclTensor* input, const aclTensor* output,
                                       const char* inputName, const char* outputName)
{
    auto status = CheckOutSameBase(ctx, input, output, inputName, outputName);
    if (status != ACLNN_SUCCESS || input == nullptr) {
        return status;
    }
    // 连续性一致：ViewCopy 直拷要求源连续，非连续输入 + 连续输出（或反之）在此拦截，
    // 避免错误延后到 process 阶段才以内部错误码暴露（weight 转置直拷两侧同为非连续，不受影响）
    OP_CHECK(
        IsContiguous(input) == IsContiguous(output),
        LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "%s and %s must have the same contiguity.", outputName, inputName),
        return ACLNN_ERR_PARAM_INVALID);
    auto inFormat = input->GetStorageFormat();
    auto outFormat = output->GetStorageFormat();
    OP_CHECK(outFormat == inFormat,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                 "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outFormat).GetString() + std::string(", ") + op::ToString(inFormat).GetString()).c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same format when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inDtype = input->GetDataType();
    auto outDtype = output->GetDataType();
    OP_CHECK(outDtype == inDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outDtype).GetString() + std::string(", ") + op::ToString(inDtype).GetString()).c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same dtype when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inStorageShape = input->GetStorageShape();
    auto outStorageShape = output->GetStorageShape();
    OP_CHECK(
        outStorageShape == inStorageShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(inStorageShape).GetString())
                .c_str(),
            (std::string(outputName) + " and " + inputName + " must have the same storageShape when dataFlow is " +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// out weight ND 直拷：与 weight 完全一致
static aclnnStatus CheckOutWeightSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.weight, ctx.outWeight, "weight", "outWeight");
}

static aclnnStatus CheckOutWeightScaleSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.weightScale, ctx.outWeightScale, "weightScale", "outWeightScale");
}

static aclnnStatus CheckOutWeightOffsetOptionalSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.weightOffsetOptional, ctx.outWeightOffsetOptional, "weightOffsetOptional",
                               "outWeightOffsetOptional");
}

static aclnnStatus CheckOutBiasOptionalSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.biasOptional, ctx.outBiasOptional, "biasOptional", "outBiasOptional");
}

// out weight NZ 出：只要求 nullptr/empty 一致 + viewShape 一致（format/storageShape 合法异于输入；
// dtype 一致性由 CheckOutWeightDtypeSame 单独承担）
static aclnnStatus CheckOutWeightSameBase(const QuantContext& ctx)
{
    return CheckOutSameBase(ctx, ctx.weight, ctx.outWeight, "weight", "outWeight");
}

// out weight NZ 出：dtype 与输入一致（真实场景 NZ 出 dtype 恒等于输入，不一致会在 process ViewCopy 失败）
static aclnnStatus CheckOutWeightDtypeSame(const QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto outWeightDtype = ctx.outWeight->GetDataType();
    OP_CHECK(outWeightDtype == weightDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeight, weight",
                 (op::ToString(outWeightDtype).GetString() + std::string(", ") + op::ToString(weightDtype).GetString())
                     .c_str(),
                 (std::string("outWeight and weight must have the same dtype when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t viewKIdx>
static aclnnStatus ProcessWeightTransNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = ctx.outWeight->GetStorageFormat();

    op::Shape storageShape(viewShape);
    // torch 侧传入的 storageShape 为 1 维，需要手动构造转置后的 storageShape
    std::swap(storageShape[viewKIdx], storageShape[viewKIdx + IDX_1]);

    auto weightTensor = const_cast<aclTensor*>(ctx.weight);
    // TransData 要求输入连续 Tensor，设置所有 Shape 为转置
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);
    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    // TransData 输出 storageFormat 为 (dstFormat, 0) 组合状态，因 ViewCopy 要求完全一致，需要重新设置
    outTensor->SetStorageFormat(dstFormat);
    outTensor->SetViewShape(viewShape);
    // SetViewShape 内部会隐式设置 viewStride 为非转置连续，重新设置 viewStride 为转置非连续
    op::Strides viewStrides(outTensor->GetViewStrides());
    OP_CHECK(static_cast<int64_t>(viewStrides.size()) > static_cast<int64_t>(viewKIdx + IDX_1),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "viewStrides size too small for viewKIdx."),
             return ACLNN_ERR_PARAM_INVALID);
    viewStrides[viewKIdx] = 1;
    viewStrides[viewKIdx + IDX_1] = viewShape.GetDim(viewKIdx);
    outTensor->SetViewStrides(viewStrides);

    auto viewCopyResult = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(viewCopyResult != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR,
                                "ViewCopy failed, outTensor viewShape=%s format=%d, outWeight viewShape=%s format=%d.",
                                op::ToString(outTensor->GetViewShape()).GetString(),
                                static_cast<int>(outTensor->GetStorageFormat()),
                                op::ToString(ctx.outWeight->GetViewShape()).GetString(),
                                static_cast<int>(ctx.outWeight->GetStorageFormat())),
             return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

// 非转置 weight 的 ND 到 NZ 转换（用于 A16INT4 等场景）
template <size_t viewKIdx>
static aclnnStatus ProcessWeightNonTransNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = ctx.outWeight->GetStorageFormat();

    // INT4 和 FP4 一样，传入逻辑 shape [K, N]，不除以 2
    // runtime 会根据 dtype 自动计算物理大小
    op::Shape storageShape(viewShape);

    auto weightTensor = const_cast<aclTensor*>(ctx.weight);
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);

    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    outTensor->SetStorageFormat(dstFormat);
    outTensor->SetViewShape(viewShape);
    op::Strides viewStrides(outTensor->GetViewStrides());
    OP_CHECK(static_cast<int64_t>(viewStrides.size()) > static_cast<int64_t>(viewKIdx + IDX_1),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "viewStrides size too small for viewKIdx."),
             return ACLNN_ERR_PARAM_INVALID);
    viewStrides[viewKIdx] = viewShape.GetDim(viewKIdx + IDX_1);
    viewStrides[viewKIdx + IDX_1] = 1;
    outTensor->SetViewStrides(viewStrides);

    auto viewCopyResult = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(viewCopyResult != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR,
                                "ViewCopy failed, outTensor viewShape=%s format=%d, outWeight viewShape=%s format=%d.",
                                op::ToString(outTensor->GetViewShape()).GetString(),
                                static_cast<int>(outTensor->GetStorageFormat()),
                                op::ToString(ctx.outWeight->GetViewShape()).GetString(),
                                static_cast<int>(ctx.outWeight->GetStorageFormat())),
             return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightScaleDirectCopy(QuantContext& ctx)
{
    auto srcScale = const_cast<aclTensor*>(ctx.weightScale);
    auto dstScale = ctx.outWeightScale;

    if (srcScale->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        auto srcView = ctx.executor->CreateView(srcScale, srcScale->GetViewShape(), srcScale->GetViewOffset());
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_INT8);

        auto dstView = ctx.executor->CreateView(dstScale, dstScale->GetViewShape(), dstScale->GetViewOffset());
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_INT8);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale as INT8 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto result = l0op::ViewCopy(srcScale, dstScale, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessBiasDirectCopy(QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto result = l0op::ViewCopy(const_cast<aclTensor*>(ctx.biasOptional), ctx.outBiasOptional, ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy bias failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightOffsetDirectCopy(QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto result = l0op::ViewCopy(const_cast<aclTensor*>(ctx.weightOffsetOptional), ctx.outWeightOffsetOptional,
                                 ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightOffset failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightDirectCopy(QuantContext& ctx)
{
    auto srcWeight = const_cast<aclTensor*>(ctx.weight);
    auto dstWeight = ctx.outWeight;

    if (srcWeight->GetDataType() == op::DataType::DT_INT4 || srcWeight->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        auto weightViewShape = srcWeight->GetViewShape();
        auto weightViewStrides = srcWeight->GetViewStrides();
        op::Shape packedShape(weightViewShape);
        size_t lastDimIdx = packedShape.GetDimNum() - 1;
        size_t secondLastDimIdx = packedShape.GetDimNum() - IDX_2;
        bool isTransposed = (weightViewStrides[secondLastDimIdx] == 1 && weightViewShape.GetDim(secondLastDimIdx) > 1);
        size_t packDimIdx = isTransposed ? secondLastDimIdx : lastDimIdx;
        packedShape.SetDim(packDimIdx, packedShape.GetDim(packDimIdx) / B4_NUMS_PER_BYTE);

        auto srcView = ctx.executor->CreateView(srcWeight, packedShape, srcWeight->GetViewOffset() / B4_NUMS_PER_BYTE);
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weight failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_UINT8);

        auto dstView = ctx.executor->CreateView(dstWeight, packedShape, dstWeight->GetViewOffset() / B4_NUMS_PER_BYTE);
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeight failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_UINT8);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weight as UINT8 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    auto result = l0op::ViewCopy(srcWeight, dstWeight, ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weight failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

// ===== A16S4/A16F4 检查组合：按粒度拼装，条目间无内部分流 =====
// 输入公共检查（紧凑 4-bit 打包维须为偶数；4-bit 直拷视图须连续/严格转置且偏移字节对齐）
// scale/offset/bias 为透传参数：out 侧 same 校验已保证输出与输入完全一致，
// 其自身的 dtype/format/shape 约束 preprocess 不消费也不处理，统一下放下游 wqbmmv2 拦截
const std::vector<CheckFunc> INPUT_BASE_CHECKS = {CheckWeightNotEmpty, CheckWeightFormatND, CheckWeightPackingDimEven,
                                                  CheckWeight4BitDirectCopyView};

// F4/A8W4-MX 条目无 offset 直拷 process：入参 offset 必须 nullptr，否则会被静默丢弃
const std::vector<CheckFunc> OFFSET_NULL_CHECK = {CheckWeightOffsetOptionalNull};

// out weight ND 直拷检查
const std::vector<CheckFunc> OUT_WEIGHT_ND_CHECKS = {CheckOutWeightSameAsInput};

// 非转置防御（judge 已保证非转置）：供 CombineChecks 条目在 OUT_WEIGHT_NZ_C016_CHECKS 前插入
const std::vector<CheckFunc> WEIGHT_NOT_TRANS_CHECK = {CheckWeightNotTrans};

// out weight NZ_C0_16 分形检查（非转置防御由 WEIGHT_NOT_TRANS_CHECK 承担）
const std::vector<CheckFunc> OUT_WEIGHT_NZ_C016_CHECKS = {
    CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
    CheckOutWeightTransNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16>};

// 输出 scale/offset/bias 与入参一致性检查（公共尾部）
const std::vector<CheckFunc> OUT_TAIL_CHECKS = {CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                                                CheckOutBiasOptionalSameAsInput};

template <typename... Groups>
static std::vector<CheckFunc> CombineChecks(const Groups&... groups)
{
    std::vector<CheckFunc> combined;
    (combined.insert(combined.end(), groups.begin(), groups.end()), ...);
    return combined;
}

const std::unordered_map<NpuArch, std::vector<DataFlowEntry>> NPU_DATA_FLOW_REGISTRY_MAP = {
    {NpuArch::DAV_3510,
     {{.judge = IsMMMxA8W4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightTrans,              // 校验 weight 最后 2 维是否转置
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,        // out 与 input 的 nullptr/empty/viewShape 一致
                  CheckOutWeightDtypeSame,       // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim,    // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
                  CheckOutWeightTransNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32>, // 校验输出 weight storageShape
                                                                                        // 符合 C0_32
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightTransNd2Nz<IDX_0>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsGMMMxA8W4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightTrans,              // 校验 weight 最后 2 维是否转置
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,        // out 与 input 的 nullptr/empty/viewShape 一致
                  CheckOutWeightDtypeSame,       // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim,    // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
                  CheckOutWeightTransNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32>, // 校验输出 weight storageShape
                                                                                        // 符合 C0_32
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightTransNd2Nz<IDX_1>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMA16S4PerTensorDataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightPackingDimEven,     // 紧凑 4-bit 打包维须为偶数
                  CheckWeight4BitDirectCopyView, // 4-bit 直拷视图须连续/严格转置且偏移字节对齐
                  // per-tensor 不支持 NZ：outWeight 须与 weight 完全一致（format==），转置/非转置 weight 均直拷
                  CheckOutWeightSameAsInput, CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-channel 转置：ND 直拷（物理透传）
      {.judge = IsMMA16S4PerChannelTransDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-channel 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerChannelNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OUT_WEIGHT_NZ_C016_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 转置：ND 直拷（scale 形状须匹配 per-group 分组语义）
      {.judge = IsMMA16S4PerGroupTransDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerGroupNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OUT_WEIGHT_NZ_C016_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      {.judge = IsMMA16F4PerGroupDataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightNotTrans,       // A16F4 per-group NZ 路径仅支持非转置 weight，转置返回错误
                  CheckWeightPackingDimEven, // 紧凑 FP4 打包维须为偶数（与 INT4 同约束）
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,
                  CheckOutWeightDtypeSame, // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim, CheckOutWeightTransNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16>,
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // A16MXFP4 转置：ND 直拷（wqbmmv2 MX kernel 支持 ND 转置输入；offset 不支持必须 nullptr）
      {.judge = IsMMA16MXF4TransDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, OFFSET_NULL_CHECK, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // A16MXFP4 非转置 + NZ_C0_16 出：ND→NZ 分形转换（NZ 出仅支持非转置）
      {.judge = IsMMA16MXF4NonTransNzDataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightNotTrans, // A16MXFP4 NZ 路径仅支持非转置 weight，转置由转置条目拦截
                  CheckWeightPackingDimEven,
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,
                  CheckOutWeightDtypeSame, // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim, CheckOutWeightTransNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16>,
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}}}}};
