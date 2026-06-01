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
 * \file truncate_div_tiling_arch35.cpp
 * \brief truncate_div_tiling source file
 */

#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/truncate_div/op_kernel/arch35/truncate_div_dag.h"
#include "math/truncate_div/op_kernel/arch35/truncate_div_struct.h"
#include "truncate_div_tiling_arch35.h"
#include "op_host/util/fp16.h"
#include "util/bfloat16.h"

using namespace Ops::Base;
using namespace ge;

namespace optiling {

constexpr static uint64_t TRUNCATE_DIV_COMMON_TILING_PRIORITY = 0;
constexpr static uint32_t INPUT_IDX_X1 = 0;
constexpr static uint32_t INPUT_IDX_X2 = 1;
constexpr static int64_t DCACHE_SIZE = 32 * 1024;

ge::graphStatus TruncateDivTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool TruncateDivTiling::IsCapable()
{
    return true;
}

float TruncateDivTiling::GetReciprocal(float data)
{
    float invScalarVal = 0.0f;
    float scalarValue = data;
    if (scalarValue == 0.0f) {
        float reciprocalZero = 1.0f / scalarValue;
        if (reciprocalZero < 0) {
            invScalarVal = -INFINITY;
        } else {
            invScalarVal = INFINITY;
        }
    } else if (scalarValue == INFINITY) {
        invScalarVal = 0.0f;
    } else if (scalarValue == -INFINITY) {
        invScalarVal = -0.0f;
    } else {
        invScalarVal = 1.0f / scalarValue;
        invScalarVal = invScalarVal * (2.0f - scalarValue * invScalarVal);
    }
    return invScalarVal;
}

template <typename T>
ge::graphStatus TruncateDivTiling::GetConstData(uint32_t inputIdx, T& data)
{
    auto tensor = context_->GetInputTensor(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor);
    const T* value = tensor->GetData<T>();
    if (value == nullptr) {
        OP_LOGE(context_->GetNodeName(), "const tensor is null.");
        return ge::GRAPH_FAILED;
    }
    data = value[0];
    OP_LOGI(context_->GetNodeName(), "scalarData %f", data);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncateDivTiling::DoOpTiling()
{
    // 1. 获取输入描述
    auto x1Desc = context_->GetInputDesc(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    auto x2Desc = context_->GetInputDesc(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);

    ge::DataType x1DType = x1Desc->GetDataType();
    ge::DataType x2DType = x2Desc->GetDataType();

    // 2. 获取形状
    auto x1StorageShape = context_->GetInputShape(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1StorageShape);
    auto x2StorageShape = context_->GetInputShape(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2StorageShape);

    auto x2Shape = x2StorageShape->GetStorageShape();

//    bool isScalar = x2Shape.IsScalar() || (x2Shape.GetDimNum() == 1 && x2Shape.GetDim(0) == 1);
    bool isScalar = x2Shape.IsScalar();
    bool canUseMul = isScalar && (x2DType == ge::DT_FLOAT || x2DType == ge::DT_FLOAT16 || x2DType == ge::DT_BF16);

    OP_LOGI(context_->GetNodeName(), "canUseMul %d", canUseMul);
    if (canUseMul) {
        bool success = false;
        float scalarValue = 0.0f;
        switch (x2DType) {
            case ge::DT_FLOAT: {
                success = (GetConstData<float>(INPUT_IDX_X2, scalarValue) == ge::GRAPH_SUCCESS);
                if (success) {
                    reciprocal_ = GetReciprocal(scalarValue);
                }
                break;
            }
            case ge::DT_FLOAT16: {
                uint16_t tmpValue = 0;
                success = (GetConstData<uint16_t>(INPUT_IDX_X2, tmpValue) == ge::GRAPH_SUCCESS);
                if (success) {
                    scalarValue = float(*(reinterpret_cast<const fp16_t*>(&tmpValue)));
                    reciprocal_ = GetReciprocal(scalarValue);
                }
                break;
            }
            case ge::DT_BF16: {
                uint16_t tmpValue = 0;
                success = (GetConstData<uint16_t>(INPUT_IDX_X2, tmpValue) == ge::GRAPH_SUCCESS);
                if (success) {
                    scalarValue = float(*(reinterpret_cast<const bfloat16*>(&tmpValue)));
                    reciprocal_ = GetReciprocal(scalarValue);
                }
                break;
            }
            default:
                OP_LOGE(
                    context_->GetNodeName(), "Unsupported scalar type for reciprocal: %s",
                    ge::TypeUtils::DataTypeToSerialString(x2DType).c_str());
                return ge::GRAPH_FAILED;
        }
        if (!success) {
            return ge::GRAPH_FAILED;
        }
        OP_LOGI(context_->GetNodeName(), "scalar value = %f, reciprocal value = %f", scalarValue, reciprocal_);
    }


    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    uint32_t schMode = 0;
    int64_t maxLiveNodeCnt = 0;
    int64_t extraBuf = DCACHE_SIZE;

    // 5. 定义模板 lambda：封装重复的 tiling 执行逻辑
    auto execTiling = [&, this]<typename OpDag>(bool isScalarBranch = false) {
        BroadcastBaseTiling<OpDag> brcTiling(context_);
        if (isScalarBranch) {
            brcTiling.SetScalar(reciprocal_);
        }

        ret = brcTiling.DoTiling();
        schMode = brcTiling.GetSchMode();

        tilingKey_ = GET_TPL_TILING_KEY(schMode, canUseMul);
    };

    // 6. 根据数据类型组合调用 lambda
    if (x1DType == ge::DT_FLOAT16 && x2DType == ge::DT_FLOAT) {
        if (canUseMul) {
            execTiling.template operator()<TruncateDivOp::TruncateDivFloatWithCastScalar<half, float, float>::OpDag>(
                true);
        } else {
            execTiling.template operator()<TruncateDivOp::TruncateDivFloatWithCast<half, float, float>::OpDag>(false);
        }
    } else if (x1DType == ge::DT_FLOAT16 || x1DType == ge::DT_BF16) {
        if (x2DType == x1DType) {
            if (canUseMul) {
                execTiling.template operator()<TruncateDivOp::TruncateDivFloat16Scalar<half, float>::OpDag>(true);
            } else {
                execTiling.template operator()<TruncateDivOp::TruncateDivFloat16<half, float>::OpDag>(false);
            }
        }
    } else if (x1DType == ge::DT_FLOAT) {
        if (x2DType == ge::DT_FLOAT) {
            if (canUseMul) {
                execTiling.template operator()<TruncateDivOp::TruncateDivFloatScalar<float>::OpDag>(true);
            } else {
                execTiling.template operator()<TruncateDivOp::TruncateDivFloat<float>::OpDag>(false);
            }
        } else if (x2DType == ge::DT_INT32) {
            execTiling.template operator()<TruncateDivOp::TruncateDivFloatToLowBit<float, int32_t, float>::OpDag>(
                false);
        } else if (x2DType == ge::DT_FLOAT16) {
            if (canUseMul) {
                execTiling.template
                operator()<TruncateDivOp::TruncateDivFloatWithCastScalar<float, half, float>::OpDag>(true);
            } else {
                execTiling.template operator()<TruncateDivOp::TruncateDivFloatToLowBit<float, half, float>::OpDag>(false);
            }
        }
    } else if (x1DType == ge::DT_INT8 || x1DType == ge::DT_UINT8) {
        // 这些分支不支持 scalar 优化（canUseMul 应该为 false）
        if (x1DType == ge::DT_INT8) {
            execTiling.template operator()<TruncateDivOp::TruncateDivIntS8<int8_t, half>::OpDag>(false);
        } else {
            execTiling.template operator()<TruncateDivOp::TruncateDivIntU8<uint8_t, uint16_t>::OpDag>(false);
        }
    } else if (x1DType == ge::DT_INT16) {
        execTiling.template operator()<TruncateDivOp::TruncateDivInt<int16_t>::OpDag>(false);
    } else if (x1DType == ge::DT_INT32 && x2DType == ge::DT_INT32) {
        execTiling.template operator()<TruncateDivOp::TruncateDivInt<int32_t>::OpDag>(false);
    } else if (x1DType == ge::DT_INT64) {
        BroadcastBaseTiling<TruncateDivOp::TruncateDivInt64<int64_t>::OpDag> brcTiling(context_);
        ret = brcTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        schMode = brcTiling.GetSchMode();
        tilingKey_ = GET_TPL_TILING_KEY(schMode, canUseMul);
    } else if (x1DType == ge::DT_INT32 && x2DType == ge::DT_FLOAT) {
        execTiling.template operator()<TruncateDivOp::TruncateDivIntToFloat<int32_t, float, float>::OpDag>(false);
    } else {
        OP_LOGE(
            context_->GetNodeName(), "Unsupported dtype combination: self=%s, other=%s",
            ge::TypeUtils::DataTypeToSerialString(x1DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(x2DType).c_str());
        return ge::GRAPH_FAILED;
    }

    // 7. 完成后打印结果
    OP_LOGI(
        context_, "TruncateDiv tiling completed: tilingKey_=%lu, schMode=%u, canUseMul=%d, reciprocal %f", tilingKey_, schMode,
        static_cast<int>(canUseMul), reciprocal_);

    return ret;
}

ge::graphStatus TruncateDivTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t TruncateDivTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus TruncateDivTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncateDivTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncateDivTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const BroadcastCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(context_->GetNodeName(), "Get ubSize form compileInfo is: %ld", ubSize_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
        OP_LOGD(context_->GetNodeName(), "Get ubSize form ascendcPlatform is: %ld", ubSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForTruncateDiv(gert::TilingContext* context)
{
    OP_LOGD("TruncateDivTiling", "Enter TilingForTruncateDiv");
    if (context == nullptr) {
        OP_LOGE("TruncateDivTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc TruncateDivTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForTruncateDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncateDiv)
    .Tiling(TilingForTruncateDiv)
    .TilingInputsDataDependency({INPUT_IDX_X2})
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForTruncateDiv);

REGISTER_OPS_TILING_TEMPLATE(TruncateDiv, TruncateDivTiling, TRUNCATE_DIV_COMMON_TILING_PRIORITY);
} // namespace optiling