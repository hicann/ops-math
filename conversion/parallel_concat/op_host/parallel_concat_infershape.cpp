/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "graph/types.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

namespace {

// Attr indexes in IR declaration order:
// attr 0 = "shape" (REQUIRED ListInt), attr 1 = "N" (REQUIRED Int).
constexpr size_t ATTR_INDEX_SHAPE = 0U;
constexpr size_t ATTR_INDEX_N = 1U;
// IR index of the single DYNAMIC_INPUT "values".
constexpr size_t IR_INDEX_VALUES = 0U;
// rank(values) ∈ [1, 8].
constexpr size_t MIN_RANK = 1U;
constexpr size_t MAX_RANK = 8U;

// A shape is unknown-rank when it is the single-dim [-2] marker.
inline bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1U && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

// Checked u64 multiply (scale-overflow guard): false on overflow.
inline bool SafeMulU64(uint64_t a, uint64_t b, uint64_t* r) { return !__builtin_mul_overflow(a, b, r); }

// Two known-rank shapes are compatible when every dim pair is equal or one
// side is ge::UNKNOWN_DIM (-1): the unknown side resolves from the other.
bool ShapesCompatible(const gert::Shape* a, const gert::Shape* b)
{
    if (a->GetDimNum() != b->GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a->GetDimNum(); ++i) {
        const int64_t da = a->GetDim(i);
        const int64_t db = b->GetDim(i);
        if (da == db || da == ge::UNKNOWN_DIM || db == ge::UNKNOWN_DIM) {
            continue;
        }
        return false;
    }
    return true;
}

} // namespace

/**
 * InferShapeParallelConcat: GE shape inference callback.
 *
 * Validation chain (same chain and order as the TilingFunc backstop):
 * null defense -> rank ∈ [1, 8] -> attr N/shape value checks ->
 * first dim == 1 -> same-shape inputs -> attr/input shape consistency ->
 * declared-output consistency -> scale-overflow guard, then writes
 * output_data.shape = attr shape (fully defined by design).
 *
 * Returns GRAPH_SUCCESS after the chain passes and the derived shape is
 * written; GRAPH_FAILED with the spec error-code semantics (null_input /
 * dtype_not_supported / shape_mismatch / attribute_value_out_of_range).
 */
static ge::graphStatus InferShapeParallelConcat(gert::InferShapeContext* context)
{
    // Null defense (null_input) — the macro covers the context itself.
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "InferShape: enter");
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // Attr "shape" (REQUIRED ListInt, attr index 0) / attr "N" (REQUIRED Int,
    // attr index 1): a null pointer is the constructible form of the attrs
    // null defense (GetAttrs() yields null attr pointers when absent).
    const gert::TypedContinuousVector<int64_t>* attrShapeVec = attrs->GetListInt(ATTR_INDEX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrShapeVec);
    const int64_t* nPtr = attrs->GetInt(ATTR_INDEX_N);
    OP_CHECK_NULL_WITH_CONTEXT(context, nPtr);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::AnchorInstanceInfo* instInfo = context->GetIrInputInstanceInfo(IR_INDEX_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context, instInfo);
    const size_t inputCount = instInfo->GetInstanceNum(); // len(values)

    const int64_t n = *nPtr; // attr "N"
    const size_t attrShapeLen = attrShapeVec->GetSize();
    const int64_t* attrShape = attrShapeVec->GetData();

    // Attr value checks (attribute_value_out_of_range).
    if (n < 1) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "N", std::to_string(n), ">= 1");
        return GRAPH_FAILED;
    }
    if (attrShapeLen == 0UL) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(nodeName, "shape", "0", ">= 1");
        return GRAPH_FAILED;
    }
    for (size_t j = 0; j < attrShapeLen; ++j) {
        if (attrShape[j] < 0) {
            OP_LOGE_WITH_INVALID_ATTR(nodeName, "shape[" + std::to_string(j) + "]", std::to_string(attrShape[j]),
                                      "fully defined and non-negative");
            return GRAPH_FAILED;
        }
    }
    if (attrShape[0] != n) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "shape[0]", std::to_string(attrShape[0]), std::to_string(n));
        return GRAPH_FAILED;
    }
    if (static_cast<uint64_t>(n) != static_cast<uint64_t>(inputCount)) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "N", std::to_string(inputCount), std::to_string(n));
        return GRAPH_FAILED;
    }

    // Per-input checks (rank / first dim / same shape).
    // shape0: first known-rank instance as the same-shape representative;
    // unknown-rank instances ([-2]) skip per-input shape checks (their dims
    // cannot be rank/first-dim checked; the output is still attr-derived).
    const gert::Shape* shape0 = nullptr;
    for (size_t i = 0; i < inputCount; ++i) {
        const gert::Shape* shp = context->GetInputShape(i); // flat instance index
        OP_CHECK_NULL_WITH_CONTEXT(context, shp);
        if (IsUnknownRank(shp)) {
            continue; // unknown-rank instance: no rank/first-dim/same-shape check possible
        }
        const size_t rank = shp->GetDimNum();
        if (rank < MIN_RANK || rank > MAX_RANK) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(nodeName, static_cast<int>(i), "rank " + std::to_string(rank),
                                             "rank in [1, 8]");
            return GRAPH_FAILED; // shape_mismatch (rank out of range)
        }
        const int64_t dim0 = shp->GetDim(0);
        if (dim0 != 1 && dim0 != ge::UNKNOWN_DIM) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(nodeName, static_cast<int>(i), "first dim " + std::to_string(dim0),
                                             "first dim 1");
            return GRAPH_FAILED; // shape_mismatch
        }
        if (shape0 == nullptr) {
            shape0 = shp;
        } else if (!ShapesCompatible(shp, shape0)) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(nodeName, static_cast<int>(i), "different shape",
                                             "identical to the representative input");
            return GRAPH_FAILED; // shape_mismatch
        }
    }

    // attr shape[1:] == values.shape[1:] (双源合一, shape_mismatch).
    if (shape0 != nullptr) {
        if (attrShapeLen != shape0->GetDimNum()) {
            OP_LOGE_WITH_INVALID_ATTR_SIZE(nodeName, "shape", std::to_string(attrShapeLen),
                                           std::to_string(shape0->GetDimNum()));
            return GRAPH_FAILED;
        }
        for (size_t j = 1; j < attrShapeLen; ++j) {
            const int64_t inDim = shape0->GetDim(j);
            if (inDim == ge::UNKNOWN_DIM) {
                continue; // dynamic-shape input dim: resolved by the fully-defined attr shape
            }
            if (inDim != attrShape[j]) {
                OP_LOGE_WITH_INVALID_ATTR(
                    nodeName, "shape[" + std::to_string(j) + "]", std::to_string(attrShape[j]),
                    "equal to values.shape[" + std::to_string(j) + "] = " + std::to_string(inDim));
                return GRAPH_FAILED;
            }
        }
    }

    // Declared output shape consistent with attr shape (shape_mismatch).
    // The output slot carries the caller-declared desc shape
    // (update_output_desc); a conflicting declaration is rejected at the
    // inference stage. Unknown dims (-1) / unknown rank (-2) / an empty slot
    // carry no comparable information and resolve from attr.
    if (outputShape->GetDimNum() > 0UL && !IsUnknownRank(outputShape)) {
        if (outputShape->GetDimNum() != attrShapeLen) {
            OP_LOGE(nodeName,
                    "InferShape: declared output shape rank mismatch with attr shape: "
                    "rank(output)=%zu, len(shape)=%zu!",
                    outputShape->GetDimNum(), attrShapeLen);
            return GRAPH_FAILED;
        }
        for (size_t j = 0; j < attrShapeLen; ++j) {
            const int64_t outDim = outputShape->GetDim(j);
            if (outDim == ge::UNKNOWN_DIM) {
                continue; // dynamic-shape declared output dim: resolved by attr
            }
            if (outDim != attrShape[j]) {
                OP_LOGE(nodeName,
                        "InferShape: declared output shape mismatch with attr shape at "
                        "dim %zu: %lld != %lld!",
                        j, static_cast<long long>(outDim), static_cast<long long>(attrShape[j]));
                return GRAPH_FAILED;
            }
        }
    }

    // Scale-overflow guard (attribute_value_out_of_range):
    // totalElements = N × ∏(shape[1:]): every factor is non-negative (the
    // attr checks above); a u64 product overflow is rejected explicitly.
    uint64_t totalElements = static_cast<uint64_t>(n);
    for (size_t j = 1; j < attrShapeLen; ++j) {
        if (!SafeMulU64(totalElements, static_cast<uint64_t>(attrShape[j]), &totalElements)) {
            OP_LOGE(nodeName, "InferShape: overflow when computing totalElements at dim %zu (N=%lld)!", j,
                    static_cast<long long>(n));
            return GRAPH_FAILED;
        }
    }

    OP_LOGI(nodeName, "InferShape: output rank=%zu, totalElements=%llu", attrShapeLen,
            static_cast<unsigned long long>(totalElements));
    outputShape->SetDimNum(attrShapeLen);
    for (size_t j = 0; j < attrShapeLen; ++j) {
        outputShape->SetDim(j, attrShape[j]);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ParallelConcat).InferShape(InferShapeParallelConcat);

} // namespace ops
