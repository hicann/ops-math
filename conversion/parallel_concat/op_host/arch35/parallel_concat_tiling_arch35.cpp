/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h" // TilingParseForParallelConcat (R15 no-op callback)
#include "register/op_def_registry.h"               // IMPL_OP_OPTILING
#include "op_common/log/log.h"                      // OP_LOGE / OP_LOGI / OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/math_util.h"       // Ops::Base::FloorAlign / CeilAlign / CeilDiv
#include "op_common/op_host/util/platform_util.h"   // Ops::Base::GetUbBlockSize
#include "graph/types.h"                            // ge::GetSizeByDataType
#include "tiling/platform/platform_ascendc.h"
#include "parallel_concat_tiling_arch35.h"
#include "../../op_kernel/arch35/parallel_concat_tiling_struct.h"

namespace optiling {

namespace {

// Checked u64 multiply: false on overflow.
inline bool SafeMulU64(uint64_t a, uint64_t b, uint64_t* r) { return !__builtin_mul_overflow(a, b, r); }

inline bool IsDtypeWhitelisted(ge::DataType dt)
{
    constexpr ge::DataType kWhitelist[] = {ge::DT_BOOL,  ge::DT_INT8,   ge::DT_UINT8,  ge::DT_FLOAT16,
                                           ge::DT_BF16,  ge::DT_INT16,  ge::DT_UINT16, ge::DT_FLOAT,
                                           ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,  ge::DT_UINT64};
    return std::find(std::begin(kWhitelist), std::end(kWhitelist), dt) != std::end(kWhitelist);
}

constexpr uint64_t UB_SINGLE_BLOCK_CAP_BYTES = 64ULL * 1024ULL;
constexpr uint64_t NARROW_MIN_ROW_BYTES_PER_CORE = 8;

constexpr uint64_t NARROW_MIN_TOTAL_BYTES_PER_CORE = 256;
// 中行补核分支下界（与 bufferSize 基座档解耦的独立调优边界：仅
// 64KB <= rowBytes < 512KB 的欠并行输入走 8KB 最小 chunk 拆行）。
constexpr uint64_t MID_ROW_FILL_MIN_ROW_BYTES = 64ULL * 1024ULL;
// 中行补核的最小 chunk 字节数（拆行时每 chunk 至少 8KB，限制 chunk 数上限）。
constexpr uint64_t SPLIT_MIN_CHUNK_BYTES = 8192;
// SIMD 补核小总量护栏（与 SIMT 窄分支 NARROW_MIN_TOTAL_BYTES_PER_CORE 对称）：
// 拆行补核时每核至少摊到该字节数——总量小时收缩拆分，避免 8KB 级小 chunk
// 摊满全部核后落入逐 chunk MTE 延迟区（实测 0.3-2.4MB 欠并行输入摊满 36-56 核
// 后 MBU 仅 10-19%；每核 ≥61KB 的健康用例不受影响，48KB 为实测分隔阈值）。
constexpr uint64_t FILL_MIN_BYTES_PER_CORE = 48ULL * 1024ULL;
// 护栏启用下限：总量低于该值时摊多核在 launch 阴影内完成、无害，不收缩
// （实测 92KB/131KB 小总量被护栏压到 1-2 核反而损失带宽 -23%）。
constexpr uint64_t FILL_GUARD_MIN_TOTAL_BYTES = 256ULL * 1024ULL;
constexpr size_t MAX_OP_RANK = 8;
const char* const
    DTYPE_WHITELIST_TEXT = "float32/float16/bfloat16/int8/int16/int32/int64/uint8/uint16/uint32/uint64/bool";

inline std::string ShapeToString(const std::vector<int64_t>& dims)
{
    std::string text;
    for (size_t d = 0; d < dims.size(); ++d) {
        if (d != 0) {
            text += ",";
        }
        text += std::to_string(dims[d]);
    }
    return text;
}

// TilingHandles: null 防御段一次解析、后续全程复用的 context 派生句柄。
struct TilingHandles {
    const gert::ComputeNodeInfo* computeNodeInfo;
    const gert::TypedContinuousVector<int64_t>* attrShapeVec;
    const int64_t* nPtr;
};

// 校验链第 1 段 —— null 防御：computeNodeInfo / attrs 容器 / attr shape
// (ListInt, REQUIRED) / attr N (Int, REQUIRED)。GRAPH_FAILED = 拒绝（日志已打）。
inline ge::graphStatus ResolveTilingHandles(gert::TilingContext* context, TilingHandles& handles)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetComputeNodeInfo());
    handles.computeNodeInfo = context->GetComputeNodeInfo();
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    handles.attrShapeVec = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, handles.attrShapeVec);
    handles.nPtr = attrs->GetInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, handles.nPtr);
    return ge::GRAPH_SUCCESS;
}

// 平台动态查询：AIV 核数 + UB 大小（HOST-4：绝不硬编码、绝不走 CompileInfo）。
inline ge::graphStatus QueryPlatform(gert::TilingContext* context, uint32_t& coreNum, uint64_t& ubSize)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC plat(platformInfo);
    coreNum = plat.GetCoreNumAiv();                                 // AIV 核数 [cores]
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); // [bytes]
    return ge::GRAPH_SUCCESS;
}

// 校验链第 2 段 —— dtype：12-dtype 白名单 + 输入间一致 + 输出一致。
// 成功时经 dtypeSize 输出元素字节数（ge::GetSizeByDataType）。
inline ge::graphStatus CheckDtypes(gert::TilingContext* context, const char* nodeName, size_t inputCount,
                                   uint64_t& dtypeSize)
{
    const auto* desc0 = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc0);
    const ge::DataType inDtype0 = desc0->GetDataType();
    if (!IsDtypeWhitelisted(inDtype0)) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "values[0]", std::to_string(static_cast<int>(inDtype0)),
                                         DTYPE_WHITELIST_TEXT);
        return ge::GRAPH_FAILED; // dtype_not_supported
    }
    dtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(inDtype0));
    // values[0] 已由上方 desc0 / 白名单校验覆盖，循环从 1 起避免重复检查。
    for (size_t i = 1; i < inputCount; ++i) {
        const auto* desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        const ge::DataType dt = desc->GetDataType();
        if (!IsDtypeWhitelisted(dt)) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "values[" + std::to_string(i) + "]",
                                             std::to_string(static_cast<int>(dt)), DTYPE_WHITELIST_TEXT);
            return ge::GRAPH_FAILED; // dtype_not_supported
        }
        if (dt != inDtype0) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "values", std::to_string(static_cast<int>(dt)),
                                             "identical to values[0] (no promotion)");
            return ge::GRAPH_FAILED; // dtype_not_supported
        }
    }
    const auto* outDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    if (outDesc->GetDataType() != inDtype0) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "output_data",
                                         std::to_string(static_cast<int>(outDesc->GetDataType())),
                                         "identical to values[0] (no promotion)");
        return ge::GRAPH_FAILED; // dtype_not_supported
    }
    return ge::GRAPH_SUCCESS;
}

// 校验链第 4 段 —— rank 界限 rank(values[i]) ∈ [1, MAX_OP_RANK]（逐输入；
// 跨输入一致性由第 6 段检查）+ inputShapes 收集。
inline ge::graphStatus CollectInputShapes(gert::TilingContext* context, const char* nodeName, size_t inputCount,
                                          std::vector<std::vector<int64_t>>& inputShapes)
{
    inputShapes.reserve(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
        const auto* shp = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shp);
        const gert::Shape s = shp->GetStorageShape();
        const size_t rank = s.GetDimNum();
        if (rank < 1UL || rank > MAX_OP_RANK) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(nodeName, static_cast<int>(i), "rank " + std::to_string(rank),
                                             "rank in [1, 8]");
            return ge::GRAPH_FAILED; // shape_mismatch (rank out of range)
        }
        std::vector<int64_t> dims;
        dims.reserve(rank);
        for (size_t d = 0; d < rank; ++d) {
            dims.push_back(s.GetDim(d));
        }
        inputShapes.push_back(dims);
    }
    return ge::GRAPH_SUCCESS;
}

// 校验链第 5 段 —— attr 值域：N >= 1；attr shape 非空、shape[0] == N、
// N == len(values)、attr shape 全定义且非负。
inline ge::graphStatus CheckAttrValues(const char* nodeName, const gert::TypedContinuousVector<int64_t>* attrShapeVec,
                                       int64_t n, size_t inputCount)
{
    if (n < 1) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "N", std::to_string(n), ">= 1");
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    const size_t attrShapeLen = attrShapeVec->GetSize();
    if (attrShapeLen == 0UL) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(nodeName, "shape", "0", ">= 1");
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    if (static_cast<uint64_t>(attrShapeVec->GetData()[0]) != static_cast<uint64_t>(n)) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "shape[0]", std::to_string(attrShapeVec->GetData()[0]), std::to_string(n));
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    if (static_cast<uint64_t>(n) != static_cast<uint64_t>(inputCount)) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "N", std::to_string(inputCount), std::to_string(n));
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    for (size_t j = 0; j < attrShapeLen; ++j) {
        if (attrShapeVec->GetData()[j] < 0) {
            OP_LOGE_WITH_INVALID_ATTR(nodeName, "shape[" + std::to_string(j) + "]",
                                      std::to_string(attrShapeVec->GetData()[j]), "fully defined and non-negative");
            return ge::GRAPH_FAILED; // attribute_value_out_of_range
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 校验链第 6a 段 —— 输入间 shape 一致性：首维 == 1、输入同形、
// attrShape[1:] == values.shape[1:]。
inline ge::graphStatus CheckInputConsistency(const char* nodeName,
                                             const gert::TypedContinuousVector<int64_t>* attrShapeVec,
                                             const std::vector<std::vector<int64_t>>& inputShapes, size_t inputCount)
{
    const std::vector<int64_t>& s0 = inputShapes[0];
    for (size_t i = 0; i < inputCount; ++i) {
        if (inputShapes[i][0] != 1) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(nodeName, static_cast<int>(i),
                                             "first dim " + std::to_string(inputShapes[i][0]), "first dim 1");
            return ge::GRAPH_FAILED; // shape_mismatch
        }
    }
    for (size_t i = 1; i < inputCount; ++i) {
        if (inputShapes[i] != s0) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(nodeName, static_cast<int>(i), ShapeToString(inputShapes[i]),
                                             "identical to values[0]");
            return ge::GRAPH_FAILED; // shape_mismatch
        }
    }
    const size_t attrShapeLen = attrShapeVec->GetSize();
    if (attrShapeLen != s0.size()) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(nodeName, "shape", std::to_string(attrShapeLen), std::to_string(s0.size()));
        return ge::GRAPH_FAILED; // shape_mismatch
    }
    for (size_t j = 1; j < s0.size(); ++j) {
        if (attrShapeVec->GetData()[j] != s0[j]) {
            OP_LOGE_WITH_INVALID_ATTR(nodeName, "shape[" + std::to_string(j) + "]",
                                      std::to_string(attrShapeVec->GetData()[j]),
                                      "equal to values.shape[" + std::to_string(j) + "] = " + std::to_string(s0[j]));
            return ge::GRAPH_FAILED; // shape_mismatch
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 校验链第 6b 段 —— 输出 shape 一致性（双源合一的输出侧）：output == attrShape。
inline ge::graphStatus CheckOutputShape(gert::TilingContext* context, const char* nodeName,
                                        const gert::TypedContinuousVector<int64_t>* attrShapeVec)
{
    const size_t attrShapeLen = attrShapeVec->GetSize();
    const auto* outShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    const gert::Shape outShape = outShapePtr->GetStorageShape();
    if (outShape.GetDimNum() != attrShapeLen) {
        OP_LOGE(nodeName, "TilingFunc: output shape mismatch with attr shape: rank(output)=%zu, len(shape)=%zu!",
                outShape.GetDimNum(), attrShapeLen);
        return ge::GRAPH_FAILED; // shape_mismatch
    }
    for (size_t j = 0; j < attrShapeLen; ++j) {
        if (outShape.GetDim(j) != attrShapeVec->GetData()[j]) {
            OP_LOGE(nodeName, "TilingFunc: output shape mismatch with attr shape at dim %zu: %lld != %lld!", j,
                    static_cast<long long>(outShape.GetDim(j)), static_cast<long long>(attrShapeVec->GetData()[j]));
            return ge::GRAPH_FAILED; // shape_mismatch
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ValidatedInputs: 校验链通过后解析的事实集（后续 tiling 计算的唯一输入）。
struct ValidatedInputs {
    int64_t n;                                     // attr "N"（已证 == len(values) == shape[0]）
    size_t inputCount;                             // len(values)
    uint64_t dtypeSize;                            // 元素字节数 [bytes/element]
    std::vector<std::vector<int64_t>> inputShapes; // 逐输入 shape（已证同形）
};

// 校验链总入口：attr 解析（n / inputCount）+ 第 2-6 段（dtype ->
// rank -> attr 值域 -> shape 一致性）。GRAPH_FAILED = 拒绝（日志已打）。
inline ge::graphStatus ValidateInputs(gert::TilingContext* context, const char* nodeName, const TilingHandles& handles,
                                      ValidatedInputs& v)
{
    // Attr resolution (GetAttrs reads only; no GetData dereference).
    v.n = *handles.nPtr;                                    // attr "N" (Int, attr index 1)
    v.inputCount = handles.computeNodeInfo->GetInputsNum(); // len(values)
    if (v.inputCount == 0UL) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "N", std::to_string(v.inputCount), std::to_string(v.n));
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    if (CheckDtypes(context, nodeName, v.inputCount, v.dtypeSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CollectInputShapes(context, nodeName, v.inputCount, v.inputShapes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttrValues(nodeName, handles.attrShapeVec, v.n, v.inputCount) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputConsistency(nodeName, handles.attrShapeVec, v.inputShapes, v.inputCount) != ge::GRAPH_SUCCESS ||
        CheckOutputShape(context, nodeName, handles.attrShapeVec) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 合轴 [N, L] 行视图（SafeMulU64 溢出链，每步显式拒绝）：
// rowElems -> rowBytes -> totalBytes。
inline ge::graphStatus ComputeRowView(const char* nodeName, const std::vector<int64_t>& s0, uint64_t dtypeSize,
                                      uint64_t n, uint64_t& rowElems, uint64_t& rowBytes, uint64_t& totalBytes)
{
    rowElems = 1; // L [elements]；k == 0（rank=1）→ 1；任一 d_m == 0 → 0（空 tensor）
    for (size_t ax = 1; ax < s0.size(); ++ax) {
        if (!SafeMulU64(rowElems, static_cast<uint64_t>(s0[ax]), &rowElems)) {
            OP_LOGE(nodeName, "TilingFunc: overflow when collapsing values shape to rowElems!");
            return ge::GRAPH_FAILED; // attribute_value_out_of_range
        }
    }
    rowBytes = 0; // L × dtypeSize [bytes]
    if (!SafeMulU64(rowElems, dtypeSize, &rowBytes)) {
        OP_LOGE(nodeName,
                "TilingFunc: overflow when computing rowBytes = rowElems(%" PRIu64 ") × dtypeSize(%" PRIu64 ")!",
                rowElems, dtypeSize);
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    totalBytes = 0; // N × rowBytes [bytes]
    if (!SafeMulU64(n, rowBytes, &totalBytes)) {
        OP_LOGE(nodeName, "TilingFunc: overflow when computing totalBytes = N(%lld) × rowBytes(%" PRIu64 ")!",
                static_cast<long long>(n), rowBytes);
        return ge::GRAPH_FAILED; // attribute_value_out_of_range
    }
    return ge::GRAPH_SUCCESS;
}

// UB 切分：bufferSize 自适应三档的基座档选择 + 退化平台快照防御。
// 对齐基值 ubBlockSize 平台动态查询（Ops::Base::GetUbBlockSize），不硬编码。
inline ge::graphStatus SelectBaseBufferSize(const char* nodeName, uint64_t ubSize, uint64_t ubBlockSize,
                                            uint64_t rowBytes, uint64_t& bufferSize64)
{
    const uint64_t ubHalf = Ops::Base::FloorAlign(ubSize / UINT64_C(2), ubBlockSize); // ping-pong 上限 [bytes]
    const uint64_t rowCap = Ops::Base::CeilAlign(rowBytes, ubBlockSize);              // 行覆盖档块大小 [bytes]
    bufferSize64 = std::min(ubHalf, std::max(UB_SINGLE_BLOCK_CAP_BYTES, rowCap));
    if (bufferSize64 == 0ULL || bufferSize64 < SIMT_ROW_BYTES_THRESHOLD) {
        OP_LOGE(nodeName,
                "TilingFunc: bufferSize %" PRIu64 " degenerates below the routing "
                "threshold %" PRIu64 " on this platform snapshot (ubSize=%" PRIu64 ")!",
                bufferSize64, SIMT_ROW_BYTES_THRESHOLD, ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 输入、n × chunksPerRow >= coreNum 的已满核输入。
inline void TuneBufferSizeForUnderfill(uint64_t rowBytes, uint64_t n, uint32_t coreNum, bool useSimdTemplate,
                                       uint64_t ubBlockSize, uint64_t& bufferSize64, bool& overlapFill)
{
    const uint64_t overlapFillMinRowBytes = 512ULL * 1024ULL; // 大行门限 [bytes]
    const uint64_t overlapFillTargetBurst = 48ULL * 1024ULL;  // 目标突发 [bytes]
    overlapFill = false; // 大行门触发标志（核数封顶 CeilDiv(totalChunks,2)）
    if (useSimdTemplate && rowBytes >= 2ULL * SPLIT_MIN_CHUNK_BYTES) {
        // 一行数据切的块数
        const uint64_t chunksPerRowBase = Ops::Base::CeilDiv(rowBytes, bufferSize64);
        if (n * chunksPerRowBase < static_cast<uint64_t>(coreNum)) {
            if (rowBytes >= overlapFillMinRowBytes) {
                // 大行分支：~48KB 目标突发 + 每核 2 chunk 重叠保持补核。
                const uint64_t targetChunksPerRow = Ops::Base::CeilDiv(rowBytes, overlapFillTargetBurst);
                bufferSize64 = Ops::Base::CeilAlign(Ops::Base::CeilDiv(rowBytes, targetChunksPerRow), ubBlockSize);
                overlapFill = true;
            } else if (rowBytes >= MID_ROW_FILL_MIN_ROW_BYTES) {
                // 中行分支（64KB <= rowBytes < 512KB）：8KB 最小 chunk 拆行补核；
                // 小总量护栏（总量 >= FILL_GUARD_MIN_TOTAL_BYTES 时每核摊派下限
                // FILL_MIN_BYTES_PER_CORE）。
                uint64_t targetChunksPerRow = std::min(Ops::Base::CeilDiv(static_cast<uint64_t>(coreNum), n),
                                                       rowBytes / SPLIT_MIN_CHUNK_BYTES);
                if (n * rowBytes >= FILL_GUARD_MIN_TOTAL_BYTES) {
                    const uint64_t maxUsefulChunks = std::max(n, (n * rowBytes) / FILL_MIN_BYTES_PER_CORE);
                    targetChunksPerRow = std::min(targetChunksPerRow, Ops::Base::CeilDiv(maxUsefulChunks, n));
                }
                bufferSize64 = Ops::Base::CeilAlign(Ops::Base::CeilDiv(rowBytes, targetChunksPerRow), ubBlockSize);
            }
        }
    }
}

// 窄分支（rowBytes < SIMT_ROW_BYTES_THRESHOLD）双维度封顶核数：
//   1) 行字节维度：每核至少摊到 NARROW_MIN_ROW_BYTES_PER_CORE 字节的
//      行数据（如 N=122 个 [1] 标量 u32：单核 2.2us vs 56 核 10.3us）；
//   2) 总字节维度：每核至少摊到 NARROW_MIN_TOTAL_BYTES_PER_CORE 字节的
//      总输出（总数据太小摊不平多核分发开销，如 n=4 × 16B 输入开 2 核
//      实测净亏 ~0.5us）。
inline void CapNarrowBranchCores(uint64_t rowBytes, uint64_t totalBytes, uint64_t& numActiveCores)
{
    const uint64_t narrowCoreCap = std::max(
        static_cast<uint64_t>(1),
        std::min(rowBytes / NARROW_MIN_ROW_BYTES_PER_CORE, totalBytes / NARROW_MIN_TOTAL_BYTES_PER_CORE));
    if (numActiveCores > narrowCoreCap) {
        numActiveCores = narrowCoreCap;
    }
}

inline ge::graphStatus SplitMulticoreChunks(const char* nodeName, int64_t n, uint64_t rowBytes, uint32_t bufferSize,
                                            uint32_t coreNum, uint64_t totalBytes, bool useSimdTemplate,
                                            bool overlapFill, uint64_t& numActiveCores, uint64_t& perCoreChunks)
{
    numActiveCores = 0; // [cores]
    perCoreChunks = 0;  // baseC [chunks]
    if (rowBytes == 0ULL) {
        // 空 tensor 短路：单核零迭代成功返回（无字段重映射）。
        numActiveCores = 1ULL;
        perCoreChunks = 0ULL;
    } else {
        const uint64_t chunksPerRow = Ops::Base::CeilDiv(rowBytes, static_cast<uint64_t>(bufferSize));
        uint64_t totalChunks = 0; // n × chunksPerRow
        if (!SafeMulU64(static_cast<uint64_t>(n), chunksPerRow, &totalChunks)) {
            OP_LOGE(nodeName,
                    "TilingFunc: overflow when computing totalChunks = N(%lld) × "
                    "chunksPerRow(%" PRIu64 ")!",
                    static_cast<long long>(n), chunksPerRow);
            return ge::GRAPH_FAILED; // attribute_value_out_of_range
        }
        // overlapFill 门触发时 numActiveCores 封顶 min(coreNum,
        // CeilDiv(totalChunks, 2)) → perCoreChunks >= 2（floor+remC 原路径，
        // MTE2/MTE3 逐 chunk 重叠保持；实测 n=1 单行放开减核反而 -13~15%，
        // 减核设计保留）。其余输入保持 numActiveCores = min(totalChunks, coreNum)。
        numActiveCores = overlapFill ?
                             std::min(static_cast<uint64_t>(coreNum), Ops::Base::CeilDiv(totalChunks, UINT64_C(2))) :
                             std::min(totalChunks, static_cast<uint64_t>(coreNum));

        if (!useSimdTemplate) {
            CapNarrowBranchCores(rowBytes, totalBytes, numActiveCores);
        }
        if (numActiveCores == 0ULL) {
            // coreNum == 0 的退化平台快照：拒绝，绝不除零崩溃（防御性契约）。
            OP_LOGE(nodeName, "TilingFunc: coreNum is 0 on this platform snapshot, cannot split chunks!");
            return ge::GRAPH_FAILED;
        }
        perCoreChunks = totalChunks / numActiveCores; // 前 remC 核 baseC+1，其余 baseC
        if (perCoreChunks > 0xFFFFFFFFULL) {
            // perCoreChunks u32 字段承载界（attribute_value_out_of_range）。
            OP_LOGE(nodeName,
                    "TilingFunc: perCoreChunks %" PRIu64 " exceeds the uint32 "
                    "field capacity 0xFFFFFFFF!",
                    perCoreChunks);
            return ge::GRAPH_FAILED; // attribute_value_out_of_range
        }
    }
    return ge::GRAPH_SUCCESS;
}

// FillAndLogTilingData: 8 fields (units documented on the struct) + OP_LOGI anchors.
inline void FillAndLogTilingData(const char* nodeName, ParallelConcatTilingData* td, int64_t n, uint64_t rowElems,
                                 uint64_t rowBytes, uint64_t totalBytes, uint8_t dtypeSize, uint32_t numActiveCores,
                                 uint32_t perCoreChunks, uint32_t bufferSize)
{
    td->n = static_cast<uint64_t>(n);
    td->rowElems = rowElems;
    td->rowBytes = rowBytes;
    td->totalBytes = totalBytes;
    td->dtypeSize = dtypeSize;
    td->numActiveCores = numActiveCores;
    td->perCoreChunks = perCoreChunks;
    td->bufferSize = bufferSize;
    OP_LOGI(nodeName, "row: n=%lld, L=%" PRIu64 ", rowBytes=%" PRIu64 ", totalBytes=%" PRIu64 ", dtypeSize=%" PRIu64,
            static_cast<long long>(n), rowElems, rowBytes, totalBytes, static_cast<uint64_t>(dtypeSize));
    OP_LOGI(nodeName, "multicore: activeCores=%" PRIu32 ", perCoreChunks=%" PRIu32, numActiveCores, perCoreChunks);
    OP_LOGI(nodeName, "buffer: bufferSize=%" PRIu32 ", pingpong=%" PRIu32, bufferSize, bufferSize * 2U);
}

// 落盘：填 TilingData（8 字段）+ SetBlockDim / SetTilingKey / workspace。
inline ge::graphStatus FinalizeTiling(gert::TilingContext* context, const char* nodeName, int64_t n, uint64_t rowElems,
                                      uint64_t rowBytes, uint64_t totalBytes, uint64_t dtypeSize,
                                      uint64_t numActiveCores, uint64_t perCoreChunks, uint32_t bufferSize,
                                      bool useSimdTemplate)
{
    ParallelConcatTilingData* td = context->GetTilingData<ParallelConcatTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    FillAndLogTilingData(nodeName, td, n, rowElems, rowBytes, totalBytes, static_cast<uint8_t>(dtypeSize),
                         static_cast<uint32_t>(numActiveCores), static_cast<uint32_t>(perCoreChunks), bufferSize);
    // SetBlockDim：仅激活核参与（状态检查——失败会让 blockDim 槽位残留旧值而
    // TilingFunc 仍报成功）。
    if (context->SetBlockDim(static_cast<uint32_t>(numActiveCores)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "TilingFunc: SetBlockDim(%" PRIu64 ") failed!", numActiveCores);
        return ge::GRAPH_FAILED;
    }
    // SetTilingKey：单 1bit UINT 参数 COPY_MODE（下标 == 值）：
    // key 0 = narrow-simt（纯 SIMT VF 直拷 GM->GM）/ key 1 = wide-simd（SIMD）。
    const uint64_t tilingKey = useSimdTemplate ? 1ULL : 0ULL;
    if (context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "TilingFunc: SetTilingKey(%" PRIu64 ") failed!", tilingKey);
        return ge::GRAPH_FAILED;
    }
    // workspace：无 GM workspace（GM→UB→GM 中继）。
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingFuncParallelConcat(gert::TilingContext* context)
{
    // Null defense — the macro covers the context itself, ahead of every
    // dereference including GetNodeName.
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter TilingFuncParallelConcat");

    // —— 准备段：null 防御 -> 平台动态查询 -> 校验链（dtype -> rank -> attr -> shape）——
    TilingHandles handles{};
    if (ResolveTilingHandles(context, handles) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED; // null_input
    }
    uint32_t coreNum = 0; // AIV 核数 [cores]
    uint64_t ubSize = 0;  // UB 大小 [bytes]
    if (QueryPlatform(context, coreNum, ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED; // null_input
    }
    const uint64_t ubBlockSize = Ops::Base::GetUbBlockSize(context); // UB 块对齐基值 [bytes]
    ValidatedInputs v{};
    if (ValidateInputs(context, nodeName, handles, v) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t rowElems = 0;   // L [elements]
    uint64_t rowBytes = 0;   // L × dtypeSize [bytes]
    uint64_t totalBytes = 0; // N × rowBytes [bytes]
    if (ComputeRowView(nodeName, v.inputShapes[0], v.dtypeSize, static_cast<uint64_t>(v.n), rowElems, rowBytes,
                       totalBytes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t bufferSize64 = 0;
    if (SelectBaseBufferSize(nodeName, ubSize, ubBlockSize, rowBytes, bufferSize64) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Template routing: 64B 边界唯一归 key 1，空 tensor 归 key 0。
    const bool useSimdTemplate = (rowBytes >= SIMT_ROW_BYTES_THRESHOLD);
    bool overlapFill = false;
    TuneBufferSizeForUnderfill(rowBytes, static_cast<uint64_t>(v.n), coreNum, useSimdTemplate, ubBlockSize,
                               bufferSize64, overlapFill);
    const uint32_t bufferSize = static_cast<uint32_t>(bufferSize64); // ≤ ubHalf ≪ 0xFFFFFFFF
    uint64_t numActiveCores = 0;                                     // [cores]
    uint64_t perCoreChunks = 0;                                      // baseC [chunks]
    if (SplitMulticoreChunks(nodeName, v.n, rowBytes, bufferSize, coreNum, totalBytes, useSimdTemplate, overlapFill,
                             numActiveCores, perCoreChunks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (FinalizeTiling(context, nodeName, v.n, rowElems, rowBytes, totalBytes, v.dtypeSize, numActiveCores,
                       perCoreChunks, bufferSize, useSimdTemplate) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForParallelConcat(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ParallelConcat)
    .Tiling(TilingFuncParallelConcat)                                      // 运行期 tiling 入口
    .TilingParse<ParallelConcatCompileInfo>(TilingParseForParallelConcat); // 编译期 no-op（R15 载体）

} // namespace optiling
