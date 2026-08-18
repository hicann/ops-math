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
 * \file cdist_grad_tiling_arch22.cpp
 * \brief CdistGrad AscendC kernel Tiling for asc910b (arch22)
 *
 * All inputs arrive as broadcast [B, P, Q, M] continuous tensors (prepared by aclnn
 * UnsqueezeNd + BroadcastTo). Supports all p values: p=0, p=1, p=2, p=inf, general p.
 * fp32 and fp16, FullM mode only.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "tiling/tiling_api.h"
#include "math/cdist_grad/op_kernel/cdist_grad_tiling_data_arch22.h"
#include "math/cdist_grad/op_kernel/arch22/cdist_grad_tiling_key.h"

namespace optiling {

constexpr int64_t BLOCK_SIZE = 32;
// fp32 elements per 256B (Compare requires 256B-aligned count)
constexpr int64_t COMPARE_ALIGN = 64;
constexpr int64_t NUM_CHUNK_QUEUES = 3; // x2 / grad / dist
constexpr int64_t NUM_CHUNK_CPIES = 3;  // private per-chunk TBuf copies (both dtypes)
constexpr int64_t DOUBLE_BUFFER = 2;    // each chunk queue double buffered
// m-sized fp32 TBufs: x1Row/accum/diff/sign/powDst/zero/one/negOne (8)
// + outQueue/wsRead/rowInQueue (3). MUST match kernel InitBuffer exactly — an
// under-budgeted fixedBytes makes rTile too large and the last TBufs silently
// overlap the calc buffers (Duplicate(zero_) then corrupts chunk data).
constexpr int64_t NUM_FIXED_MBUF = 11;
// Safety margin: ccec adds hidden UB overhead (buffer alignment, queue management)
// on top of the explicit InitBuffer sizes. Budgeting to the last byte makes the
// final buffers silently overlap the calc buffers.
constexpr int64_t UB_SAFETY_MARGIN = 8192;

static inline int64_t AlignUp(int64_t val, int64_t align) { return ((val + align - 1) / align) * align; }

static inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static ge::graphStatus CdistGradTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSizeU64 = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU64);
    int64_t ubSize = static_cast<int64_t>(ubSizeU64);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);

    // Attr p → P_MODE (0=p1, 1=p2, 2=pinf, 3=pgeneral, 4=p0)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    float pValue = 2.0f;
    if (attrs->GetAttrNum() > 0) {
        const float* pAttr = attrs->GetAttrPointer<float>(0);
        pValue = (pAttr == nullptr) ? 2.0f : *pAttr;
    }
    uint32_t pModeInt = 1;
    if (pValue == 0.0f) {
        pModeInt = 4;
    } else if (pValue == 1.0f) {
        pModeInt = 0;
    } else if (pValue == 2.0f) {
        pModeInt = 1;
    } else if (std::isinf(pValue)) {
        pModeInt = 2;
    } else {
        pModeInt = 3;
    }

    // Shape after aclnn broadcast: grad storage = [B, P, Q, M].
    // B = product of leading dims, P = dim-3, Q = dim-2, M = dim-1.
    auto gradShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    auto gradStorageShape = gradShape->GetStorageShape();
    int64_t dimNum = static_cast<int64_t>(gradStorageShape.GetDimNum());
    OP_CHECK_IF(dimNum < 3,
                OP_LOGE(context->GetNodeName(), "CdistGrad requires at least 3D broadcast input, got: %ld", dimNum),
                return ge::GRAPH_FAILED);

    int64_t B = 1;
    for (int64_t d = 0; d < dimNum - 3; d++) {
        B *= gradStorageShape.GetDim(d);
    }
    int64_t P = gradStorageShape.GetDim(dimNum - 3);
    int64_t Q = gradStorageShape.GetDim(dimNum - 2);
    int64_t M = gradStorageShape.GetDim(dimNum - 1);

    OP_LOGI(context->GetNodeName(), "CdistGrad Tiling: B=%ld P=%ld Q=%ld M=%ld p=%.4f pMode=%u dimNum=%ld", B, P, Q, M,
            pValue, pModeInt, dimNum);

    // dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType inputDType = inputDesc->GetDataType();
    bool isFp16 = (inputDType == ge::DT_FLOAT16);
    int64_t inputTypeSize = isFp16 ? 2 : 4;

    // mAligned: fp32 element count aligned to 256B (64 fp32) — required by Compare API
    int64_t mAligned = AlignUp(M, COMPARE_ALIGN);
    int64_t mBytes = mAligned * 4; // one fp32 row in bytes

    // Power temporary space (used by p-general). Shared budget reserved for all p branches.
    std::vector<int64_t> shape1Vec = {mAligned};
    std::vector<int64_t> shape2Vec = {1};
    ge::Shape powShape1(shape1Vec);
    ge::Shape powShape2(shape2Vec);
    uint32_t tmpMax = 0;
    uint32_t tmpMin = 0;
    AscendC::GetPowerMaxMinTmpSize(powShape1, powShape2, false, 4, false, tmpMax, tmpMin);
    int64_t tmpBytes = static_cast<int64_t>(tmpMax);
    if (tmpBytes < BLOCK_SIZE)
        tmpBytes = BLOCK_SIZE;

    // ---- Joint mTile/rTile solve (M-tiling) ----
    // Prefer the largest mTile (fewest M segments); halve it until the per-segment
    // footprint fits. mTile floor of 64 floats always fits, so any M is supported.
    // Note: per-segment aligned width mTileAligned replaces mAligned in all buffer
    // sizing; the full-row aligned width (mAligned) is only the workspace stride.
    int64_t mTileSize = (M > COMPARE_ALIGN) ? ((M / COMPARE_ALIGN) * COMPARE_ALIGN) : M;
    int64_t mTileAligned = mAligned;
    int64_t mTileBytes = mBytes;
    int64_t rTile = 1;
    int64_t perTileBytes = 0;
    int64_t fixedBytesSeg = 0;
    while (true) {
        mTileAligned = AlignUp(mTileSize, COMPARE_ALIGN);
        mTileBytes = mTileAligned * 4;
        fixedBytesSeg = NUM_FIXED_MBUF * mTileBytes + AlignUp(mTileAligned / 8, BLOCK_SIZE) + (isFp16 ? mTileBytes : 0);
        perTileBytes = (NUM_CHUNK_QUEUES * DOUBLE_BUFFER + NUM_CHUNK_CPIES) * mTileBytes + mTileAligned / 8;
        int64_t avail = ubSize - fixedBytesSeg - tmpBytes - UB_SAFETY_MARGIN;
        if (avail >= perTileBytes) {
            rTile = avail / perTileBytes;
            if (rTile > Q)
                rTile = Q;
            break;
        }
        if (mTileSize <= COMPARE_ALIGN) {
            rTile = 1; // minimum footprint; guaranteed to fit for M-segment 64
            break;
        }
        // Keep mTileSize a multiple of 64: every non-tail segment then satisfies
        // mTileReal == mTileAligned and takes the contiguous-chunk fast path.
        mTileSize = std::max(COMPARE_ALIGN, (mTileSize / 2 / COMPARE_ALIGN) * COMPARE_ALIGN);
    }
    if (rTile < 1)
        rTile = 1;

    int64_t numMTiles = CeilDiv(M, mTileSize);
    int64_t lastMTileSize = M - (numMTiles - 1) * mTileSize;
    int64_t numRChunks = CeilDiv(Q, rTile);
    int64_t lastRChunkSize = Q - (numRChunks - 1) * rTile;
    if (lastRChunkSize <= 0)
        lastRChunkSize = rTile;

    // Multi-core split along B*P tasks; Q-split when B*P < coreNum (load balancing).
    int64_t totalTasks = B * P;
    int64_t qSplit = 1;
    if (totalTasks > 0 && totalTasks < coreNum && Q > 1) {
        qSplit = CeilDiv(coreNum, totalTasks);
        if (qSplit > Q)
            qSplit = Q;
    }
    int64_t qPartSize = CeilDiv(Q, qSplit);
    int64_t totalSubTasks = totalTasks * qSplit;
    int64_t tasksPerCore = CeilDiv(totalSubTasks, coreNum);
    int64_t usedCoreNum = CeilDiv(totalSubTasks, tasksPerCore);
    int64_t tailCoreTasks = totalSubTasks - (usedCoreNum - 1) * tasksPerCore;
    if (totalSubTasks <= 0) {
        tasksPerCore = 0;
        usedCoreNum = 1;
        tailCoreTasks = 0;
    }

    OP_LOGI(context->GetNodeName(),
            "UB=%ld M=%ld mAligned=%ld mTile=%ld numMTiles=%ld rTile=%ld numRChunks=%ld usedCoreNum=%ld "
            "tasksPerCore=%ld qSplit=%ld qPartSize=%ld",
            ubSize, M, mAligned, mTileSize, numMTiles, rTile, numRChunks, usedCoreNum, tasksPerCore, qSplit, qPartSize);

    // Workspace layout: [system workspace (GetLibApiWorkSpaceSize) | user workspace].
    // Always request at least the system part: a zero-size workspace yields an invalid
    // kernel workspace pointer on 910B and faults at launch.
    {
        size_t* workspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
        size_t usrSize = (qSplit > 1) ? static_cast<size_t>(totalSubTasks * mAligned * inputTypeSize) : 0;
        size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        workspace[0] = usrSize + sysWorkspaceSize;
    }

    // Fill TilingData
    auto tiling = context->GetTilingData<CdistGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CdistGradTilingData), 0, sizeof(CdistGradTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "memset_s tiling failed"), return ge::GRAPH_FAILED);

    tiling->batchSize = B;
    tiling->pSize = P;
    tiling->rSize = Q;
    tiling->mSize = M;
    tiling->mAligned = mAligned;
    tiling->mTileSize = mTileSize;
    tiling->numMTiles = numMTiles;
    tiling->lastMTileSize = lastMTileSize;
    tiling->rTile = rTile;
    tiling->numRChunks = numRChunks;
    tiling->lastRChunkSize = lastRChunkSize;
    tiling->tasksPerCore = tasksPerCore;
    tiling->tailCoreTasks = tailCoreTasks;
    tiling->usedCoreNum = usedCoreNum;
    tiling->qSplit = qSplit;
    tiling->qPartSize = qPartSize;
    tiling->tmpBufSize = tmpBytes;
    tiling->pValueF = pValue;

    // Q-split path: all cores participate in SyncAll-based two-phase reduce.
    if (qSplit > 1) {
        context->SetScheduleMode(1);
        context->SetBlockDim(coreNum);
    } else {
        context->SetBlockDim(usedCoreNum);
    }

    // SCH_MODE = 0 (FullM)
    uint32_t schMode = 0;
    uint32_t dType = static_cast<uint32_t>(inputDType);
    ASCENDC_TPL_SEL_PARAM(context, dType, pModeInt, schMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCdistGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CdistGradCompileInfo {};

IMPL_OP_OPTILING(CdistGrad).Tiling(CdistGradTilingFunc).TilingParse<CdistGradCompileInfo>(TilingParseForCdistGrad);

} // namespace optiling
