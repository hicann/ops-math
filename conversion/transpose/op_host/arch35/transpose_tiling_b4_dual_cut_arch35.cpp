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
 * \file transpose_tiling_b4_dual_cut_arch35.cpp
 * \brief HIF4 dual-cut tiling implementationi */

#include "transpose_tiling_b4_dual_cut_arch35.h"
#include <cmath>
#include <algorithm>

namespace optiling {
namespace TransHIF4DualCut {

// ---------------------------------------------------------------------------
// AxisCutter::FindCutAxis
//   Scan from innermost axis outward, accumulating product.
//   The first axis whose cumulative product exceeds ubBound is the cut axis.
// ---------------------------------------------------------------------------
void AxisCutter::FindCutAxis(int64_t ubBound)
{
    // Handle empty axis list (all axes already in UB)
    if (axisList_.empty()) {
        cutSet_ = 0;
        cutIdx_ = -1;
        cutFinal_ = false;
        innerCutAxis_.Reset(1, 1, 0);
        outerCutAxis_.Reset(1, 1, 0);
        return;
    }

    int32_t idx = static_cast<int32_t>(axisList_.size()) - 1;
    int64_t prod = 1;
    ubSet_ = 0;

    for (; idx >= 0; idx--) {
        prod *= axisList_[idx].dim;
        if (prod > ubBound) {
            break;
        }
        ubSet_ = ubSet_ | axisList_[idx].code;
    }

    cutSet_ = (idx >= 0) ? axisList_[idx].code : 0;
    cutIdx_ = idx;

    // If all axes fit in UB (idx == -1), no cutting needed
    if (idx < 0) {
        cutFinal_ = false;
        innerCutAxis_.Reset(1, 1, 0);
        outerCutAxis_.Reset(1, 1, 0);
        return;
    }

    AxisInf* cutAxisPtr = &axisList_[cutIdx_];
    // cutFinal_ = true only if the cut axis is THIS VIEW's final axis.
    // finalAxisIdx_ is set to axes.back().idx in the constructor,
    // which is the original input index of the view's last axis.
    cutFinal_ = (cutAxisPtr->idx == finalAxisIdx_);

    int64_t cutDim = cutAxisPtr->dim;
    // The constructor aligns the final axis up to alignedFactor_.
    // All calculations involving the "original" dimension must use
    // originFinalDim_ instead of the aligned cutDim.
    int64_t effectiveCutDim = cutFinal_ ? originFinalDim_ : cutDim;
    prod = prod / cutDim;

    // Avoid division by zero
    if (prod == 0) {
        prod = 1;
    }

    int64_t innerDim = ubBound / prod;
    // For HIF4: ensure even and at least 2 only for last axis
    if (cutFinal_ && (innerDim % NUM_TWO != 0)) {
        innerDim = (innerDim / NUM_TWO) * NUM_TWO;
    }
    if (innerDim == 0) {
        innerDim = cutFinal_ ? NUM_TWO : 1;
    }
    // innerDim cannot exceed the original (non-aligned) dimension
    if (innerDim > effectiveCutDim) {
        innerDim = effectiveCutDim;
    }

    int64_t outerDim = (effectiveCutDim + innerDim - 1) / innerDim;

    innerCutAxis_.Reset(innerDim, cutAxisPtr->stride, cutAxisPtr->idx);
    outerCutAxis_.Reset(outerDim, innerDim * cutAxisPtr->stride, cutAxisPtr->idx);
}

// ---------------------------------------------------------------------------
// AxisCutter::RemoveAxis
//   Remove axes whose bitmask intersects setCode from axisList_.
// ---------------------------------------------------------------------------
void AxisCutter::RemoveAxis(uint32_t setCode)
{
    std::vector<AxisInf> tmp;
    for (auto& a : axisList_) {
        if ((a.code & setCode) == 0) {
            tmp.push_back(a);
        }
    }
    axisList_ = tmp;
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::Init
//   inShape:  input tensor shape (original order, in HIF4 elements)
//   outShape: output tensor shape (transposed order, in HIF4 elements)
//   Both views use the same axis indices, no sorting needed.
//   alignedFactor = 64 HIF4 elements (32 bytes alignment)
// ---------------------------------------------------------------------------
void DualCutAxisSeeker::Init(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                             const std::vector<int64_t>& perm, int32_t dimNums, gert::TilingContext* context)
{
    context_ = context;
    dimNums_ = dimNums;
    outShape_ = outShape;
    perm_ = perm;
    // For HIF4: alignedFactor = 64 elements (32 bytes)
    alignedFactor_ = HIF4_ALIGN_ELEMENTS;

    // Build input strides (row-major, element units)
    std::vector<int64_t> inStrides(dimNums, 1);
    for (int32_t i = dimNums - NUM_TWO; i >= 0; i--) {
        inStrides[i] = inStrides[i + 1] * inShape[i + 1];
    }

    // Build output strides (row-major, element units)
    std::vector<int64_t> outStrides(dimNums, 1);
    for (int32_t i = dimNums - NUM_TWO; i >= 0; i--) {
        outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }

    // Initialize inputAxis_ with input shape and input strides (no sorting)
    inputAxis_.clear();
    for (int32_t i = 0; i < dimNums_; i++) {
        inputAxis_.emplace_back(inShape[i], inStrides[i], i);
    }

    // Initialize outputAxis_ in output order (outputAxis_[i] = output axis i)
    // idx = perm[i] indicates which input axis this output axis corresponds to
    // stride = input stride of the corresponding input axis (for as_strided compatibility)
    outputAxis_.clear();
    for (int32_t i = 0; i < dimNums_; i++) {
        int32_t inputAxisIdx = static_cast<int32_t>(perm[i]);
        int64_t inputStride = inStrides[inputAxisIdx];
        outputAxis_.emplace_back(outShape[i], inputStride, inputAxisIdx);
    }

    inputCutter_.Init(inputAxis_, alignedFactor_);
    outputCutter_.Init(outputAxis_, alignedFactor_);

    OP_LOGD(context_, "[HIF4DualCut] DualCutAxisSeeker: dimNums=%d, alignedFactor=%d", dimNums_, alignedFactor_);
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::ComputeSetAxisProd
// ---------------------------------------------------------------------------
int64_t DualCutAxisSeeker::ComputeSetAxisProd(uint32_t axisSet) const
{
    int64_t factor = 1;
    for (const auto& a : outputAxis_) {
        if ((a.code & axisSet) != 0) {
            factor *= a.dim;
            if (factor <= 0) {
                OP_LOGW(context_, "[HIF4DualCut] ComputeSetAxisProd overflow, axisSet=0x%x", axisSet);
                return INT64_MAX;
            }
        }
    }
    return factor;
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::ComputeRemainUB
//   Compute remaining UB space after placing axes in axisSet.
//   IMPORTANT: Must consider alignment for BOTH input and output final axes.
//   For transpose, input final axis and output final axis are different axes.
// ---------------------------------------------------------------------------
int64_t DualCutAxisSeeker::ComputeRemainUB(int64_t ubNum, uint32_t axisSet) const
{
    // Identify input and output final axes
    uint32_t inFinalCode = inputAxis_[dimNums_ - 1].code;   // input axis (dimNums-1)
    uint32_t outFinalCode = outputAxis_[dimNums_ - 1].code; // output axis (dimNums-1) -> input axis perm[dimNums-1]

    // Compute product of all axes in axisSet
    int64_t prod = 1;
    for (const auto& a : outputAxis_) {
        if ((a.code & axisSet) != 0) {
            // Check if this axis needs alignment
            bool needAlign = (a.code == inFinalCode) || (a.code == outFinalCode);
            if (needAlign) {
                int64_t alignedDim = ((a.dim + alignedFactor_ - 1) / alignedFactor_) * alignedFactor_;
                prod *= alignedDim;
            } else {
                prod *= a.dim;
            }
        }
    }

    if (prod <= 0) {
        OP_LOGW(context_, "[HIF4DualCut] ComputeRemainUB: prod overflow detected, axisSet=0x%x, prod=%ld", axisSet,
                prod);
        return 0;
    }
    return ubNum / prod;
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::FindCommonUbAxis / UpdateCommonAxis
// ---------------------------------------------------------------------------
uint32_t DualCutAxisSeeker::FindCommonUbAxis(const AxisCutter* c1, const AxisCutter* c2) const
{
    uint32_t commonUb = c1->ubSet_ & c2->ubSet_;
    uint32_t cutCommon1 = c1->cutSet_ & c2->ubSet_;
    uint32_t cutCommon2 = c2->cutSet_ & c1->ubSet_;
    return commonUb | cutCommon1 | cutCommon2;
}

void DualCutAxisSeeker::UpdateCommonAxis(uint32_t commonSet)
{
    outputCutter_.RemoveAxis(commonSet);
    inputCutter_.RemoveAxis(commonSet);
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::FindDualCutAxis
//   Main loop: iteratively find common axes between output/input cutters,
//   absorb them into ubAxisSet, then call CutAxis to finalize gmAxis_/ubAxis_.
//   For HIF4: ubSize is in bytes, shapes are in HIF4 elements (2 elements per byte)
// ---------------------------------------------------------------------------
bool DualCutAxisSeeker::FindDualCutAxis(int32_t ubSize, int32_t bufferNum)
{
    ubSize_ = ubSize;

    if (bufferNum == 0) {
        OP_LOGW(context_, "[HIF4DualCut] bufferNum == 0, reset to 2");
        bufferNum = NUM_TWO;
    }

    // ubNum: available UB in HIF4 elements
    // ubSize is in bytes, 2 HIF4 elements per byte
    int64_t ubNum = static_cast<int64_t>(ubSize) * HIF4_ELEMENTS_PER_BYTE / bufferNum;
    OP_LOGD(context_,
            "[HIF4DualCut] FindDualCutAxis start: ubSize=%d bytes, bufferNum=%d, ubNum=%ld HIF4 elements, "
            "alignedFactor=%d, lastDim=%ld",
            ubSize, bufferNum, ubNum, alignedFactor_, outputAxis_[dimNums_ - 1].dim);

    // Note: Allow dual-cut even if last dim < alignedFactor
    // The alignment will be handled during axis cutting

    uint32_t ubAxisSet = 0;
    uint32_t joinUbAxisSet = 0;

    for (int32_t findLoops = DUAL_CUT_GM_LEN + DUAL_CUT_UB_LEN; findLoops >= 0; findLoops--) {
        int64_t remainUB = ComputeRemainUB(ubNum, ubAxisSet);
        if (remainUB <= 0) {
            break;
        }
        int64_t ubBound = static_cast<int64_t>(std::floor(std::sqrt(static_cast<double>(remainUB))));
        if (ubBound <= 0) {
            break;
        }

        outputCutter_.FindCutAxis(ubBound);
        inputCutter_.FindCutAxis(ubBound);

        uint32_t commonUbAxisSet = FindCommonUbAxis(&outputCutter_, &inputCutter_);
        joinUbAxisSet = outputCutter_.ubSet_ | inputCutter_.ubSet_;
        if (commonUbAxisSet == 0) {
            // No shared UB axis: two independent cut axes, or both cutters cut the same axis.
            bool sameCutAxis = (inputCutter_.cutSet_ & outputCutter_.cutSet_) != 0;
            OP_LOGD(context_, "[HIF4DualCut] Search stop: %s.", sameCutAxis ? "one cut axis" : "two cut axes");
            break;
        } else {
            ubAxisSet = ubAxisSet | commonUbAxisSet;
            UpdateCommonAxis(commonUbAxisSet);
        }
    }

    ubAxisSet = ubAxisSet | joinUbAxisSet;
    int64_t remainNums = ComputeRemainUB(ubNum, ubAxisSet);
    OP_LOGD(context_, "[HIF4DualCut] FindDualCutAxis done: remainNums=%ld", remainNums);
    return CutAxis(ubAxisSet, remainNums);
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::CutAxis
//   Orchestrate the cut, then build gmAxis_ / ubAxis_.
// ---------------------------------------------------------------------------
bool DualCutAxisSeeker::CutAxis(uint32_t ubAxisSet, int64_t remainNums)
{
    std::vector<AxisInf> innerAxis;
    std::vector<AxisInf> outerAxis;

    uint32_t cutAxisSet = inputCutter_.cutSet_ | outputCutter_.cutSet_;
    DispatchCut(ubAxisSet, remainNums, cutAxisSet, innerAxis, outerAxis);

    // Filter out "fake" cuts where outerDim == 1 (treat as full axis in UB)
    FilterFakeCuts(ubAxisSet, innerAxis, outerAxis);

    // Partition outputAxis_ into gmAxis_ / ubAxis_
    FillGmUbAxis(ubAxisSet, cutAxisSet);

    // If ub axes + inner cut axes exceed MOVE_ALIGN limit, move axes to gm.
    EvictExcessUbAxis(innerAxis);

    AppendCutAxes(innerAxis, outerAxis);
    SortGmUbAxis();

    // Validate actual UB footprint with final-axis alignment.
    ShrinkInnerCutDims(innerAxis, outerAxis);

    // Handle "all in UB" case: if gmAxis_ is empty, add a dummy axis with dim=1
    if (gmAxis_.empty()) {
        gmAxis_.emplace_back(1, 1, 0);
    }

    return true;
}

// Dispatch to CutTwoAxis/CutOneAxis, falling back to a 0-axis cut on failure.
void DualCutAxisSeeker::DispatchCut(uint32_t& ubAxisSet, int64_t remainNums, uint32_t cutAxisSet,
                                    std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis)
{
    bool ok = true;
    if (cutAxisSet == 0) {
        OP_LOGD(context_, "[HIF4DualCut] CutAxis: all axes fit in UB.");
        // No inner/outer axis to add
    } else if (outputCutter_.cutSet_ != inputCutter_.cutSet_) {
        OP_LOGD(context_, "[HIF4DualCut] CutAxis: CutTwoAxis.");
        ok = CutTwoAxis(ubAxisSet, remainNums, innerAxis, outerAxis);
        if (!ok) {
            // Fallback: treat both cut axes as UB axes (0-axis cut)
            OP_LOGW(context_, "[HIF4DualCut] CutTwoAxis failed, treat cut axes as UB axes.");
            ubAxisSet |= cutAxisSet;
            innerAxis.clear();
            outerAxis.clear();
        }
    } else {
        OP_LOGD(context_, "[HIF4DualCut] CutAxis: CutOneAxis.");
        ok = CutOneAxis(ubAxisSet, remainNums, innerAxis, outerAxis);
        if (!ok) {
            // Fallback: treat cut axis as UB axis (0-axis cut)
            OP_LOGW(context_, "[HIF4DualCut] CutOneAxis failed, treat cut axis as UB axis.");
            ubAxisSet |= cutAxisSet;
            innerAxis.clear();
            outerAxis.clear();
        }
    }
}

// Drop "fake" cuts where outerDim == 1 (the axis is essentially uncut).
void DualCutAxisSeeker::FilterFakeCuts(uint32_t& ubAxisSet, std::vector<AxisInf>& innerAxis,
                                       std::vector<AxisInf>& outerAxis)
{
    std::vector<AxisInf> filteredInner;
    std::vector<AxisInf> filteredOuter;
    for (size_t i = 0; i < innerAxis.size(); i++) {
        if (outerAxis[i].dim == 1) {
            // Fake cut: outer=1 means the axis is essentially uncut
            // Add the original full axis to ubAxisSet instead
            ubAxisSet |= outerAxis[i].code;
        } else {
            filteredInner.push_back(innerAxis[i]);
            filteredOuter.push_back(outerAxis[i]);
        }
    }
    innerAxis = filteredInner;
    outerAxis = filteredOuter;

    cutAxisNums_ = static_cast<int32_t>(innerAxis.size());
}

// Partition outputAxis_ into UB axes (ubAxisSet) and GM axes (not cut).
void DualCutAxisSeeker::FillGmUbAxis(uint32_t ubAxisSet, uint32_t cutAxisSet)
{
    for (const auto& a : outputAxis_) {
        if ((a.code & ubAxisSet) != 0) {
            ubAxis_.push_back(a);
        } else if ((a.code & cutAxisSet) == 0) {
            gmAxis_.push_back(a);
        }
    }
}

// If ub axes + inner cut axes exceed the MOVE_ALIGN limit, move axes to gm.
// Priority: evict non-final axes first; always keep the output final axis in UB.
void DualCutAxisSeeker::EvictExcessUbAxis(const std::vector<AxisInf>& innerAxis)
{
    if (ubAxis_.size() + innerAxis.size() <= static_cast<size_t>(DUAL_CUT_UB_LEN)) {
        return;
    }

    uint32_t inFinalCode = inputAxis_[dimNums_ - 1].code;
    uint32_t outFinalCode = outputAxis_[dimNums_ - 1].code;

    int32_t excess = static_cast<int32_t>(ubAxis_.size() + innerAxis.size()) - DUAL_CUT_UB_LEN;

    // Separate non-final and final axes
    std::vector<AxisInf> nonFinal;
    std::vector<AxisInf> finals;
    for (auto& a : ubAxis_) {
        if ((a.code == inFinalCode) || (a.code == outFinalCode)) {
            finals.push_back(a);
        } else {
            nonFinal.push_back(a);
        }
    }

    // Evict non-final axes first (largest stride first to keep small strides in UB)
    std::sort(nonFinal.begin(), nonFinal.end(), [](const AxisInf& a, const AxisInf& b) {
        return (a.stride == b.stride) ? (a.dim > b.dim) : (a.stride < b.stride);
    });
    int32_t evicted = 0;
    while (evicted < excess && !nonFinal.empty()) {
        gmAxis_.push_back(nonFinal.back());
        nonFinal.pop_back();
        evicted++;
    }

    // If still excess, evict final axes (prefer input final over output final)
    while (evicted < excess && !finals.empty()) {
        gmAxis_.push_back(finals.back());
        finals.pop_back();
        evicted++;
    }

    // Rebuild ubAxis_
    ubAxis_.clear();
    ubAxis_.insert(ubAxis_.end(), nonFinal.begin(), nonFinal.end());
    ubAxis_.insert(ubAxis_.end(), finals.begin(), finals.end());
}

void DualCutAxisSeeker::AppendCutAxes(const std::vector<AxisInf>& innerAxis, const std::vector<AxisInf>& outerAxis)
{
    for (const auto& a : innerAxis) {
        ubAxis_.push_back(a);
    }
    for (const auto& a : outerAxis) {
        gmAxis_.push_back(a);
    }
}

void DualCutAxisSeeker::SortGmUbAxis()
{
    // Sort both gmAxis_ and ubAxis_ by input axis index (inShape order)
    std::sort(gmAxis_.begin(), gmAxis_.end(), [](const AxisInf& a, const AxisInf& b) { return a.idx < b.idx; });
    std::sort(ubAxis_.begin(), ubAxis_.end(), [](const AxisInf& a, const AxisInf& b) { return a.idx < b.idx; });
}

// Validate the actual UB footprint with final-axis alignment; shrink inner
// cut dims if it overflows.
//   ND-DMA requires both input and output innermost axes to be aligned to
//   alignedFactor_.  Fake-cut axes that end up in UB must also use their
//   aligned dims when checking capacity.
void DualCutAxisSeeker::ShrinkInnerCutDims(std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis)
{
    int64_t ubNum = static_cast<int64_t>(ubSize_) * HIF4_ELEMENTS_PER_BYTE / HIF4_BUFFER_NUM;
    int64_t alignedProd = ComputeAlignedUbProd();
    if (alignedProd <= ubNum) {
        return;
    }

    OP_LOGW(context_,
            "[HIF4DualCut] UB overflow after fake cuts: alignedProd=%ld > ubNum=%ld, "
            "shrinking inner cut dims",
            alignedProd, ubNum);
    int64_t ratio = (alignedProd + ubNum - 1) / ubNum; // ceil
    if (ratio < NUM_TWO) {
        ratio = NUM_TWO;
    }

    for (size_t i = 0; i < innerAxis.size(); i++) {
        ShrinkSingleCutAxis(i, ratio, innerAxis, outerAxis);
    }
}

// Aligned product of UB dims (final axes rounded up to alignedFactor_).
int64_t DualCutAxisSeeker::ComputeAlignedUbProd() const
{
    int64_t alignedProd = 1;
    for (const auto& a : ubAxis_) {
        bool isInFinal = (a.code == inputAxis_[dimNums_ - 1].code);
        bool isOutFinal = (a.code == outputAxis_[dimNums_ - 1].code);
        int64_t dim = a.dim;
        if (isInFinal || isOutFinal) {
            dim = ((dim + alignedFactor_ - 1) / alignedFactor_) * alignedFactor_;
        }
        alignedProd *= dim;
    }
    return alignedProd;
}

// Shrink one inner cut axis by ratio and sync its dim into ubAxis_ / gmAxis_.
void DualCutAxisSeeker::ShrinkSingleCutAxis(size_t i, int64_t ratio, std::vector<AxisInf>& innerAxis,
                                            std::vector<AxisInf>& outerAxis)
{
    int64_t newDim = innerAxis[i].dim / ratio;
    if (newDim < 1) {
        newDim = 1;
    }

    bool isInFinal = (innerAxis[i].code == inputAxis_[dimNums_ - 1].code);
    bool isOutFinal = (innerAxis[i].code == outputAxis_[dimNums_ - 1].code);
    if (isInFinal || isOutFinal) {
        if (newDim % NUM_TWO != 0) {
            newDim = (newDim / NUM_TWO) * NUM_TWO;
        }
        if (newDim < NUM_TWO) {
            newDim = NUM_TWO;
        }
    }
    innerAxis[i].dim = newDim;

    // Recompute outer dim from original dim
    int64_t origDim = 0;
    if (isInFinal) {
        origDim = inputCutter_.originFinalDim_;
    } else if (isOutFinal) {
        origDim = outputCutter_.originFinalDim_;
    } else {
        for (const auto& oa : outputAxis_) {
            if (oa.code == innerAxis[i].code) {
                origDim = oa.dim;
                break;
            }
        }
    }
    if (origDim == 0) {
        origDim = innerAxis[i].dim;
    }
    outerAxis[i].dim = (origDim + innerAxis[i].dim - 1) / innerAxis[i].dim;
    outerAxis[i].stride = innerAxis[i].dim * innerAxis[i].stride;

    // Sync the dim in ubAxis_ (find by matching code)
    for (auto& ua : ubAxis_) {
        if (ua.code == innerAxis[i].code) {
            ua.dim = innerAxis[i].dim;
            break;
        }
    }
    // Sync the outer dim in gmAxis_ (find by matching code)
    for (auto& ga : gmAxis_) {
        if (ga.code == innerAxis[i].code) {
            ga.dim = outerAxis[i].dim;
            ga.stride = outerAxis[i].stride;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::CutTwoAxis
// ---------------------------------------------------------------------------
bool DualCutAxisSeeker::CutTwoAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis,
                                   std::vector<AxisInf>& outerAxis)
{
    (void)ubAxisSet;
    AxisInf outInner = outputCutter_.innerCutAxis_;
    AxisInf outOuter = outputCutter_.outerCutAxis_;
    AxisInf inInner = inputCutter_.innerCutAxis_;
    AxisInf inOuter = inputCutter_.outerCutAxis_;

    int32_t inCutAxisIdx = (inputCutter_.cutIdx_ >= 0) ? inputCutter_.axisList_[inputCutter_.cutIdx_].idx : -1;
    int32_t outCutAxisIdx = (outputCutter_.cutIdx_ >= 0) ? outputCutter_.axisList_[outputCutter_.cutIdx_].idx : -1;

    // Pre-compute original dimensions (use originFinalDim_ for final axis because
    // AxisCutter aligns the final axis up to alignedFactor_)
    int64_t inOrigDim = GetCutterOrigDim(&inputCutter_);
    int64_t outOrigDim = GetCutterOrigDim(&outputCutter_);
    if (inOrigDim == 0) {
        inOrigDim = inInner.dim;
    }
    if (outOrigDim == 0) {
        outOrigDim = outInner.dim;
    }

    // Decide the inner dims (alignment + even/min-2 rules depending on final axes)
    ComputeCutInnerDims(inInner, outInner, inOrigDim, outOrigDim, inCutAxisIdx, outCutAxisIdx, remainNums);

    // Clamp inner dims to original dims (no cut if inner >= original)
    if (inInner.dim > inOrigDim) {
        inInner.dim = inOrigDim;
    }
    if (outInner.dim > outOrigDim) {
        outInner.dim = outOrigDim;
    }

    inOuter.dim = (inOrigDim + inInner.dim - 1) / inInner.dim;
    outOuter.dim = (outOrigDim + outInner.dim - 1) / outInner.dim;

    inOuter.stride = inInner.dim * inInner.stride;
    outOuter.stride = outInner.dim * outInner.stride;

    outerAxis.push_back(outOuter);
    outerAxis.push_back(inOuter);
    innerAxis.push_back(outInner);
    innerAxis.push_back(inInner);
    return true;
}

// True when axisIdx is the last axis or the axis transposed to last position.
bool DualCutAxisSeeker::NeedEvenDim(int32_t axisIdx) const
{
    int32_t lastAxis = dimNums_ - 1;
    int32_t transposedLastAxis = static_cast<int32_t>(perm_[lastAxis]);
    return (axisIdx == lastAxis) || (axisIdx == transposedLastAxis);
}

// Ensure even and >= 2 only for the last axis / transposed-last axis.
void DualCutAxisSeeker::FixEvenMin2IfNeeded(int64_t& v, int32_t axisIdx) const
{
    if (NeedEvenDim(axisIdx)) {
        if (v % NUM_TWO != 0) {
            v = (v / NUM_TWO) * NUM_TWO;
        }
        if (v <= 0) {
            v = NUM_TWO;
        }
    } else {
        if (v <= 0) {
            v = 1;
        }
    }
}

// Original (non-aligned) dim of the cutter's cut axis.
int64_t DualCutAxisSeeker::GetCutterOrigDim(const AxisCutter* cutter) const
{
    if (cutter->cutIdx_ < 0) {
        return 0;
    }
    if (cutter->cutFinal_) {
        return cutter->originFinalDim_;
    }
    return cutter->axisList_[cutter->cutIdx_].dim;
}

// Align the inner dim for a final axis, guarding against fake cut / overflow.
void DualCutAxisSeeker::AlignInnerDimIfNeeded(int64_t& inner, int64_t orig, int32_t axisIdx, bool isFinal,
                                              int64_t remainNums) const
{
    if (!isFinal) {
        FixEvenMin2IfNeeded(inner, axisIdx);
        return;
    }
    if (inner >= alignedFactor_) {
        inner = (inner / alignedFactor_) * alignedFactor_;
    } else if (orig >= alignedFactor_ && remainNums >= static_cast<int64_t>(alignedFactor_) * NUM_TWO) {
        inner = alignedFactor_;
    }
    FixEvenMin2IfNeeded(inner, axisIdx);
    // If alignment causes fake cut (inner >= original), revert to original
    if (inner >= orig) {
        inner = orig;
        FixEvenMin2IfNeeded(inner, axisIdx);
    }
}

// Compute inInner/outInner dims for CutTwoAxis, honoring which cut axes are final.
void DualCutAxisSeeker::ComputeCutInnerDims(AxisInf& inInner, AxisInf& outInner, int64_t inOrigDim, int64_t outOrigDim,
                                            int32_t inCutAxisIdx, int32_t outCutAxisIdx, int64_t remainNums)
{
    if (inputCutter_.cutFinal_ || outputCutter_.cutFinal_) {
        if (inputCutter_.cutFinal_) {
            AlignInnerDimIfNeeded(inInner.dim, inOrigDim, inCutAxisIdx, true, remainNums);
            outInner.dim = remainNums / inInner.dim;
            if (outInner.dim <= 0) {
                outInner.dim = NUM_TWO;
                inInner.dim = remainNums / outInner.dim;
                AlignInnerDimIfNeeded(inInner.dim, inOrigDim, inCutAxisIdx, true, remainNums);
            }
            FixEvenMin2IfNeeded(outInner.dim, outCutAxisIdx);
            if (outputCutter_.cutFinal_) {
                int64_t newInInner = remainNums / outInner.dim;
                if (newInInner > 0) {
                    inInner.dim = newInInner;
                    AlignInnerDimIfNeeded(inInner.dim, inOrigDim, inCutAxisIdx, true, remainNums);
                }
            }
        } else {
            // outputCutter_.cutFinal_
            AlignInnerDimIfNeeded(outInner.dim, outOrigDim, outCutAxisIdx, true, remainNums);
            inInner.dim = remainNums / outInner.dim;
            if (inInner.dim <= 0) {
                inInner.dim = NUM_TWO;
                outInner.dim = remainNums / inInner.dim;
                AlignInnerDimIfNeeded(outInner.dim, outOrigDim, outCutAxisIdx, true, remainNums);
            }
            FixEvenMin2IfNeeded(inInner.dim, inCutAxisIdx);
            if (inputCutter_.cutFinal_) {
                int64_t newOutInner = remainNums / inInner.dim;
                if (newOutInner > 0) {
                    outInner.dim = newOutInner;
                    AlignInnerDimIfNeeded(outInner.dim, outOrigDim, outCutAxisIdx, true, remainNums);
                }
            }
        }
    } else {
        FixEvenMin2IfNeeded(inInner.dim, inCutAxisIdx);
        FixEvenMin2IfNeeded(outInner.dim, outCutAxisIdx);
    }
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::CutOneAxis
// ---------------------------------------------------------------------------
bool DualCutAxisSeeker::CutOneAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis,
                                   std::vector<AxisInf>& outerAxis)
{
    (void)ubAxisSet;
    int64_t innerDim = inputCutter_.cutFinal_ ? (remainNums / alignedFactor_ * alignedFactor_) : remainNums;

    AxisInf* cutPtr = (inputCutter_.cutIdx_ >= 0) ? &inputCutter_.axisList_[inputCutter_.cutIdx_] :
                                                    &inputCutter_.axisList_[0];

    int32_t axisIdx = cutPtr->idx;
    int32_t lastAxis = dimNums_ - 1;
    int32_t transposedLastAxis = static_cast<int32_t>(perm_[lastAxis]);
    bool needEven = (axisIdx == lastAxis) || (axisIdx == transposedLastAxis);

    // For HIF4: ensure even number only for last axis or transposed-last axis
    if (needEven && (innerDim % NUM_TWO != 0)) {
        innerDim = (innerDim / NUM_TWO) * NUM_TWO;
    }
    // Ensure minimum value
    if (innerDim <= 0) {
        innerDim = needEven ? NUM_TWO : 1;
    }

    // Get original dim (originFinalDim_ for final axis because constructor aligns it)
    int64_t origDim = (inputCutter_.cutFinal_) ? inputCutter_.originFinalDim_ : cutPtr->dim;

    // If innerDim >= original dim, treat as no-cut (outerDim will be 1, filtered later)
    if (innerDim >= origDim) {
        innerDim = origDim;
    }

    int64_t innerStride = cutPtr->stride;
    int64_t outerDim = (origDim + innerDim - 1) / innerDim;
    int64_t outerStride = innerStride * innerDim;

    innerAxis.emplace_back(innerDim, innerStride, axisIdx);
    outerAxis.emplace_back(outerDim, outerStride, axisIdx);
    return true;
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::SetCutAxisIdx
//   After gmAxis_/ubAxis_ are finalized, find which ubAxis slot corresponds
//   to each cut axis and compute the tail size.
// ---------------------------------------------------------------------------
void DualCutAxisSeeker::SetCutAxisIdx()
{
    int32_t cutAxisIdx[2] = {-1, -1};
    int32_t cutAxisTail[2] = {-1, -1};

    // gmAxis_ is sorted by input axis index, so cut axes are interleaved with non-cut axes.
    // Scan gmAxis_ in order and find axes whose code also appears in ubAxis_ (cut axes).
    int32_t found = 0;
    for (int32_t k = 0; k < outerAxisNums_ && found < cutAxisNums_; k++) {
        for (int32_t j = 0; j < static_cast<int32_t>(ubAxis_.size()); j++) {
            if (ubAxis_[j].code == gmAxis_[k].code) {
                // Use merged axis id (ubAxis_[j].idx) instead of array position
                cutAxisIdx[found] = ubAxis_[j].idx;
                // tail = original dim - (outerDim - 1) * innerDim
                cutAxisTail[found] = static_cast<int32_t>(inputAxis_[gmAxis_[k].idx].dim -
                                                          (gmAxis_[k].dim - 1) * ubAxis_[j].dim);
                found++;
                break;
            }
        }
    }
    cutAxisIdx01_ = cutAxisIdx[0];
    cutAxisIdx02_ = cutAxisIdx[1];
    cutAxisTail01_ = cutAxisTail[0];
    cutAxisTail02_ = cutAxisTail[1];
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::GenTilingData
//   Populate gmShape_/gmInStride_/gmOutStride_ and ubShape_/ubInStride_/ubOutStride_.
//   IMPORTANT: All UB arrays must be in INPUT AXIS ORDER for ND-DMA to work correctly:
//   - ubShape: UB axes in input axis order
//   - ubInStride: input stride for each input axis
//   - ubOutStride: output stride for each input axis (where it appears in output)
// ---------------------------------------------------------------------------
void DualCutAxisSeeker::GenTilingData()
{
    outerAxisNums_ = static_cast<int32_t>(gmAxis_.size());
    innerAxisNums_ = static_cast<int32_t>(ubAxis_.size());

    FillGmArrays();
    FillUbShapeAndId();
    FillUbStrides();
    FillUbPermAndMoveStride();
    FillCutIds();
    FixGmOutStrideForCuts();

    SetCutAxisIdx();
}

// Output position (index in output order) of the given input axis, or -1.
int32_t DualCutAxisSeeker::FindOutputPos(int32_t axisIdx) const
{
    for (int32_t j = 0; j < dimNums_; j++) {
        if (static_cast<int32_t>(perm_[j]) == axisIdx) {
            return j;
        }
    }
    return -1;
}

// Fill gmShape_ / gmInStride_ / gmOutStride_ (right-aligned).
void DualCutAxisSeeker::FillGmArrays()
{
    int32_t gmBaseIdx = DUAL_CUT_GM_LEN - outerAxisNums_;
    for (int32_t i = 0; i < outerAxisNums_; i++) {
        int32_t slot = gmBaseIdx + i;
        gmShape_[slot] = static_cast<int32_t>(gmAxis_[i].dim);
        gmInStride_[slot] = gmAxis_[i].stride; // input stride from outputAxis
        // For dummy axis (dim=1), outStride should be 1 (no skipping)
        if (gmAxis_[i].dim == 1) {
            gmOutStride_[slot] = 1;
        } else {
            gmOutStride_[slot] = ComputeOutputStride(gmAxis_[i]); // compute output stride
        }
    }
}

// Fill ubShape_ (input axis order) and ubShapeId_ (-1 for unused slots).
void DualCutAxisSeeker::FillUbShapeAndId()
{
    int32_t baseIdx = DUAL_CUT_UB_LEN - innerAxisNums_;
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        ubShape_[baseIdx + i] = static_cast<int32_t>(ubAxis_[i].dim);
    }
    for (int32_t i = 0; i < DUAL_CUT_UB_LEN; i++) {
        ubShapeId_[i] = -1;
    }
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        ubShapeId_[baseIdx + i] = ubAxis_[i].idx;
    }
}

// Fill ubInStride_ / ubOutStride_ aligned with ubShape_.
void DualCutAxisSeeker::FillUbStrides()
{
    int32_t baseIdx = DUAL_CUT_UB_LEN - innerAxisNums_;
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t slot = baseIdx + i;
        int32_t axisIdx = ubAxis_[i].idx;

        // Input stride for this input axis
        ubInStride_[slot] = inputAxis_[axisIdx].stride;

        // Output stride: product of output dims to the right of this axis's output position
        int32_t outPos = FindOutputPos(axisIdx);
        if (outPos < 0) {
            OP_LOGW(context_, "[HIF4DualCut] GenTilingData: axis idx=%d not found in perm!", axisIdx);
        }
        int64_t outStride = 1;
        for (int32_t j = outPos + 1; j < dimNums_; j++) {
            outStride *= outShape_[j];
        }
        ubOutStride_[slot] = outStride;
    }
}

// Fill ubPerm_ / moveUbStride_: ub axes reordered by output position.
void DualCutAxisSeeker::FillUbPermAndMoveStride()
{
    int32_t baseIdx = DUAL_CUT_UB_LEN - innerAxisNums_;
    std::vector<std::pair<int32_t, int32_t>> ubOutOrder; // (outPos, ubAxisIndex)
    ubOutOrder.reserve(innerAxisNums_);
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t axisIdx = ubAxis_[i].idx;
        int32_t outPos = FindOutputPos(axisIdx);
        if (outPos < 0) {
            OP_LOGW(context_, "[HIF4DualCut] GenTilingData: ubAxis idx=%d not found in perm!", axisIdx);
            outPos = 0;
        }
        ubOutOrder.emplace_back(outPos, i);
    }
    std::sort(ubOutOrder.begin(), ubOutOrder.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Fill dummy positions (before baseIdx) as identity permutation
    for (int32_t i = 0; i < baseIdx; i++) {
        ubPerm_[i] = i;
    }

    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t slot = baseIdx + i;
        int32_t outPos = ubOutOrder[i].first;
        int32_t ubIdx = ubOutOrder[i].second;

        // ubPerm: output-order axis -> source slot index in ubShape_
        ubPerm_[slot] = baseIdx + ubIdx;

        // moveUbStride: product of output dims to the right of outPos
        int64_t stride = 1;
        for (int32_t j = outPos + 1; j < dimNums_; j++) {
            stride *= outShape_[j];
        }
        moveUbStride_[slot] = stride;
    }
}

// Fill outCutId_ / inCutId_ (position within gmAxis_).
void DualCutAxisSeeker::FillCutIds()
{
    // Collect all real cut axes from gmAxis_ (those whose code also appears in ubAxis_).
    std::vector<int32_t> cutAxisPositions;
    for (int32_t k = 0; k < outerAxisNums_; k++) {
        for (const auto& ua : ubAxis_) {
            if (ua.code == gmAxis_[k].code) {
                cutAxisPositions.push_back(k);
                break;
            }
        }
    }
    outCutId_ = -1;
    inCutId_ = -1;
    if (cutAxisNums_ == 1) {
        // Single real cut axis: both outCut and inCut point to it
        if (!cutAxisPositions.empty()) {
            outCutId_ = cutAxisPositions[0];
            inCutId_ = cutAxisPositions[0];
        }
    } else if (cutAxisNums_ == NUM_TWO) {
        // Two real cut axes: match by cutter cutSet_ bitmask
        for (int32_t pos : cutAxisPositions) {
            if ((gmAxis_[pos].code & outputCutter_.cutSet_) != 0) {
                outCutId_ = pos;
            }
            if ((gmAxis_[pos].code & inputCutter_.cutSet_) != 0) {
                inCutId_ = pos;
            }
        }
    }
}

// Multiply gmOutStride_ of cut outer axes by the corresponding inner dim.
void DualCutAxisSeeker::FixGmOutStrideForCuts()
{
    int32_t gmBaseIdx = DUAL_CUT_GM_LEN - outerAxisNums_;
    for (int32_t k = 0; k < outerAxisNums_; k++) {
        uint32_t axisCode = gmAxis_[k].code;
        for (const auto& ua : ubAxis_) {
            if (ua.code == axisCode) {
                int32_t slot = gmBaseIdx + k;
                gmOutStride_[slot] *= static_cast<int64_t>(ua.dim);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::ComputeOutputStride
//   Compute output stride for a given axis (similar to as_strided's ComputeOutputShape).
//   For transpose: find the output position of this input axis, then compute stride
//   as the product of all output dims to the right of that position.
// ---------------------------------------------------------------------------
int64_t DualCutAxisSeeker::ComputeOutputStride(const AxisInf& axis) const
{
    // Find output position of this input axis
    int32_t outPos = FindOutputPos(axis.idx);
    if (outPos < 0) {
        OP_LOGW(context_, "[HIF4DualCut] ComputeOutputStride: axis idx=%d not found in perm!", axis.idx);
        return 1;
    }

    // Compute stride: product of all output dims to the right
    int64_t prod = 1;
    for (int32_t i = outPos + 1; i < dimNums_; i++) {
        prod *= outShape_[i];
    }
    return prod;
}

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::ComputeBlockTiling
//   NK10: distribute total outer iterations across coreNum cores.
// ---------------------------------------------------------------------------
void DualCutAxisSeeker::ComputeBlockTiling(int32_t coreNum)
{
    int64_t totalOuter = 1;
    for (const auto& a : gmAxis_) {
        totalOuter *= a.dim;
    }
    dimsPerCore_ = static_cast<int32_t>(Ops::Base::CeilDiv(totalOuter, static_cast<int64_t>(coreNum)));
    useCoreNum_ = static_cast<int32_t>((totalOuter + dimsPerCore_ - 1) / dimsPerCore_);
    dimsPerCoreTail_ = static_cast<int32_t>(totalOuter - static_cast<int64_t>(useCoreNum_ - 1) * dimsPerCore_);
}

// ---------------------------------------------------------------------------
// Print helpers shared by PrintTilingData
// ---------------------------------------------------------------------------
namespace {

// Append "\n  <name><v0>,<v1>,..." for a fixed-length integer array.
template <typename T>
void AppendArray(std::ostringstream& oss, const char* name, const T* data, int32_t len)
{
    oss << "\n  " << name;
    for (int32_t i = 0; i < len; i++) {
        oss << data[i] << (i < len - 1 ? "," : "");
    }
}

// Append "\n  <name>  (idx,dim,stride) ..." for an axis list.
void AppendAxisList(std::ostringstream& oss, const char* name, const std::vector<AxisInf>& axes)
{
    oss << "\n  " << name << "  (idx,dim,stride): ";
    for (const auto& a : axes) {
        oss << "(" << a.idx << "," << a.dim << "," << a.stride << ") ";
    }
}

} // namespace

// ---------------------------------------------------------------------------
// DualCutAxisSeeker::PrintTilingData  (debug helper)
// ---------------------------------------------------------------------------
void DualCutAxisSeeker::PrintTilingData() const
{
    int32_t ubSizePerBuf = (ubSize_ / HIF4_BUFFER_NUM) / HIF4_BLOCK_BYTES * HIF4_BLOCK_BYTES;
    if (ubSizePerBuf <= 0) {
        ubSizePerBuf = HIF4_BLOCK_BYTES;
    }
    std::ostringstream oss;

    oss << "\n[HIF4DualCut] ===== TilingData =====";

    // --- original shape info ---
    AppendAxisList(oss, "[inShape]", inputAxis_);
    AppendArray(oss, "[outShape]   ", outShape_.data(), dimNums_);
    AppendArray(oss, "[perm]       ", perm_.data(), dimNums_);

    // --- multi-core split ---
    oss << "\n  [MultiCore] useCoreNum=" << useCoreNum_ << "  dimsPerCore=" << dimsPerCore_
        << "  dimsPerCoreTail=" << dimsPerCoreTail_;

    // --- UB budget ---
    oss << "\n  [UB]        ubSize=" << ubSizePerBuf << " bytes";

    // --- axis counts ---
    oss << "\n  [AxisNums]  outerAxisNums=" << outerAxisNums_ << "  innerAxisNums=" << innerAxisNums_
        << "  cutAxisNums=" << cutAxisNums_;

    // --- cut axis tail info ---
    oss << "\n  [CutAxis]   cutAxisIdx01=" << cutAxisIdx01_ << "  cutAxisTail01=" << cutAxisTail01_
        << "  cutAxisIdx02=" << cutAxisIdx02_ << "  cutAxisTail02=" << cutAxisTail02_;

    // --- cut axis id in gmShape ---
    oss << "\n  [CutId]     outCutId=" << outCutId_ << "  inCutId=" << inCutId_;

    // --- GM arrays ---
    AppendArray(oss, "[gmShape]    ", gmShape_, DUAL_CUT_GM_LEN);
    AppendArray(oss, "[gmInStride] ", gmInStride_, DUAL_CUT_GM_LEN);
    AppendArray(oss, "[gmOutStride]", gmOutStride_, DUAL_CUT_GM_LEN);

    // --- UB arrays ---
    AppendArray(oss, "[ubShape]    ", ubShape_, DUAL_CUT_UB_LEN);
    AppendArray(oss, "[ubInStride] ", ubInStride_, DUAL_CUT_UB_LEN);
    AppendArray(oss, "[ubOutStride]", ubOutStride_, DUAL_CUT_UB_LEN);
    AppendArray(oss, "[moveUbStride]", moveUbStride_, DUAL_CUT_UB_LEN);
    AppendArray(oss, "[ubPerm]     ", ubPerm_, DUAL_CUT_UB_LEN);
    AppendArray(oss, "[ubShapeId]  ", ubShapeId_, DUAL_CUT_UB_LEN);

    // --- gmAxis / ubAxis detail (for debug) ---
    AppendAxisList(oss, "[gmAxis]", gmAxis_);
    AppendAxisList(oss, "[ubAxis]", ubAxis_);

    oss << "\n[HIF4DualCut] ===== End =====";
    OP_LOGD(context_, "%s", oss.str().c_str());
}

// ===========================================================================
// TransposeTilingHIF4DualCut
// ===========================================================================

// ---------------------------------------------------------------------------
// TransposeTilingHIF4DualCut constructor
//   Prepare inShape_/outShape_ from shapeInfo_ and initialize the seeker.
//   For HIF4: use b4 view (HIF4 element count), shapes are NOT divided by 2.
// ---------------------------------------------------------------------------
TransposeTilingHIF4DualCut::TransposeTilingHIF4DualCut(gert::TilingContext* context, const ShapeInfo& shapeInfo,
                                                       int64_t coreNum, int64_t ubSize)
    : context_(context),
      shapeInfo_(shapeInfo),
      coreNum_(static_cast<int32_t>(coreNum)),
      ubSize_(static_cast<int32_t>(ubSize))
{
    int32_t dim = static_cast<int32_t>(shapeInfo_.dim);
    if (dim < 1) {
        return;
    }

    inShape_.resize(dim);
    outShape_.resize(dim);
    std::vector<int64_t> reducedPerm(dim);
    for (int32_t i = 0; i < dim; i++) {
        inShape_[i] = shapeInfo_.reducedInShape[i];
        reducedPerm[i] = shapeInfo_.reducedPerm[i];
        outShape_[i] = shapeInfo_.reducedInShape[reducedPerm[i]];
    }
    seeker_.Init(inShape_, outShape_, reducedPerm, dim, context_);
}

// ---------------------------------------------------------------------------
// BuildAxisInfo
//   Validate the reduced shape prepared by the constructor.
// ---------------------------------------------------------------------------
ge::graphStatus TransposeTilingHIF4DualCut::BuildAxisInfo()
{
    int32_t dim = static_cast<int32_t>(shapeInfo_.dim);
    OP_CHECK_IF(dim < 1, OP_LOGE(context_, "[HIF4DualCut] Invalid dim=%d", dim), return ge::GRAPH_FAILED);
    OP_LOGD(context_, "[HIF4DualCut] BuildAxisInfo: dim=%d, inShape and outShape prepared.", dim);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// RunDualCut
//   For HIF4: shapes are in HIF4 elements (b4 view, no division by 2)
//   UB budget: (ubSize / 10) aligned down to 32B, then * bufferNum
//   Each buffer holds HIF4 elements, 2 elements per byte
//   Alignment: 64 HIF4 elements = 32 bytes
// ---------------------------------------------------------------------------
ge::graphStatus TransposeTilingHIF4DualCut::RunDualCut()
{
    int32_t dim = static_cast<int32_t>(shapeInfo_.dim);

    static constexpr int32_t HIF4_TOTAL_BUFFER_NUM = 5; // 2 input + 2 output + 1 workspace
    int32_t perBufUbSize = (ubSize_ / (HIF4_TOTAL_BUFFER_NUM * HIF4_BUFFER_NUM)) / HIF4_BLOCK_BYTES * HIF4_BLOCK_BYTES;
    if (perBufUbSize <= 0) {
        perBufUbSize = HIF4_BLOCK_BYTES;
    }
    int32_t effectiveUbSize = perBufUbSize * HIF4_BUFFER_NUM;

    OP_LOGD(context_,
            "[HIF4DualCut] RunDualCut: dim=%d, ubSize=%d, perBufUbSize=%d (aligned), "
            "effectiveUbSize=%d bytes (%ld HIF4 elements), coreNum=%d",
            dim, ubSize_, perBufUbSize, effectiveUbSize,
            static_cast<int64_t>(effectiveUbSize) * HIF4_ELEMENTS_PER_BYTE / HIF4_BUFFER_NUM, coreNum_);

    // The seeker is initialized in the constructor; run the dual-cut search here.
    bool ok = seeker_.FindDualCutAxis(effectiveUbSize, HIF4_BUFFER_NUM);
    if (!ok) {
        OP_LOGE(context_, "[HIF4DualCut] FindDualCutAxis failed — "
                          "last dim too small or UB budget exhausted.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_, "[HIF4DualCut] FindDualCutAxis success, generating tiling data.");
    seeker_.GenTilingData();
    seeker_.ComputeBlockTiling(coreNum_);
    seeker_.PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// FillTilingData
// ---------------------------------------------------------------------------
ge::graphStatus TransposeTilingHIF4DualCut::FillTilingData()
{
    OP_LOGD(context_, "[HIF4DualCut] FillTilingData start.");

    tilingData_.set_gmShape(seeker_.gmShape_);
    tilingData_.set_gmInStride(seeker_.gmInStride_);
    tilingData_.set_gmOutStride(seeker_.gmOutStride_);
    tilingData_.set_ubShape(seeker_.ubShape_);
    tilingData_.set_ubInStride(seeker_.ubInStride_);
    tilingData_.set_ubOutStride(seeker_.ubOutStride_);
    tilingData_.set_moveUbStride(seeker_.moveUbStride_);
    tilingData_.set_ubPerm(seeker_.ubPerm_);
    tilingData_.set_ubShapeId(seeker_.ubShapeId_);
    tilingData_.set_cutAxisIdx01(seeker_.cutAxisIdx01_);
    tilingData_.set_cutAxisIdx02(seeker_.cutAxisIdx02_);
    tilingData_.set_cutAxisTail01(seeker_.cutAxisTail01_);
    tilingData_.set_cutAxisTail02(seeker_.cutAxisTail02_);
    tilingData_.set_outCutId(seeker_.outCutId_);
    tilingData_.set_inCutId(seeker_.inCutId_);
    tilingData_.set_outerAxisNums(seeker_.outerAxisNums_);
    tilingData_.set_innerAxisNums(seeker_.innerAxisNums_);
    tilingData_.set_dimsPerCore(seeker_.dimsPerCore_);
    tilingData_.set_useCoreNum(seeker_.useCoreNum_);
    tilingData_.set_dimsPerCoreTail(seeker_.dimsPerCoreTail_);
    int32_t ubSizePerBuf = (seeker_.ubSize_ / HIF4_BUFFER_NUM) / HIF4_BLOCK_BYTES * HIF4_BLOCK_BYTES;
    if (ubSizePerBuf <= 0) {
        ubSizePerBuf = HIF4_BLOCK_BYTES;
    }
    tilingData_.set_ubSize(ubSizePerBuf);

    // Fill perm info for kernel
    int32_t permSize = static_cast<int32_t>(shapeInfo_.dim);
    tilingData_.set_permSize(permSize);
    int64_t permArr[MAX_AXIS_NUM_FOR_TRANSPOSE] = {0};
    for (int32_t i = 0; i < MAX_AXIS_NUM_FOR_TRANSPOSE; i++) {
        if (i < permSize) {
            permArr[i] = shapeInfo_.reducedPerm[i];
        } else {
            permArr[i] = 0;
        }
    }
    tilingData_.set_perm(permArr);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;

    context_->SetBlockDim(static_cast<uint32_t>(seeker_.useCoreNum_));
    context_->SetTilingKey(static_cast<uint64_t>(SplitMode::B4_DOUBLE_SPLIT));

    OP_LOGD(context_,
            "[HIF4DualCut] FillTilingData done: "
            "TilingKey=%lu, BlockDim=%d",
            static_cast<uint64_t>(SplitMode::B4_DOUBLE_SPLIT), seeker_.useCoreNum_);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// DoTiling: BuildAxisInfo -> RunDualCut -> FillTilingData
//           Dual-cut only (no sole-cut fallback)
// ---------------------------------------------------------------------------
ge::graphStatus TransposeTilingHIF4DualCut::DoTiling()
{
    OP_LOGD(context_, "[HIF4DualCut] DoTiling start.");

    OP_CHECK_IF(BuildAxisInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "[HIF4DualCut] BuildAxisInfo failed."),
                return ge::GRAPH_FAILED);

    // Dual-cut only (no sole-cut fallback)
    OP_CHECK_IF(RunDualCut() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "[HIF4DualCut] RunDualCut failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "[HIF4DualCut] FillTilingData failed."),
                return ge::GRAPH_FAILED);

    OP_LOGD(context_, "[HIF4DualCut] DoTiling success.");
    return ge::GRAPH_SUCCESS;
}

} // namespace TransHIF4DualCut
} // namespace optiling
