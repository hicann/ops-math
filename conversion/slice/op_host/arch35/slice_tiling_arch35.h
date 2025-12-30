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
 * \file slice_tiling_base.h
 * \brief
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_SLICE_TILING_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_SLICE_TILING_H_

#include "conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h"

namespace optiling {

constexpr int64_t SLICE_KEY_MOVE_UNALIGN_GATHER = 301;
constexpr int64_t SLICE_KEY_TWO_DIM_SMALL_SHAPE = 400;
constexpr int64_t MIN_OUTPUT_SIZE = int64_t(512 * 1024);
constexpr size_t INPUT_X_INDEX = 0;

struct SliceCompileParam {
    int64_t block_dim{1};
    int64_t ub_size{0};
    uint32_t cacheLineSize{0};
    bool isAscendc{false};
};

struct SliceParasRuntime2 {
    gert::Shape input;
    gert::Shape output_shape;
    gert::Shape begin_list;
    gert::Shape end_list;
    gert::Shape stride_list;
    int64_t tiling_mode = 0;
    int64_t core_num = 0;
    bool is_begin_const = true;
};

REGISTER_TILING_DATA_CLASS(StridedSliceTilingDataOp, StridedSliceTilingData)

BEGIN_TILING_DATA_DEF(SliceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(StridedSliceTilingData, stridedSliceTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice, SliceTilingData)

BEGIN_TILING_DATA_DEF(SliceMoveAlignParams)
TILING_DATA_FIELD_DEF(uint16_t, blockCount);
TILING_DATA_FIELD_DEF(uint32_t, blockLen);
TILING_DATA_FIELD_DEF(uint32_t, srcStride);
TILING_DATA_FIELD_DEF(uint32_t, dstStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SliceMoveAlignParamsOp, SliceMoveAlignParams) // 注册中间结构体

BEGIN_TILING_DATA_DEF(SliceBaseTilingData)
TILING_DATA_FIELD_DEF(int8_t, isBeginConst);
TILING_DATA_FIELD_DEF(int64_t, blkFactor);
TILING_DATA_FIELD_DEF(int64_t, blkTailFactor);
TILING_DATA_FIELD_DEF(int64_t, ubInLoopSteps);
TILING_DATA_FIELD_DEF(int32_t, ubSize);
TILING_DATA_FIELD_DEF(int32_t, ubFactor);
TILING_DATA_FIELD_DEF(int32_t, ubTailFactor);
TILING_DATA_FIELD_DEF(int32_t, ubTailTailFactor);
TILING_DATA_FIELD_DEF(int16_t, realCoreNum);
TILING_DATA_FIELD_DEF(int16_t, inputDims);
TILING_DATA_FIELD_DEF(int16_t, blkIndex);
TILING_DATA_FIELD_DEF(int16_t, ubIndex);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_AXIS_NUM_FOR_STRIDESLICE, outputShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_AXIS_NUM_FOR_STRIDESLICE, begin);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_AXIS_NUM_FOR_STRIDESLICE, inputSteps);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_AXIS_NUM_FOR_STRIDESLICE, rowsOffsetSteps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SliceBaseTilingDataOp, SliceBaseTilingData)

BEGIN_TILING_DATA_DEF(SliceMoveAlignTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SliceBaseTilingData, sliceBaseTilingData);
TILING_DATA_FIELD_DEF_STRUCT(StridedSliceMoveAlignParams, moveAlignParams);
TILING_DATA_FIELD_DEF(int64_t, ubOutLoopSteps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_100, SliceMoveAlignTilingData)

BEGIN_TILING_DATA_DEF(SliceMoveAlignLastDimTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SliceBaseTilingData, sliceBaseTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_101, SliceMoveAlignLastDimTilingData)

BEGIN_TILING_DATA_DEF(SliceNDDMATilingData)
TILING_DATA_FIELD_DEF_STRUCT(SliceBaseTilingData, sliceBaseTilingData);

TILING_DATA_FIELD_DEF(int64_t, ubOutLoopSteps);
TILING_DATA_FIELD_DEF(int64_t, nddmaTotalNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_NDDMA_UB_SPLIT_AXIS_NUM, nddmaLoopSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_NDDMA_UB_SPLIT_AXIS_NUM, nddmaLoopSrcStride);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_NDDMA_UB_SPLIT_AXIS_NUM, nddmaLoopDstStride);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_102, SliceNDDMATilingData)

BEGIN_TILING_DATA_DEF(SliceNDDMALastDimTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SliceBaseTilingData, sliceBaseTilingData);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_NDDMA_UB_SPLIT_AXIS_NUM, nddmaLoopSrcStride);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_NDDMA_UB_SPLIT_AXIS_NUM, nddmaLoopDstStride);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_103, SliceNDDMALastDimTilingData)

BEGIN_TILING_DATA_DEF(SliceMoveAlignLast2DimTilingData)
TILING_DATA_FIELD_DEF(int64_t, blkFactor);
TILING_DATA_FIELD_DEF(int64_t, blkTailFactor);
TILING_DATA_FIELD_DEF(int64_t, ubInLoopSteps);
TILING_DATA_FIELD_DEF(int64_t, ubOutLoopSteps);
TILING_DATA_FIELD_DEF(int32_t, ubSize);
TILING_DATA_FIELD_DEF(int32_t, ubFactor);
TILING_DATA_FIELD_DEF(int32_t, ubTailFactor);
TILING_DATA_FIELD_DEF(int32_t, ubTailTailFactor);
TILING_DATA_FIELD_DEF(int16_t, realCoreNum);
TILING_DATA_FIELD_DEF(int8_t, isBeginConst);
TILING_DATA_FIELD_DEF_STRUCT(SliceMoveAlignParams, moveAlignParams);
TILING_DATA_FIELD_DEF_ARR(int64_t, NUMBER_TWO, outputShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, NUMBER_TWO, begin);
TILING_DATA_FIELD_DEF_ARR(int64_t, NUMBER_TWO, inputSteps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_150, SliceMoveAlignLast2DimTilingData)

BEGIN_TILING_DATA_DEF(SliceMoveAlignGatherTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SliceBaseTilingData, sliceBaseTilingData);

TILING_DATA_FIELD_DEF(int64_t, ubOutLoopSteps);
TILING_DATA_FIELD_DEF(int32_t, ubSizeInput);
TILING_DATA_FIELD_DEF(uint32_t, lastOneInputDim);
TILING_DATA_FIELD_DEF(uint32_t, outBlockLen);
TILING_DATA_FIELD_DEF_STRUCT(StridedSliceMoveAlignParams, moveAlignParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_300, SliceMoveAlignGatherTilingData)
REGISTER_TILING_DATA_CLASS(Slice_301, SliceMoveAlignGatherTilingData)

BEGIN_TILING_DATA_DEF(SliceTwoDimSmallSapeTilingData);
TILING_DATA_FIELD_DEF(int8_t, isBeginConst);
TILING_DATA_FIELD_DEF(int16_t, realCoreNum);
TILING_DATA_FIELD_DEF(int16_t, mainCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint32_t, blockLen);
TILING_DATA_FIELD_DEF(uint32_t, blkFactor);
TILING_DATA_FIELD_DEF(uint64_t, lastOneInputDim);
TILING_DATA_FIELD_DEF(uint64_t, lastOneOutputDim);
TILING_DATA_FIELD_DEF(uint64_t, lastOneDimOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Slice_400, SliceTwoDimSmallSapeTilingData)

class SliceTiling : public StrideSliceTiling {
public:
    explicit SliceTiling(gert::TilingContext* context) : StrideSliceTiling(context) {};

protected:
    void CalMaxSplitDim() override;
    void SetTilingMode() override;
    void FillTilingData() override;
    void PrintTilingData() override;
    void SetBlockDimAndTilingKey() override;
    void FillSliceBaseTilingData(SliceBaseTilingData& tilingData);
    void FillSliceTilingData150();
    void FillSliceTilingData101();
    void FillSliceTilingData102();
    void FillSliceTilingData103();
    void FillSliceTilingData100();
    void FillSliceTilingData300();
    void FillSliceTilingData400();
    void FillSliceTilingDataOther();
    void PrintSliceBaseTilingData(SliceBaseTilingData& tilingData);
    void PrintSliceTilingData150();
    void PrintSliceTilingData101();
    void PrintSliceTilingData102();
    void PrintSliceTilingData103();
    void PrintSliceTilingData100();
    void PrintSliceTilingData300();
    void PrintSliceTilingData400();
    void PrintSliceTilingDataOther();
    void SetRowsStepsParamsFor150(SliceMoveAlignLast2DimTilingData& tilingData);
    void SetShortMoveAlignParams(SliceMoveAlignParams& params, const MoveAlignV2Info& actInfo);
    void CalSliceRowsStepsParams();
    void SetSliceGatherTilingMode();
    void SetTwoDimSmallShapeTilingMode();
    void SliceGatherUbSplitLastTwoDim();
    void SliceGatherUbSplitLastThreeDim();
    void SliceGatherUbSplitLastFourDim();

private:
    SliceTilingData sliceTilingData_;
    SliceMoveAlignLast2DimTilingData sliceMoveAlignLast2DimTilingData_;
    SliceMoveAlignLastDimTilingData sliceMoveAlignLastDimTilingData_;
    SliceMoveAlignTilingData sliceMoveAlignTilingData_;
    SliceNDDMATilingData sliceNDDMATilingData_;
    SliceNDDMALastDimTilingData sliceNDDMALastDimTilingData_;
    SliceMoveAlignGatherTilingData sliceMoveAlignGatherTilingData_;
    SliceTwoDimSmallSapeTilingData sliceTwoDimSmallSapeTilingData_;

    int64_t inLoopSteps_ = 1;
    int64_t outLoopSteps_ = 1;
    bool isUnalignCopy_ = false;
};
} // namespace optiling
#endif // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_SLICE_TILING_H_