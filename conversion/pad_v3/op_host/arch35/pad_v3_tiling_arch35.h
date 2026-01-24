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
 * \file pad_v3_tiling_arch35.h
 * \brief ac pad tiling h
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "conversion/slice/op_host/arch35/slice_tiling_arch35.h"
#include "conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h"
#include "conversion/pad_v3/op_kernel/arch35/pad_v3_struct.h"
#include <string>

namespace optiling {
struct paddingInfo {
    gert::Shape padFront;
    gert::Shape padBack;
};

struct PadV3CompileInfo {
    int64_t max_output_size;
    int64_t core_num;
    // how much input data can the UB store, instead of UB's size in Bytes
    int64_t ub_size;
    // how much input data can 4Bytes store in 5HD format, instead of dtype's size.
    // float16: 2; float32/int32: 1.tring
    int64_t size;
    int64_t dtype_rate;
    bool paddings_contiguous;
    std::string mode;
    // chip's UB size in Bytes
    int64_t total_ub_size;
    int64_t output_dim4_max_cnt;
    // x data_size in Bytes
    int64_t x_bytes_size;
    // is support data_move_pad
    bool is_support_data_move_pad;
    std::string soc_version;
};

class PadACTiling {
public:
    explicit PadACTiling(gert::TilingContext* context) : context_(context)
    {}
    ge::graphStatus DoTiling();

private:
    template <typename T>
    std::string ToString(const T* value, size_t size);
    ge::graphStatus Init();              // 获取coreNum,ubSize,vecSize,cacheLineSize,blockSize
    ge::graphStatus GetShapeAttrsInfo(); // 获取校验入参和shape信息

    ge::graphStatus DimensionCollapse();              // 合轴
    ge::graphStatus EdgeDimensionCollapse();          // edge合轴
    ge::graphStatus ReflectDimensionCollapse();       // reflect合轴
    void CheckReplication();                          // 检查replication
    void EmptyTensorCollapse();                       // 空tensor时合为一根轴
    void EmptyTensorCollapseEdge();                   // edge空tensor时操作
    void EmptyTensorCollapseReflect();                // reflect空tensor时操作
    ge::graphStatus ComputeAfterPaddingsAndStrides(); // 计算合轴后的paddings和strides

    void DoTilingWithConstant();
    void DoTilingWithEdge();
    void DoTilingWithSliceOp();
    void DoTilingWithReflect();
    void DoTilingWithSIMTEdge();    // edge模板DoTilingWithSIMT
    void DoTilingWithSIMTReflect(); // reflect模板DoTilingWithSIMT
    void DoTilingWithSIMT();
    void FillsAndPrintTilingData();

    ge::graphStatus GetShapesAndDtypes();
    ge::graphStatus GetPaddings();
    template <typename T>
    void GetPaddingsToShape(const gert::Tensor* paddingsTensor);
    void CalculateTilingKeyReflect(); // reflect模板CalculateTilingKey
    void CalculateTilingKeyEdge();    // edge模板CalculateTilingKey
    void CalculateTilingKey();
    uint64_t GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize);
    void DoFindSplitAxis(bool isBigLastDim);
    void DoFindSplitAxisReflect(bool isBigLastDim);
    void CalculateGatherOrScatter();
    void CalculateGatherOrScatterEdge();    // edge模板CalculateGatherOrScatter
    void CalculateGatherOrScatterReflect(); // reflect模板CalculateGatherOrScatter
    void CaculateTilingParams();

public:
    bool isPadV3_{false};
    bool isMirrorPad_{false};

private:
    gert::TilingContext* context_ = nullptr;
    uint32_t coreNum_{0};
    uint64_t ubSize_{0};
    uint64_t blockSize_{32};
    uint64_t vectorSize_{256};
    uint32_t cacheLineSize_{256};
    PadACTilingData* tilingData_;
    PadACTilingData* tilingDataSIMT_;
    uint64_t tilingKey_;
    enum class ModeNum : uint8_t
    {
        CONSTANT = 0,
        EDGE = 1,
        REFLECT = 2,
        SYMMETRIC = 3
    };
    ModeNum padMode_{ModeNum::CONSTANT}; // 0为constant，1为edge,2为reflect,3为symmetric

    uint8_t dimNum_{1};
    uint8_t ubAxis_{0};
    uint32_t ubFactor_{0};
    uint32_t ubPerCount_{0};
    uint32_t ubTotalCount_{0};
    uint32_t outTileSize_{0};
    uint32_t additionTileSize_{0};

    bool isPadAllPositive_{true};
    bool isPadAllNegative_{true};
    bool paddingContiguous_{true};
    bool isEmptyTensor_{false};
    uint16_t inputRank_{0};
    paddingInfo paddings_;
    ge::DataType paramsDtype_;
    uint32_t dtypeBytes_{0};
    uint64_t bufferSize_{0};
    uint64_t inShapeSize_{1};
    uint64_t outShapeSize_{1};
    int64_t rightPad_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_TILING_H_
