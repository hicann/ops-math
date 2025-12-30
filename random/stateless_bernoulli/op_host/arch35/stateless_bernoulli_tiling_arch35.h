/**
 * Copyright (c) Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_bernoulli_tiling_arch35.h
 * \brief
 */
#ifndef STATELESS_BERNOULLI_TILING_ARCH35_H
#define STATELESS_BERNOULLI_TILING_ARCH35_H
#pragma once

#include "tiling_base/tiling_base.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;

BEGIN_TILING_DATA_DEF(StatelessBernoulliTilingData)
TILING_DATA_FIELD_DEF(uint64_t, blockNum);                        // 使用核数
TILING_DATA_FIELD_DEF(uint64_t, blockTilingSize);                 // 非尾核处理元素个数
TILING_DATA_FIELD_DEF(uint64_t, tailBlockTilingSize);             // 尾核处理元素个数
TILING_DATA_FIELD_DEF(uint64_t, blockLoopCount);                  // 非尾核核内loop次数
TILING_DATA_FIELD_DEF(uint64_t, tailBlockLoopCount);              // 尾核核内loop次数
TILING_DATA_FIELD_DEF(uint64_t, ubTilingSize);                    // 单次loop处理元素个数
TILING_DATA_FIELD_DEF(uint64_t, probTensorSize);                  // probTensor元素个数
TILING_DATA_FIELD_DEF(uint64_t, outputSize);                      // 输出元素个数
TILING_DATA_FIELD_DEF(uint64_t, isProbScalar);                    // prob是否为scalar
TILING_DATA_FIELD_DEF_ARR(uint32_t, ALG_KEY_SIZE, key);           // 输入key数组
TILING_DATA_FIELD_DEF_ARR(uint32_t, ALG_COUNTER_SIZE, counter);   // 输入counter数组
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StatelessBernoulli, StatelessBernoulliTilingData)

struct StatelessBernoulliCompileInfoArch35 {
    uint64_t aivNum;
    uint64_t ubSize;
};

class StatelessBernoulliTiling : public Ops::Math::OpTiling::TilingBaseClass {
public:
    explicit StatelessBernoulliTiling(gert::TilingContext *context) : TilingBaseClass(context) {
        Reset();
    }
    ~StatelessBernoulliTiling() override = default;
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

    enum class Algorithm : int {
        RNG_ALG_PHILOX = 1,
        RNG_ALG_THREEFRY = 2,
        RNG_ALG_AUTO_SELECT = 3
    };

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 顺序执行1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 8、dump日志
    void DumpTilingInfo() override;
    // 9、reset重置
    void Reset();

private:
    static constexpr uint64_t BUFFER_NUM = 2;
    static constexpr uint64_t EXIST_NODE_NUM = 4;
    static constexpr uint64_t CORE_ALIGN_SIZE = 512;
    static constexpr uint64_t BLOCK_SIZE_BYTES = 32;
    static constexpr uint64_t MIN_TILING_SIZE = 256;
    static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;

    template <typename T>
    ge::graphStatus GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape);
    ge::graphStatus GetIntToShape(const int64_t constIdx, gert::Shape &constShape);

    inline uint64_t GetBytePerData(const ge::DataType& dtype);
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetOutputInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetInputKeyCounter();
    int64_t GetCounterSize(Algorithm alg) const;
    void GetKeyFromMem(const int64_t key);
    void GetCounterFromMem(const std::vector<int64_t> &counter);
    void BlockTiling();
    ge::graphStatus UbTiling();
    void SetTilingData();
    ge::graphStatus GetIntValueFromProb(const gert::Shape &originShape, gert::Shape &constShape, size_t shapeSize);

private:
    gert::Shape inputShape_;
    gert::Shape inputSeed_;
    gert::Shape inputOffset_;
    ge::DataType probDtype_;
    ge::DataType inputDtype_;
    ge::DataType outputDtype_;

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t inputSize_ = 1;
    uint64_t inputDtypeSize_ = 0;
    uint64_t blockNum_ = 0;
    uint64_t blockTilingSize_ = 0;
    uint64_t tailBlockTilingSize_ = 0;
    uint64_t blockLoopCount_ = 0;
    uint64_t tailBlockLoopCount_ = 0;
    uint64_t ubTilingSize_ = 0;
    uint64_t probTensorSize_ = 0;
    uint64_t outputSize_ = 1;
    uint64_t isProbScalar_ = 1;

    Algorithm alg_ = Algorithm::RNG_ALG_PHILOX;
    uint32_t key_[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0, 0, 0, 0};

    const char *opName_ = "";
    StatelessBernoulliTilingData m_tilingData_;
};
} // namespace optiling
#endif // STATELESS_BERNOULLI_TILING_H