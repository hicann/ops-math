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
 * \file fe_fusion_pass_registry_stub.h
 * \brief GE fe 融合 pass 注册接口的最小桩声明：只声明不实现，运行期由动态链接器绑定 libregister.so
 */

#ifndef FE_FUSION_PASS_REGISTRY_STUB_H_
#define FE_FUSION_PASS_REGISTRY_STUB_H_

#include <cstdint>
#include <map>
#include <string>

namespace fe {

// pass 阶段类型，枚举值必须与 GE 的 fe::GraphFusionPassType 一致
// （来源：cann/ge graph_fusion_pass_base.h）
enum GraphFusionPassType {
    BUILT_IN_GRAPH_PASS = 0,
    BUILT_IN_VECTOR_CORE_GRAPH_PASS,
    CUSTOM_AI_CORE_GRAPH_PASS,
    CUSTOM_VECTOR_CORE_GRAPH_PASS,
    SECOND_ROUND_BUILT_IN_GRAPH_PASS,
    BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
    BUILT_IN_PREPARE_GRAPH_PASS,
    BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS,
    BUILT_IN_TF_TAG_NO_CONST_FODING_GRAPH_PASS,
    BUILT_IN_TF_MERGE_SUB_GRAPH_PASS,
    BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS,
    BUILT_IN_EN_ISA_ARCH_EXC_V300_AND_V220_GRAPH_PASS,
    BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS,
    BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS,
    BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS,
    BUILT_IN_AFTER_MULTI_DIMS_PASS,
    BUILT_IN_AFTER_OPTIMIZE_STAGE1,
    BUILT_IN_AFTER_OP_JUDGE,
    BUILT_IN_AFTER_BUFFER_OPTIMIZE,
    GRAPH_FUSION_PASS_TYPE_RESERVED
};

using PassAttr = uint64_t;

class GraphPass;

struct PassDesc {
    PassAttr attr;
    GraphPass* (*create_fn)();
};

class FusionPassRegistry {
public:
    static FusionPassRegistry& GetInstance();
    std::map<std::string, PassDesc> GetPassDesc(const GraphFusionPassType& pass_type);
};

// 构造函数有副作用：构造即完成登记
class FusionPassRegistrar {
public:
    FusionPassRegistrar(const GraphFusionPassType& pass_type, const std::string& pass_name, GraphPass* (*create_fn)(),
                        PassAttr attr);
    ~FusionPassRegistrar() {}
};

} // namespace fe

// 弱符号：查询 ge_compiler 版本；环境里没有该函数时值为 nullptr，不报错
extern "C" {
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

namespace fe {

constexpr int32_t kStubGeCompilerVersion900 = 90000000;

inline bool StubIsGeCompilerVersionSatisfied()
{
    int32_t version = 0;
    if (aclsysGetVersionNum != nullptr) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    return version >= kStubGeCompilerVersion900;
}

} // namespace fe

// 版本满足才构造 registrar（不满足则 nullptr，不登记）；工厂写死 nullptr，创建由 REG_FUSION_PASS 负责
#define STUB_REG_PASS(pass_name, pass_type, attr) STUB_REG_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass_type, attr)

#define STUB_REG_PASS_UNIQ_HELPER(ctr, pass_name, pass_type, attr) STUB_REG_PASS_UNIQ(ctr, pass_name, pass_type, attr)

#define STUB_REG_PASS_UNIQ(ctr, pass_name, pass_type, attr)                                                           \
    static ::fe::FusionPassRegistrar* stub_reg_##ctr                                                                  \
        __attribute__((unused)) = ::fe::StubIsGeCompilerVersionSatisfied() ?                                          \
                                      new ::fe::FusionPassRegistrar(                                                  \
                                          pass_type, pass_name, []() -> ::fe::GraphPass* { return nullptr; }, attr) : \
                                      nullptr

#endif // FE_FUSION_PASS_REGISTRY_STUB_H_
