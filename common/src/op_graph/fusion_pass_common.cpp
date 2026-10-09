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
 * \file fusion_pass_common.cpp
 * \brief 把桩头编入 libopgraph_math.so，并在满足条件时登记 fe 融合 pass 名字。
 */

// 仅当编译 soc 为 ascend950 时，当前 cpp 中的打桩注册内容才编入
#if defined(OP_GRAPH_MATH_ASCEND950)

#include "version/ge-compiler_version.h"
#include "fe_fusion_pass_registry_stub.h"

#define GE_COMPILER_VERSION_900 90000000
// 编译期保护：构建环境 ge_compiler >= 9.0.0 才编入（运行时检查在 STUB_REG_PASS 宏内）
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

// 登记 canndev 旧融合 pass 名字，名字/类型/attr 照抄 canndev 的 REG_PASS/REGISTER_PASS
// attr：FORBIDDEN_CLOSE=0x01、SINGLE_SCENE_OPEN=0x04；3 参 REGISTER_PASS 默认 0
// Castlike/Histogram 不登记：Stage 是 kAfterInferShape，不在 kCompatibleInherited 表，会报 "pattern fusion is nullptr"
// DropOutV3Split 已登记但当前不生效：不在 OPP 白名单 3510(ascend950) 内，待 OPP 补名单
STUB_REG_PASS("ReduceMeanWithCastFusionPass", fe::BUILT_IN_PREPARE_GRAPH_PASS, 0x05);
STUB_REG_PASS("PermuteFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x05);
STUB_REG_PASS("RandomUniformFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01);
STUB_REG_PASS("TruncatedNormalFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01);
STUB_REG_PASS("RandomStandardNormalFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01);
STUB_REG_PASS("RandomUniformIntFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01);
STUB_REG_PASS("DropOutV3FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x05);
STUB_REG_PASS("DropOutV3SplitFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x05);
STUB_REG_PASS("BernoulliFusionPass", fe::BUILT_IN_GRAPH_PASS, 0);

#endif // GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
#endif // OP_GRAPH_MATH_ASCEND950
