/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/fusion/pass/pattern_fusion_pass.h"

#ifndef OPS_MATH_CONVERSION_TRIU_FUSION_PASS_TRILU_FUSION_PASS_H_
#define OPS_MATH_CONVERSION_TRIU_FUSION_PASS_TRILU_FUSION_PASS_H_

namespace ops {
using namespace ge;
using namespace ge::fusion;

class __attribute__((visibility("default"))) TriluFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& matchResult) override;

    GraphUniqPtr Replacement(const std::unique_ptr<MatchResult>& matchResult) override;
};

static void GetInputsInfo(const std::vector<SubgraphInput> &subgraphInputs, std::vector<Shape> &inputShapes,
    std::vector<DataType> &inputDtypes, std::vector<Format> &inputFormats);
static Status InferShape(const GraphUniqPtr &replaceGraph, const std::vector<SubgraphInput> &subgraphInputs);

} // namespace ops
#endif // OPS_MATH_CONVERSION_TRIU_FUSION_PASS_TRILU_FUSION_PASS_H_