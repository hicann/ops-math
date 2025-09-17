/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_util.cpp
 * \brief
 */

#include "tiling_base/tiling_util.h"
#include "platform/platform_ascendc.h"

namespace Ops {
namespace Math {
namespace OpTiling {
static const gert::Shape g_vec_1_shape = {1};

static bool IsRegbaseSocVersion(platform_ascendc::SocVersion version)
{
    const static std::set<platform_ascendc::SocVersion> regbaseSocVersions = {
        platform_ascendc::SocVersion::ASCEND910_95};

    return regbaseSocVersions.find(version) != regbaseSocVersions.end();
}

bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

bool IsRegbaseSocVersion(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}
} // namespace OpTiling
} // namespace Math
} // namespace Ops