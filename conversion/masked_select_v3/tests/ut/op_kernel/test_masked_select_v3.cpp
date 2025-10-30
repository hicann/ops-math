/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_host/masked_select_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

extern "C" __global__ __aicore__ void masked_select_v3(
    GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeOut, GM_ADDR workspace, GM_ADDR tiling);

class MaskedSelectV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MaskedSelectV3Test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MaskedSelectV3Test TearDown\n" << endl;
    }
};

TEST_F(MaskedSelectV3Test, masked_select_v3_0_int64)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};
    
    gert::TilingContextPara tilingContextPara("MaskedSelectV3",
                                              {{{{8,}, {8,}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{8,}, {8,}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {{{{8,}, {8,}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputXByteSize = 8 * sizeof(float);
    size_t inputMaskByteSize = 8 * sizeof(bool);
    size_t outputYByteSize = 8 * sizeof(float);
    size_t shapeOutByteSize = sizeof(uint64_t) * 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* shapeOut = (uint8_t*)AscendC::GmAlloc(shapeOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    uint32_t blockDim = 1;
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(masked_select_v3, blockDim, x, mask, y, shapeOut, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);
    AscendC::GmFree(shapeOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
