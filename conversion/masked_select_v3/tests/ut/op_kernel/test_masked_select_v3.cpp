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

#include "masked_select_v3_tiling.h"

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
    size_t shapeSize = (131072 - 290) * 6;
    size_t inputXByteSize = shapeSize * sizeof(uint64_t);
    size_t inputMaskByteSize = shapeSize * sizeof(uint8_t);
    size_t outputYByteSize = shapeSize * sizeof(uint64_t);
    size_t shapeOutByteSize = sizeof(int64_t) * 2;
    size_t tilingDataSize = sizeof(MaskedSelectV3TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* shapeOut = (uint8_t*)AscendC::GmAlloc(shapeOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(inputXByteSize * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    MaskedSelectV3TilingData* tilingData = reinterpret_cast<MaskedSelectV3TilingData*>(tiling);

    tilingData->formerNum = 36;
    tilingData->formerLength = 16348;
    tilingData->formertileNum = 6;
    tilingData->formertileLength = 3000;
    tilingData->formerlasttileLength = 1348;

    tilingData->tailNum = 12;
    tilingData->tailLength = 16347;
    tilingData->tailtileNum = 6;
    tilingData->tailtileLength = 3000;
    tilingData->taillasttileLength = 1347;

    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(masked_select_v3, blockDim, x, mask, y, shapeOut, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);
    AscendC::GmFree(shapeOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(MaskedSelectV3Test, masked_select_v3_0_int32)
{
    size_t shapeSize = (131072 - 290) * 6;
    size_t inputXByteSize = shapeSize * sizeof(uint32_t);
    size_t inputMaskByteSize = shapeSize * sizeof(uint8_t);
    size_t outputYByteSize = shapeSize * sizeof(uint32_t);
    size_t shapeOutByteSize = sizeof(int64_t) * 2;
    size_t tilingDataSize = sizeof(MaskedSelectV3TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* shapeOut = (uint8_t*)AscendC::GmAlloc(shapeOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(inputXByteSize * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    MaskedSelectV3TilingData* tilingData = reinterpret_cast<MaskedSelectV3TilingData*>(tiling);

    tilingData->formerNum = 36;
    tilingData->formerLength = 16348;
    tilingData->formertileNum = 6;
    tilingData->formertileLength = 3000;
    tilingData->formerlasttileLength = 1348;

    tilingData->tailNum = 12;
    tilingData->tailLength = 16347;
    tilingData->tailtileNum = 6;
    tilingData->tailtileLength = 3000;
    tilingData->taillasttileLength = 1347;

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(masked_select_v3, blockDim, x, mask, y, shapeOut, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);
    AscendC::GmFree(shapeOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(MaskedSelectV3Test, masked_select_v3_0_int16)
{
    size_t shapeSize = (131072 - 290) * 6;
    size_t inputXByteSize = shapeSize * sizeof(uint16_t);
    size_t inputMaskByteSize = shapeSize * sizeof(uint8_t);
    size_t outputYByteSize = shapeSize * sizeof(uint16_t);
    size_t shapeOutByteSize = sizeof(int64_t) * 2;
    size_t tilingDataSize = sizeof(MaskedSelectV3TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* shapeOut = (uint8_t*)AscendC::GmAlloc(shapeOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(inputXByteSize * 3);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    MaskedSelectV3TilingData* tilingData = reinterpret_cast<MaskedSelectV3TilingData*>(tiling);

    tilingData->formerNum = 36;
    tilingData->formerLength = 16348;
    tilingData->formertileNum = 6;
    tilingData->formertileLength = 3000;
    tilingData->formerlasttileLength = 1348;

    tilingData->tailNum = 12;
    tilingData->tailLength = 16347;
    tilingData->tailtileNum = 6;
    tilingData->tailtileLength = 3000;
    tilingData->taillasttileLength = 1347;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(masked_select_v3, blockDim, x, mask, y, shapeOut, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);
    AscendC::GmFree(shapeOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(MaskedSelectV3Test, masked_select_v3_0_int8)
{
    size_t shapeSize = (131072 - 290) * 6;
    size_t inputXByteSize = shapeSize * sizeof(uint8_t);
    size_t inputMaskByteSize = shapeSize * sizeof(uint8_t);
    size_t outputYByteSize = shapeSize * sizeof(uint8_t);
    size_t shapeOutByteSize = sizeof(int64_t) * 2;
    size_t tilingDataSize = sizeof(MaskedSelectV3TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* shapeOut = (uint8_t*)AscendC::GmAlloc(shapeOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(inputXByteSize * 6);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    MaskedSelectV3TilingData* tilingData = reinterpret_cast<MaskedSelectV3TilingData*>(tiling);

    tilingData->formerNum = 36;
    tilingData->formerLength = 16348;
    tilingData->formertileNum = 6;
    tilingData->formertileLength = 3000;
    tilingData->formerlasttileLength = 1348;

    tilingData->tailNum = 12;
    tilingData->tailLength = 16347;
    tilingData->tailtileNum = 6;
    tilingData->tailtileLength = 3000;
    tilingData->taillasttileLength = 1347;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(masked_select_v3, blockDim, x, mask, y, shapeOut, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);
    AscendC::GmFree(shapeOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
