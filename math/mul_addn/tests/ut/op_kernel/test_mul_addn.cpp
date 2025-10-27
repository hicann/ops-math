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
#include "../../../op_host/mul_addn_tiling.h"
#include "data_utils.h"
#include "tensor_list_operate.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void mul_addn(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class mul_addn_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mul_addn SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mul_addn TearDown\n" << endl;
    }
};

TEST_F(mul_addn_test, test_case_mean_fp32_01)
{
    std::cout << "get_current_dir_name:" << get_current_dir_name() << std::endl;
    system(
        "cp -rf "
        "../../../../math/mul_addn/tests/ut/op_kernel/mul_addn_data/ "
        "./");
    system("chmod -R 755 ./mul_addn_data/");
    // batch_size, num_classes, dtype, reduction, flag
    system("cd ./mul_addn_data/ && python3 gen_data.py 30 1024 float32 mean True");

    std::vector<std::vector<uint64_t>> x1Shape = {{1500, 512, 1}};

    std::vector<std::vector<uint64_t>> x2Shape = {{1500, 128, 1}};
    size_t yByteSize = 1500 * 512 * 158 * sizeof(float);
    size_t tilingDataSize = sizeof(MulAddnTilingData);

    uint8_t* x1 = CreateTensorList<float>(x1Shape);
    uint8_t* x2 = CreateTensorList<float>(x2Shape);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 63;

    char* path_ = get_current_dir_name();
    string path(path_);

    MulAddnTilingData* tilingDatafromBin = reinterpret_cast<MulAddnTilingData*>(tiling);

    tilingDatafromBin->N = 1;
    tilingDatafromBin->shapeB = 1500;
    tilingDatafromBin->shapeM = 512;
    tilingDatafromBin->shapeN = 128;
    tilingDatafromBin->shapeNAlign = 128;
    tilingDatafromBin->coreTaskNum = 24;
    tilingDatafromBin->useCoreNum = 63;
    tilingDatafromBin->lastCoreTaskNum = 12;
    tilingDatafromBin->mNum = 128;
    tilingDatafromBin->mLoopNum = 4;
    tilingDatafromBin->mNumTail = 128;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(mul_addn, blockDim, x1, x2, y, workspace, tiling);

    FreeTensorList<float>(x1, x1Shape);
    FreeTensorList<float>(x2, x2Shape);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
    system("rm -rf mul_addn_data");
}
