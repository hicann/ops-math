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
 * \file test_acos.cpp
 * \brief
 */

#include "../../../op_kernel/acos.cpp"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

class AcosTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "AcosTest SetUp\n" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./acos_data/");
    }
    static void TearDownTestCase()
    {
        cout << "AcosTest TearDown\n" << endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AcosTest::rootPath = "../../../../";
const std::string AcosTest::dataPath = rootPath + "math/acos/tests/ut/op_kernel/acos_data";

TEST_F(AcosTest, test_case_0)
{
    size_t xByteSize = 32 * 8 * 7 * 7 * sizeof(float);
    size_t yByteSize = 32 * 8 * 7 * 7 * sizeof(float);
    size_t tiling_data_size = sizeof(AcosTilingData);
    uint32_t blockDim = 8;

    system("cd ./acos_data/ && python3 gen_data.py '(32, 8, 7, 7)' 'float32'");
    std::string fileName = "./acos_data/float32_input_acos.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AcosTilingData* tilingDatafromBin = reinterpret_cast<AcosTilingData*>(tiling);
    tilingDatafromBin->formerCoreNum = 4;
    tilingDatafromBin->tailCoreNum = 4;
    tilingDatafromBin->formerCoreDataNum = 1600;
    tilingDatafromBin->tailCoreDataNum = 1536;
    tilingDatafromBin->formerCoreLoopCount = 25;
    tilingDatafromBin->formerCoreFormerDataNum = 64; // 测试发现CPU上做Select选择的时候如果单片数据超过256字节会出错
    tilingDatafromBin->formerCoreTailDataNum = 64; // 测试发现CPU上做Select选择的时候如果单片数据超过256字节会出错
    tilingDatafromBin->tailCoreLoopCount = 24;
    tilingDatafromBin->tailCoreFormerDataNum = 64;
    tilingDatafromBin->tailCoreTailDataNum = 64;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(acos<float>, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    fileName = "./acos_data/float32_output_acos.bin";
    WriteFile(fileName, y, yByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./acos_data/ && python3 compare_data.py 'float32'");
    free(path_);
}
