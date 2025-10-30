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
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "kernel_fp16.h"
#include "gtest/gtest.h"
#include "test_diag_flat_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#ifdef __CCE_KT_TEST__
#include <cstdint>
#include "tikicpulib.h"
#endif

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
using namespace std;

extern "C" __global__ __aicore__ void diag_flat(GM_ADDR input, GM_ADDR output, GM_ADDR tiling, GM_ADDR workspace);

class diag_flat_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "========== diag_flat_test SetUp ==========\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "========== diag_flat_test TearDown ==========\n" << std::endl;
    }
};

size_t GetFileSizeForDiagFlat(const std::string& filePath) {
    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return 0;
    }

    std::filebuf* buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    return size;
}

bool ReadFileForDiagFlat(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize) {
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file, %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t size = GetFileSizeForDiagFlat(filePath);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    std::filebuf* buf = file.rdbuf();
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char*>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

struct DiagFlatCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(diag_flat_test, test_case_3) {
    DiagFlatCompileInfo compileInfo = {48, 196608, false};
    
    gert::TilingContextPara tilingContextPara("DiagFlat",
                                              {{{{128,}, {128,}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("diagonal", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    // input size align
    size_t inputBytesSize = 128 * 2;
    size_t outputBytesSize = 128 * 128 * 2;
    size_t tiling_data_size = sizeof(DiagV2TilingDataTest);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputBytesSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputBytesSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    uint32_t blockDim = 1;

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(diag_flat, blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}