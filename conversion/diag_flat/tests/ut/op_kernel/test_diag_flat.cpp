/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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