/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_population_count.cpp
 * @brief PopulationCount 算子 ACLNN 调用示例（两段式接口）
 */

#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn_population_count.h"

#define CHECK_ACL(expr)                                                                 \
    do {                                                                                \
        auto _ret = (expr);                                                             \
        if (_ret != ACL_SUCCESS) {                                                      \
            std::cerr << "ACL Error: " << #expr << " returned " << _ret                 \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;            \
            goto cleanup;                                                               \
        }                                                                               \
    } while (0)

// CPU Golden: 16-bit popcount (与 tf.bitwise.population_count 语义一致；int16 负数按补码逐位计数)
static inline uint8_t CpuPopcount16(uint16_t v) {
    return static_cast<uint8_t>(__builtin_popcount(static_cast<uint32_t>(v)));
}

// 通用：运行单个 case（一组 shape=[16] 输入）。dtype 由 aclDataType 参数指定。
// 返回值：0=PASS，非 0=FAIL 或运行时异常。
// label: 打印前缀（如 "INT16" / "UINT16"）
// rawU16: 每个元素的 16-bit 位模式（以 uint16_t 表达），kernel 侧按 dtype 重新解释
static int RunOneCase(const char* label,
                      aclDataType dtype,
                      const uint16_t* rawU16,
                      int64_t elemCount,
                      aclrtStream stream)
{
    const int64_t shape[]   = {elemCount};
    const int64_t strides[] = {1};
    constexpr int64_t ndim  = 1;

    int32_t ret = 1;
    void *devX = nullptr;
    void *devY = nullptr;
    void *workspace = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *yTensor = nullptr;

    size_t inputBytes  = elemCount * sizeof(uint16_t);  // INT16 / UINT16 同为 2 字节
    size_t outputBytes = elemCount * sizeof(uint8_t);

    // Host 端构造输入和 CPU Golden
    std::vector<uint16_t> hostX(rawU16, rawU16 + elemCount);
    std::vector<uint8_t>  expected(elemCount, 0);
    for (int64_t i = 0; i < elemCount; ++i) {
        expected[i] = CpuPopcount16(hostX[i]);
    }

    CHECK_ACL(aclrtMalloc(&devX, inputBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devY, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devX, inputBytes, hostX.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(devY, outputBytes, 0, outputBytes));

    xTensor = aclCreateTensor(shape, ndim, dtype, strides, 0,
                              ACL_FORMAT_ND, shape, ndim, devX);
    yTensor = aclCreateTensor(shape, ndim, ACL_UINT8, strides, 0,
                              ACL_FORMAT_ND, shape, ndim, devY);
    if (xTensor == nullptr || yTensor == nullptr) {
        std::cerr << "[" << label << "] aclCreateTensor failed" << std::endl;
        goto cleanup;
    }

    {
        uint64_t       workspaceSize = 0;
        aclOpExecutor *executor      = nullptr;

        CHECK_ACL(aclnnPopulationCountGetWorkspaceSize(
            xTensor, yTensor, &workspaceSize, &executor));

        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        CHECK_ACL(aclnnPopulationCount(workspace, workspaceSize, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));

        std::vector<uint8_t> hostY(elemCount, 0);
        CHECK_ACL(aclrtMemcpy(hostY.data(), outputBytes, devY, outputBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "PopulationCount Example [" << label << "] (shape: [" << elemCount
                  << "], y: uint8)" << std::endl;
        std::cout << "  y[i] = popcount(x[i])  (16-bit)" << std::endl;
        std::cout << "-----------------------------------------------------------------" << std::endl;
        std::printf("  %4s | %11s | %7s | %6s | %6s | %s\n",
                    "Idx", "x(u16)", "x(hex)", "NPU", "CPU", "Status");
        std::cout << "-----------------------------------------------------------------" << std::endl;

        int passCount = 0;
        for (int64_t i = 0; i < elemCount; ++i) {
            bool pass = (hostY[i] == expected[i]);
            passCount += pass ? 1 : 0;
            std::printf("  %4lld | %11u | 0x%04X | %6u | %6u | %s\n",
                        static_cast<long long>(i),
                        static_cast<unsigned int>(hostX[i]),
                        static_cast<unsigned int>(hostX[i]),
                        static_cast<unsigned int>(hostY[i]),
                        static_cast<unsigned int>(expected[i]),
                        pass ? "PASS" : "FAIL");
        }

        std::cout << "-----------------------------------------------------------------" << std::endl;
        std::cout << "[" << label << "]  Result: " << passCount << "/" << elemCount << " passed";
        if (passCount == elemCount) {
            std::cout << " -- ALL PASS" << std::endl;
            ret = 0;
        } else {
            std::cout << " -- FAILED" << std::endl;
            ret = 1;
        }
    }

cleanup:
    if (xTensor)   aclDestroyTensor(xTensor);
    if (yTensor)   aclDestroyTensor(yTensor);
    if (workspace) { aclrtFree(workspace); workspace = nullptr; }
    if (devX)      aclrtFree(devX);
    if (devY)      aclrtFree(devY);
    return ret;
}

int main()
{
    constexpr int64_t ELEM_COUNT = 16;

    // =========================================================================
    // Case 1: INT16 输入（覆盖正数/负数/边界）
    //   0（0）、1（1）、-1=0xFFFF（16）、-32768=0x8000（1）、0x5555（8）、
    //   0x7FFF（15）、0xAAAA=-21846（8）等
    // =========================================================================
    int16_t int16Raw[ELEM_COUNT] = {
        /* idx  value(dec)   hex     popcount */
        /*  0 */ 0,              // 0x0000 -> 0
        /*  1 */ 1,              // 0x0001 -> 1
        /*  2 */ 2,              // 0x0002 -> 1
        /*  3 */ 3,              // 0x0003 -> 2
        /*  4 */ 7,              // 0x0007 -> 3
        /*  5 */ 15,             // 0x000F -> 4
        /*  6 */ 255,            // 0x00FF -> 8
        /*  7 */ 256,            // 0x0100 -> 1
        /*  8 */ 21845,          // 0x5555 -> 8
        /*  9 */ (int16_t)43690, // 0xAAAA -> 8  (int16 负数 -21846)
        /* 10 */ (int16_t)0x7FFF,// 0x7FFF -> 15
        /* 11 */ (int16_t)0x8000,// 0x8000 -> 1   (int16 = -32768，符号位计入)
        /* 12 */ -1,             // 0xFFFF -> 16  (int16 补码全 1)
        /* 13 */ -2,             // 0xFFFE -> 15
        /* 14 */ 16,             // 0x0010 -> 1
        /* 15 */ 4660,           // 0x1234 -> 5
    };
    uint16_t int16RawU16[ELEM_COUNT];
    for (int i = 0; i < ELEM_COUNT; ++i) {
        int16RawU16[i] = static_cast<uint16_t>(int16Raw[i]);
    }

    // =========================================================================
    // Case 2: UINT16 输入（覆盖典型无符号位模式：0x0000 / 0x8000 / 0x5555 /
    //         0xAAAA / 0xFFFF，以及 0x7FFF、0x0F0F、0xF0F0 等常用边界）
    // =========================================================================
    uint16_t uint16Raw[ELEM_COUNT] = {
        /* idx  value(hex)   popcount */
        /*  0 */ 0x0000,  //  0
        /*  1 */ 0x0001,  //  1
        /*  2 */ 0x8000,  //  1  (MSB only)
        /*  3 */ 0x5555,  //  8
        /*  4 */ 0xAAAA,  //  8
        /*  5 */ 0xFFFF,  // 16  (all ones)
        /*  6 */ 0x7FFF,  // 15
        /*  7 */ 0x0F0F,  //  8
        /*  8 */ 0xF0F0,  //  8
        /*  9 */ 0x00FF,  //  8
        /* 10 */ 0xFF00,  //  8
        /* 11 */ 0x1234,  //  5
        /* 12 */ 0xDEAD,  // 11
        /* 13 */ 0xBEEF,  // 13
        /* 14 */ 0xC3C3,  //  8
        /* 15 */ 0x3C3C,  //  8
    };

    // =========================================================================
    // ACL 初始化（整个进程仅一次）
    // =========================================================================
    int32_t ret = 1;
    aclrtStream stream = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateStream(&stream));

    {
        int retInt16  = RunOneCase("INT16",  ACL_INT16,  int16RawU16, ELEM_COUNT, stream);
        std::cout << std::endl;
        int retUint16 = RunOneCase("UINT16", ACL_UINT16, uint16Raw,   ELEM_COUNT, stream);

        std::cout << "=================================================================" << std::endl;
        std::cout << "Overall: " << ((retInt16 == 0 && retUint16 == 0) ? "ALL PASS" : "FAILED")
                  << std::endl;
        std::cout << "=================================================================" << std::endl;
        ret = (retInt16 == 0 && retUint16 == 0) ? 0 : 1;
    }

cleanup:
    if (stream)    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return ret;
}
