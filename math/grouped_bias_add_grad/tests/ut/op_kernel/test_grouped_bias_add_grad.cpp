/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "test_grouped_bias_add_grad.h"


#ifdef __CCE_KT_TEST__
#include <cstdint>
#include "tikicpulib.h"
#include "../data_utils.h"
#endif

extern "C" __global__ __aicore__ void grouped_bias_add_grad(GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias,
                                                            GM_ADDR workspace, GM_ADDR tiling);

class grouped_bias_add_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "grouped_bias_add_grad_test SetUp\n" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "grouped_bias_add_grad_test TearDown\n" << std::endl;
  }
};

TEST_F(grouped_bias_add_grad_test, test_case_float16_no_group_idx) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 0");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case0'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 16;

  size_t gradYByteSize = 16 * 568 * 128 * sizeof(half);
  size_t outByteSize = 16 * 128 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000100);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, nullptr, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_float32_no_group_idx) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 1");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case1'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 16;

  size_t gradYByteSize = 16 * 2000 * 128 * sizeof(float);
  size_t outByteSize = 16 * 128 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000001);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, nullptr, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(grouped_bias_add_grad_test, test_case_bf16_group_idx) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 2");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case2'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 6;

  size_t gradYByteSize = 100 * 256 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int32_t);
  size_t outByteSize = 3 * 256 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000112);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 3");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case3'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 12;

  size_t gradYByteSize = 1968 * 458 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int32_t);
  size_t outByteSize = 3 * 458 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp32_group_idx) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 4");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case4'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 16;

  size_t gradYByteSize = 3200 * 399 * sizeof(float);
  size_t groupIdxByteSize = 4 * sizeof(int32_t);
  size_t outByteSize = 4 * 399 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000011);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_16) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 5");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case5'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 8875 * 1228 * sizeof(half);
  size_t groupIdxByteSize = 16 * sizeof(int32_t);
  size_t outByteSize = 16 * 1228 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_c0) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 6");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case6'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 6;

  size_t gradYByteSize = 100 * 256 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int32_t);
  size_t outByteSize = 3 * 256 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp32_group_idx_perf) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 7");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case7'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 1968 * 2560 * sizeof(float);
  size_t groupIdxByteSize = 224 * sizeof(int32_t);
  size_t outByteSize = 224 * 2560 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1001011);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(grouped_bias_add_grad_test, test_case_bf16_group_idx_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 8");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case8'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 6;

  size_t gradYByteSize = 100 * 256 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int64_t);
  size_t outByteSize = 3 * 256 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010112);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 9");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case9'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 12;

  size_t gradYByteSize = 1968 * 458 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int64_t);
  size_t outByteSize = 3 * 458 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp32_group_idx_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 10");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case10'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 16;

  size_t gradYByteSize = 3200 * 399 * sizeof(float);
  size_t groupIdxByteSize = 4 * sizeof(int64_t);
  size_t outByteSize = 4 * 399 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010011);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_16_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 11");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case11'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 8875 * 1228 * sizeof(half);
  size_t groupIdxByteSize = 16 * sizeof(int64_t);
  size_t outByteSize = 16 * 1228 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_c0_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 12");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case12'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 6;

  size_t gradYByteSize = 100 * 256 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int64_t);
  size_t outByteSize = 3 * 256 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp32_group_idx_perf_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 13");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case13'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 1968 * 2560 * sizeof(float);
  size_t groupIdxByteSize = 224 * sizeof(int64_t);
  size_t outByteSize = 224 * 2560 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1011011);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_perf_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 14");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case14'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 1968 * 2560 * sizeof(half);
  size_t groupIdxByteSize = 224 * sizeof(int64_t);
  size_t outByteSize = 224 * 2560 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1011010);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_bf16_group_idx_perf_int64) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 14");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case14'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 1968 * 2560 * sizeof(half);
  size_t groupIdxByteSize = 224 * sizeof(int64_t);
  size_t outByteSize = 224 * 2560 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1011012);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_bias_add_grad_test, test_case_bf16_group_idx_perf_int64_with_ub_sum) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 16");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case16'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 300 * 2560 * sizeof(half);
  size_t groupIdxByteSize = 300 * sizeof(int64_t);
  size_t outByteSize = 300 * 2560 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1011112);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_perf_int64_with_ub_sum) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 17");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case17'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 300 * 2560 * sizeof(half);
  size_t groupIdxByteSize = 300 * sizeof(int64_t);
  size_t outByteSize = 300 * 2560 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1011110);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp32_group_idx_perf_int64_with_ub_sum) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 18");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case18'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 300 * 2560 * sizeof(float);
  size_t groupIdxByteSize = 300 * sizeof(int64_t);
  size_t outByteSize = 300 * 2560 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1011111);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx_int64_dimg_0) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 19");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case19'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 1 * 2560 * sizeof(half);
  size_t groupIdxByteSize = 1 * sizeof(int64_t);
  size_t outByteSize = 1 * 2560 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010110);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(grouped_bias_add_grad_test, test_case_fp32_group_idx_int64_dimg_0) {
  system(
      "cp -rf "
      "../../../../../../ops/math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 20");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case20'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t blockDim = 40;

  size_t gradYByteSize = 1 * 2560 * sizeof(float);
  size_t groupIdxByteSize = 1 * sizeof(int64_t);
  size_t outByteSize = 1 * 2560 * sizeof(float);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1010111);
  ICPU_RUN_KF(grouped_bias_add_grad, blockDim, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float32'");
}