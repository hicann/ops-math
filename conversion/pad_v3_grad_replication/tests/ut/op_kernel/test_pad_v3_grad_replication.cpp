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
/*!
 * \file test_pad_v3_grad_replication.cpp
 * \brief
 */
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <numeric>
#include <functional>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_data_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void pad_v3_grad_replication(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class pad_v3_grad_replication_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "pad_v3_grad_replication_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "pad_v3_grad_replication_test TearDown\n" << std::endl;
    }
};

TEST_F(pad_v3_grad_replication_test, test_small_case_0)
{
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(20072448 * 4);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(16777216 * 4);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(13107200);
    uint8_t* padvalues = (uint8_t*)AscendC::GmAlloc(10 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(PadV3GradReplicationTilingData));

    char* path_ = get_current_dir_name();
    string path(path_);

    PadV3GradReplicationTilingData* tilingData = reinterpret_cast<PadV3GradReplicationTilingData*>(tiling);
    // 填充基本变量
    tilingData->addTensorBlockNum = 2048;
    tilingData->addTensorByteSize = 65536;
    tilingData->addTensorSize = 16384;
    tilingData->moveTensorBlockNum = 6144;
    tilingData->moveTensorByteSize = 196608;
    tilingData->moveTensorSize = 49152;

    // 填充虚拟输入形状
    tilingData->inputShape[0] = 256;
    tilingData->inputShape[1] = 18;
    tilingData->inputShape[2] = 66;
    tilingData->inputShape[3] = 66;
    // 填充其他基本变量
    tilingData->inputSize = 20072448;
    tilingData->cubeInputSize = 78408;
    tilingData->layerInputSize = 4356;
    tilingData->cubeNumEachCore = 6;
    tilingData->realUsedCoreNum = 43;
    tilingData->cubeNumLastCore = 4;
    tilingData->outputShape[0] = 256;
    tilingData->outputShape[1] = 16;
    tilingData->outputShape[2] = 64;
    tilingData->outputShape[3] = 64;
    tilingData->outputSize = 16777216;
    tilingData->cubeOutputSize = 65536;
    tilingData->layerOutputSize = 4096;
    tilingData->paddings[0] = 1;
    tilingData->paddings[1] = 1;
    tilingData->paddings[2] = 1;
    tilingData->paddings[3] = 1;
    tilingData->paddings[4] = 1;
    tilingData->paddings[5] = 1;
    tilingData->topSize = 66;
    tilingData->totalTopInputSizeEachCube = 1188;
    tilingData->leftSize = 62;
    tilingData->totalLeftInputSizeEachCube = 1116;
    tilingData->innerRowLength = 62;
    tilingData->topToBottomSize = 4224;
    tilingData->topResultSize = 304128;
    tilingData->leftToRightSize = 64;
    tilingData->leftResultSize = 285696;
    tilingData->workspaceSize = 3276800;

    // 填充 EdgeTiling 结构体
    tilingData->topTiling.edgeCount = 227;
    tilingData->topTiling.tileCount = 0;
    tilingData->topTiling.additionalCount = 108;

    tilingData->leftTiling.edgeCount = 32;
    tilingData->leftTiling.tileCount = 3;
    tilingData->leftTiling.additionalCount = 12;

    tilingData->cornerTiling.edgeCount = 0;
    tilingData->cornerTiling.tileCount = 0;
    tilingData->cornerTiling.additionalCount = 108;

    tilingData->innerTiling.edgeCount = 768;
    tilingData->innerTiling.tileCount = 0;
    tilingData->innerTiling.additionalCount = 62;

    tilingData->paddingLayerTiling.edgeCount = 12;
    tilingData->paddingLayerTiling.tileCount = 0;
    tilingData->paddingLayerTiling.additionalCount = 6;

    tilingData->topTilingLastCore.edgeCount = 227;
    tilingData->topTilingLastCore.tileCount = 0;
    tilingData->topTilingLastCore.additionalCount = 72;

    tilingData->leftTilingLastCore.edgeCount = 32;
    tilingData->leftTilingLastCore.tileCount = 2;
    tilingData->leftTilingLastCore.additionalCount = 8;

    tilingData->cornerTilingLastCore.edgeCount = 0;
    tilingData->cornerTilingLastCore.tileCount = 0;
    tilingData->cornerTilingLastCore.additionalCount = 72;

    tilingData->paddingLayerTilingLastCore.edgeCount = 12;
    tilingData->paddingLayerTilingLastCore.tileCount = 0;
    tilingData->paddingLayerTilingLastCore.additionalCount = 4;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        pad_v3_grad_replication, tilingData->realUsedCoreNum, input, padvalues, output, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(padvalues);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(pad_v3_grad_replication_test, test_small_case_1)
{
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(995328 * 4);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(524288 * 4);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1835008);
    uint8_t* padvalues = (uint8_t*)AscendC::GmAlloc(10 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(PadV3GradReplicationTilingData));

    char* path_ = get_current_dir_name();
    string path(path_);

    PadV3GradReplicationTilingData* tilingData = reinterpret_cast<PadV3GradReplicationTilingData*>(tiling);
    // 填充基本变量
    tilingData->addTensorBlockNum = 2048;
    tilingData->addTensorByteSize = 65536;
    tilingData->addTensorSize = 16384;
    tilingData->moveTensorBlockNum = 6144;
    tilingData->moveTensorByteSize = 196608;
    tilingData->moveTensorSize = 49152;

    // 填充虚拟输入形状
    tilingData->inputShape[0] = 512;
    tilingData->inputShape[1] = 6;
    tilingData->inputShape[2] = 18;
    tilingData->inputShape[3] = 18;

    // 填充其他基本变量
    tilingData->inputSize = 995328;
    tilingData->cubeInputSize = 1944;
    tilingData->layerInputSize = 324;
    tilingData->cubeNumEachCore = 11;
    tilingData->realUsedCoreNum = 47;
    tilingData->cubeNumLastCore = 6;
    tilingData->outputShape[0] = 512;
    tilingData->outputShape[1] = 4;
    tilingData->outputShape[2] = 16;
    tilingData->outputShape[3] = 16;
    tilingData->outputSize = 524288;
    tilingData->cubeOutputSize = 1024;
    tilingData->layerOutputSize = 256;
    tilingData->paddings[0] = 1;
    tilingData->paddings[1] = 1;
    tilingData->paddings[2] = 1;
    tilingData->paddings[3] = 1;
    tilingData->paddings[4] = 1;
    tilingData->paddings[5] = 1;
    tilingData->topSize = 18;
    tilingData->totalTopInputSizeEachCube = 108;
    tilingData->leftSize = 14;
    tilingData->totalLeftInputSizeEachCube = 84;
    tilingData->innerRowLength = 14;
    tilingData->topToBottomSize = 288;
    tilingData->topResultSize = 55296;
    tilingData->leftToRightSize = 16;
    tilingData->leftResultSize = 43008;
    tilingData->workspaceSize = 458752;

    // 填充 EdgeTiling 结构体
    tilingData->topTiling.edgeCount = 682;
    tilingData->topTiling.tileCount = 0;
    tilingData->topTiling.additionalCount = 66;

    tilingData->leftTiling.edgeCount = 128;
    tilingData->leftTiling.tileCount = 0;
    tilingData->leftTiling.additionalCount = 66;

    tilingData->cornerTiling.edgeCount = 0;
    tilingData->cornerTiling.tileCount = 0;
    tilingData->cornerTiling.additionalCount = 66;

    tilingData->innerTiling.edgeCount = 3072;
    tilingData->innerTiling.tileCount = 0;
    tilingData->innerTiling.additionalCount = 14;

    tilingData->paddingLayerTiling.edgeCount = 192;
    tilingData->paddingLayerTiling.tileCount = 0;
    tilingData->paddingLayerTiling.additionalCount = 11;

    tilingData->topTilingLastCore.edgeCount = 682;
    tilingData->topTilingLastCore.tileCount = 0;
    tilingData->topTilingLastCore.additionalCount = 36;

    tilingData->leftTilingLastCore.edgeCount = 128;
    tilingData->leftTilingLastCore.tileCount = 0;
    tilingData->leftTilingLastCore.additionalCount = 36;

    tilingData->cornerTilingLastCore.edgeCount = 0;
    tilingData->cornerTilingLastCore.tileCount = 0;
    tilingData->cornerTilingLastCore.additionalCount = 36;

    tilingData->paddingLayerTilingLastCore.edgeCount = 192;
    tilingData->paddingLayerTilingLastCore.tileCount = 0;
    tilingData->paddingLayerTilingLastCore.additionalCount = 6;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        pad_v3_grad_replication, tilingData->realUsedCoreNum, input, padvalues, output, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(padvalues);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(pad_v3_grad_replication_test, test_tiny_case_2)
{
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(32000 * 4);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(16384 * 4);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(57344);
    uint8_t* padvalues = (uint8_t*)AscendC::GmAlloc(10 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(PadV3GradReplicationTilingData));

    char* path_ = get_current_dir_name();
    string path(path_);

    PadV3GradReplicationTilingData* tilingData = reinterpret_cast<PadV3GradReplicationTilingData*>(tiling);
    // 填充基本变量
    tilingData->addTensorBlockNum = 2048;
    tilingData->addTensorByteSize = 65536;
    tilingData->addTensorSize = 16384;
    tilingData->moveTensorBlockNum = 6144;
    tilingData->moveTensorByteSize = 196608;
    tilingData->moveTensorSize = 49152;

    // 填充虚拟输入形状
    tilingData->inputShape[0] = 32;
    tilingData->inputShape[1] = 10;
    tilingData->inputShape[2] = 10;
    tilingData->inputShape[3] = 10;

    // 填充其他基本变量
    tilingData->inputSize = 32000;
    tilingData->cubeInputSize = 1000;
    tilingData->layerInputSize = 100;
    tilingData->cubeNumEachCore = 1;
    tilingData->realUsedCoreNum = 32;
    tilingData->cubeNumLastCore = 1;
    tilingData->outputShape[0] = 32;
    tilingData->outputShape[1] = 8;
    tilingData->outputShape[2] = 8;
    tilingData->outputShape[3] = 8;
    tilingData->outputSize = 16384;
    tilingData->cubeOutputSize = 512;
    tilingData->layerOutputSize = 64;
    tilingData->paddings[0] = 1;
    tilingData->paddings[1] = 1;
    tilingData->paddings[2] = 1;
    tilingData->paddings[3] = 1;
    tilingData->paddings[4] = 1;
    tilingData->paddings[5] = 1;
    tilingData->topSize = 10;
    tilingData->totalTopInputSizeEachCube = 100;
    tilingData->leftSize = 6;
    tilingData->totalLeftInputSizeEachCube = 60;
    tilingData->innerRowLength = 6;
    tilingData->topToBottomSize = 80;
    tilingData->topResultSize = 3200;
    tilingData->leftToRightSize = 8;
    tilingData->leftResultSize = 1920;
    tilingData->workspaceSize = 14336;

    // 填充 EdgeTiling 结构体
    tilingData->topTiling.edgeCount = 1024;
    tilingData->topTiling.tileCount = 0;
    tilingData->topTiling.additionalCount = 10;

    tilingData->leftTiling.edgeCount = 256;
    tilingData->leftTiling.tileCount = 0;
    tilingData->leftTiling.additionalCount = 10;

    tilingData->cornerTiling.edgeCount = 0;
    tilingData->cornerTiling.tileCount = 0;
    tilingData->cornerTiling.additionalCount = 10;

    tilingData->innerTiling.edgeCount = 6144;
    tilingData->innerTiling.tileCount = 0;
    tilingData->innerTiling.additionalCount = 6;

    tilingData->paddingLayerTiling.edgeCount = 768;
    tilingData->paddingLayerTiling.tileCount = 0;
    tilingData->paddingLayerTiling.additionalCount = 1;

    tilingData->topTilingLastCore.edgeCount = 1024;
    tilingData->topTilingLastCore.tileCount = 0;
    tilingData->topTilingLastCore.additionalCount = 10;

    tilingData->leftTilingLastCore.edgeCount = 256;
    tilingData->leftTilingLastCore.tileCount = 0;
    tilingData->leftTilingLastCore.additionalCount = 10;

    tilingData->cornerTilingLastCore.edgeCount = 0;
    tilingData->cornerTilingLastCore.tileCount = 0;
    tilingData->cornerTilingLastCore.additionalCount = 10;

    tilingData->paddingLayerTilingLastCore.edgeCount = 768;
    tilingData->paddingLayerTilingLastCore.tileCount = 0;
    tilingData->paddingLayerTilingLastCore.additionalCount = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        pad_v3_grad_replication, tilingData->realUsedCoreNum, input, padvalues, output, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(padvalues);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(pad_v3_grad_replication_test, test_blank_pad_case3)
{
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(640000 * 4);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(451584 * 4);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(931328);
    uint8_t* padvalues = (uint8_t*)AscendC::GmAlloc(10 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(PadV3GradReplicationTilingData));

    char* path_ = get_current_dir_name();
    string path(path_);

    PadV3GradReplicationTilingData* tilingData = reinterpret_cast<PadV3GradReplicationTilingData*>(tiling);
    // 填充基本变量
    tilingData->addTensorBlockNum = 2048;
    tilingData->addTensorByteSize = 65536;
    tilingData->addTensorSize = 16384;
    tilingData->moveTensorBlockNum = 6144;
    tilingData->moveTensorByteSize = 196608;
    tilingData->moveTensorSize = 49152;

    // 填充虚拟输入形状
    tilingData->inputShape[0] = 32;
    tilingData->inputShape[1] = 10;
    tilingData->inputShape[2] = 10;
    tilingData->inputShape[3] = 200;

    // 填充其他基本变量
    tilingData->inputSize = 640000;
    tilingData->cubeInputSize = 20000;
    tilingData->layerInputSize = 2000;
    tilingData->cubeNumEachCore = 1;
    tilingData->realUsedCoreNum = 32;
    tilingData->cubeNumLastCore = 1;
    tilingData->outputShape[0] = 32;
    tilingData->outputShape[1] = 9;
    tilingData->outputShape[2] = 8;
    tilingData->outputShape[3] = 196;
    tilingData->outputSize = 451584;
    tilingData->cubeOutputSize = 14112;
    tilingData->layerOutputSize = 1568;
    tilingData->paddings[0] = 2;
    tilingData->paddings[1] = 2;
    tilingData->paddings[2] = 2;
    tilingData->paddings[3] = 0;
    tilingData->paddings[4] = 0;
    tilingData->paddings[5] = 1;
    tilingData->topSize = 200;
    tilingData->totalTopInputSizeEachCube = 2000;
    tilingData->leftSize = 7;
    tilingData->totalLeftInputSizeEachCube = 70;
    tilingData->innerRowLength = 194;
    tilingData->topToBottomSize = 1800;
    tilingData->topResultSize = 64000;
    tilingData->leftToRightSize = 197;
    tilingData->leftResultSize = 2240;
    tilingData->workspaceSize = 232832;

    // 填充 EdgeTiling 结构体
    tilingData->topTiling.edgeCount = 81;
    tilingData->topTiling.tileCount = 0;
    tilingData->topTiling.additionalCount = 10;

    tilingData->leftTiling.edgeCount = 256;
    tilingData->leftTiling.tileCount = 0;
    tilingData->leftTiling.additionalCount = 10;

    tilingData->cornerTiling.edgeCount = 0;
    tilingData->cornerTiling.tileCount = 0;
    tilingData->cornerTiling.additionalCount = 10;

    tilingData->innerTiling.edgeCount = 245;
    tilingData->innerTiling.tileCount = 0;
    tilingData->innerTiling.additionalCount = 6;

    tilingData->paddingLayerTiling.edgeCount = 31;
    tilingData->paddingLayerTiling.tileCount = 0;
    tilingData->paddingLayerTiling.additionalCount = 1;

    tilingData->topTilingLastCore.edgeCount = 81;
    tilingData->topTilingLastCore.tileCount = 0;
    tilingData->topTilingLastCore.additionalCount = 10;

    tilingData->leftTilingLastCore.edgeCount = 256;
    tilingData->leftTilingLastCore.tileCount = 0;
    tilingData->leftTilingLastCore.additionalCount = 10;

    tilingData->cornerTilingLastCore.edgeCount = 0;
    tilingData->cornerTilingLastCore.tileCount = 0;
    tilingData->cornerTilingLastCore.additionalCount = 10;

    tilingData->paddingLayerTilingLastCore.edgeCount = 31;
    tilingData->paddingLayerTilingLastCore.tileCount = 0;
    tilingData->paddingLayerTilingLastCore.additionalCount = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        pad_v3_grad_replication, tilingData->realUsedCoreNum, input, padvalues, output, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(padvalues);
    AscendC::GmFree(tiling);
    free(path_);
}