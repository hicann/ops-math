/**
 * Copyright (c) Huawei Technologies Co., Ltd.2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_stack_ball_query.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API文档
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xyzShape = {20,3};
    std::vector<int64_t> centerXyzShape = {10, 3};
    std::vector<int64_t> xyzBatchCntShape = {2};
    std::vector<int64_t> centerXyzBatchCntShape = {2};
    std::vector<int64_t> outShape = {10, 5};

    void* xyzDeviceAddr = nullptr;
    void* centerXyzDeviceAddr = nullptr;
    void* xyzBatchCntDeviceAddr = nullptr;
    void* centerXyzBatchCntDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* xyz = nullptr;
    aclTensor* centerXyz = nullptr;
    aclTensor* xyzBatchCnt = nullptr;
    aclTensor* centerXyzBatchCnt = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> xyzData = {-0.0740, 1.3147, -1.3625, 0.5555, 1.0399, -1.3634,
                                -0.4003, 2.4666, -0.5116, -0.5251, 2.4379, -0.8466,
                                -0.9691, 1.1418, -1.3733, -0.2232, 0.9561, -1.3626,
                                -2.2769, 2.7817, -0.2334, -0.2822, 1.3192, -1.3645,
                                0.1533, 1.5024, -1.0432, 0.4917, 1.1529, -1.3496,
                                -2.0289, 2.4952, -0.1708, -0.7188, 0.9956, -0.5096,
                                -2.0668, 6.0278, -0.4875, -1.9304, 3.3092, 0.6610,
                                0.0949, 1.4332, 0.3140, -1.2879, 2.0008, -0.7791,
                                -0.7252, 0.9611, -0.6371, 0.4066, 1.4211, -0.2947,
                                0.3220, 1.4447, 0.3548, -0.9744, 2.3856, -1.2000};
    std::vector<float> centerXyzData = {-0.0740, 1.3147, -1.3625, -2.2769, 2.7817, -0.2334,
                                    -0.4003, 2.4666, -0.5116, -0.0740, 1.3147, -1.3625,
                                    -0.0740, 1.3147, -1.3625, -2.0289, 2.4952, -0.1708,
                                    -2.0668, 6.0278, -0.4875, 0.4066, 1.4211, -0.2947,
                                    -2.0289, 2.4952, -0.1708, -2.0289, 2.4952, -0.1708};
    std::vector<int32_t> xyzBatchCntData = {10, 10};
    std::vector<int32_t> centerXyzBatchCntData = {5, 5};
    std::vector<int32_t> outData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 创建in aclTensor
    ret = CreateAclTensor(xyzData, xyzShape, &xyzDeviceAddr, aclDataType::ACL_FLOAT, &xyz);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(centerXyzData, centerXyzShape, &centerXyzDeviceAddr, aclDataType::ACL_FLOAT, &centerXyz);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(xyzBatchCntData, xyzBatchCntShape, &xyzBatchCntDeviceAddr, aclDataType::ACL_INT32, &xyzBatchCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建in aclTensor
    ret = CreateAclTensor(centerXyzBatchCntData, centerXyzBatchCntShape, &centerXyzBatchCntDeviceAddr, aclDataType::ACL_INT32, &centerXyzBatchCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double maxRadius = 0.2;
    int64_t sampleNum = 5;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnStackBallQuery第一段接口
    ret = aclnnStackBallQueryGetWorkspaceSize(xyz, centerXyz, xyzBatchCnt, centerXyzBatchCnt, maxRadius, sampleNum, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStackBallQueryGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnStackBallQuery第二段接口
    ret = aclnnStackBallQuery(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStackBallQuery failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(xyz);
    aclDestroyTensor(centerXyz);
    aclDestroyTensor(xyzBatchCnt);
    aclDestroyTensor(centerXyzBatchCnt);
    aclDestroyTensor(out);

    // 7. 释放device资源
    aclrtFree(xyzDeviceAddr);
    aclrtFree(centerXyzDeviceAddr);
    aclrtFree(xyzBatchCntDeviceAddr);
    aclrtFree(centerXyzBatchCntDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}