/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"

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

int main()
{
    const int32_t deviceId = 0;
    const std::vector<int64_t> weightShape = {32, 32};
    const uint64_t logicalElements = static_cast<uint64_t>(GetShapeSize(weightShape));
    const std::vector<int8_t> hostWeight(logicalElements, 1);

    aclrtStream stream = nullptr;
    void* weightDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* weight = nullptr;
    aclOpExecutor* executor = nullptr;
    aclIntArray* shapeArray = nullptr;
    uint64_t workspaceSize = 0;
    uint64_t capacityElements = 0;
    uint64_t capacityBytes = 0;
    std::vector<int64_t> strides;
    std::vector<int64_t> storageShape;
    bool aclInitialized = false;
    bool deviceSet = false;
    bool streamCreated = false;
    int ret = ACL_SUCCESS;
    int result = ACL_SUCCESS;

    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    aclInitialized = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }
    deviceSet = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }
    streamCreated = true;

    shapeArray = aclCreateIntArray(weightShape.data(), weightShape.size());
    if (shapeArray == nullptr) {
        LOG_PRINT("aclCreateIntArray failed\n");
        result = ACL_ERROR_BAD_ALLOC;
        goto cleanup;
    }
    ret = aclnnCalculateMatmulWeightSizeV2(shapeArray, ACL_INT8, &capacityElements);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }
    aclDestroyIntArray(shapeArray);
    shapeArray = nullptr;

    // The size API returns the aligned capacity in elements. Allocate bytes, but
    // copy only the logical input so padding bytes remain unspecified.
    capacityBytes = capacityElements * sizeof(int8_t);
    if (capacityElements < logicalElements) {
        LOG_PRINT("calculated capacity is smaller than logical weight\n");
        result = ACL_ERROR_INVALID_PARAM;
        goto cleanup;
    }
    ret = aclrtMalloc(&weightDeviceAddr, capacityBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc weight failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }
    ret = aclrtMemcpy(weightDeviceAddr, capacityBytes, hostWeight.data(), logicalElements, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy weight to device failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }

    strides = {weightShape[1], 1};
    storageShape = {static_cast<int64_t>(capacityElements)};
    weight = aclCreateTensor(weightShape.data(), weightShape.size(), ACL_INT8, strides.data(), 0, ACL_FORMAT_ND,
                             storageShape.data(), storageShape.size(), weightDeviceAddr);
    if (weight == nullptr) {
        LOG_PRINT("aclCreateTensor failed\n");
        result = ACL_ERROR_BAD_ALLOC;
        goto cleanup;
    }

    ret = aclnnTransMatmulWeightGetWorkspaceSize(weight, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            result = ret;
            goto cleanup;
        }
    }
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        result = ret;
        goto cleanup;
    }

    {
        std::vector<int8_t> converted(capacityElements, 0);
        ret = aclrtMemcpy(converted.data(), capacityBytes, weightDeviceAddr, capacityBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy converted weight from device failed. ERROR: %d\n", ret);
            result = ret;
            goto cleanup;
        }
        for (uint64_t i = 0; i < logicalElements; ++i) {
            if (converted[i] != 1) {
                LOG_PRINT("converted weight mismatch at element %lu: got %d\n", i, converted[i]);
                result = ACL_ERROR_INVALID_PARAM;
                goto cleanup;
            }
        }
    }
    LOG_PRINT("capacity: %lu bytes\n", capacityBytes);
    LOG_PRINT("PASS\n");

cleanup:
    if (shapeArray != nullptr) {
        aclDestroyIntArray(shapeArray);
    }
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (weightDeviceAddr != nullptr) {
        aclrtFree(weightDeviceAddr);
    }
    if (streamCreated) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(deviceId);
    }
    if (aclInitialized) {
        aclFinalize();
    }
    return result;
}
