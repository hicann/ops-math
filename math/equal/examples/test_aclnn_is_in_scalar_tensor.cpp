/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isin.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using ScalarPtr = std::unique_ptr<aclScalar, decltype(&aclDestroyScalar)>;

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

aclError InitAcl(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

aclError CreateInputs(std::vector<int64_t>& testElementsShape, std::vector<int64_t>& outShape,
                      DeviceMemPtr& testElementsDeviceAddr, DeviceMemPtr& outDeviceAddr, TensorPtr& testElements,
                      TensorPtr& out, ScalarPtr& element, bool& assumeUnique, bool& invert)
{
    std::vector<double> testElementsHostData = {1.0, 2.0, 3.0};
    std::vector<char> outHostData = {0};
    double elementValue = 4.0;

    // 创建 testElements Tensor
    auto ret = CreateAclTensor(testElementsHostData, testElementsShape, aclDataType::ACL_DOUBLE, testElementsDeviceAddr,
                               testElements);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 element Scalar
    aclScalar* rawScalar = aclCreateScalar(&elementValue, aclDataType::ACL_DOUBLE);
    CHECK_RET(rawScalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
    element.reset(rawScalar);

    // 创建 out Tensor
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_BOOL, outDeviceAddr, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

aclError ExecOpApi(ScalarPtr& element, TensorPtr& testElements, bool assumeUnique, bool invert, TensorPtr& out,
                   DeviceMemPtr& workspaceAddr, uint64_t& workspaceSize, DeviceMemPtr& outDeviceAddr, StreamPtr& stream)
{
    aclOpExecutor* executor;

    // 第一段接口
    auto ret = aclnnIsInScalarTensorGetWorkspaceSize(element.get(), testElements.get(), assumeUnique, invert, out.get(),
                                                     &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 分配 workspace
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    // 第二段接口
    ret = aclnnIsInScalarTensor(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsInScalarTensor failed. ERROR: %d\n", ret); return ret);

    // 同步
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 输出拷贝
    char resultData = 0;
    ret = aclrtMemcpy(&resultData, sizeof(char), outDeviceAddr.get(), sizeof(char), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("result is: %d\n", static_cast<bool>(resultData));

    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);

    auto ret = InitAcl(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> testElementsShape = {3};
    std::vector<int64_t> outShape = {};
    DeviceMemPtr testElementsDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
    TensorPtr testElements(nullptr, &aclDestroyTensor);
    TensorPtr out(nullptr, &aclDestroyTensor);
    ScalarPtr element(nullptr, &aclDestroyScalar);

    bool assumeUnique = false;
    bool invert = true;

    ret = CreateInputs(testElementsShape, outShape, testElementsDeviceAddr, outDeviceAddr, testElements, out, element,
                       assumeUnique, invert);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);

    ret = ExecOpApi(element, testElements, assumeUnique, invert, out, workspaceAddr, workspaceSize, outDeviceAddr,
                    stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return 0;
}
