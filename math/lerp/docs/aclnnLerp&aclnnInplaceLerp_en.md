# aclnnLerp&aclnnInplaceLerp

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/lerp)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Performs linear interpolation between the start and end tensors based on the given weight and returns the interpolated tensor.

- Formula:

$$
\text { out }_i=\text { start }_i+\text { weight }_i \times\left(\text { end }_i-\text { start }_i\right)
$$

## Prototype

- `aclnnLerp` and `aclnnInplaceLerp` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnLerp`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceLerp`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLerpGetWorkspaceSize` or `aclnnInplaceLerpGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLerp` or `aclnnInplaceLerp` is called to perform computation.

  - `aclnnStatus aclnnLerpGetWorkspaceSize(const aclTensor* self, const aclTensor* end, const aclTensor* weight, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnLerp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceLerpGetWorkspaceSize(aclTensor* selfRef, const aclTensor* end, const aclTensor* weight, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnInplaceLerp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnLerpGetWorkspaceSize

- **Parameters:**
  - `self` (aclTensor\*, compute input): input `start` in the formula. It is an aclTensor on the device. The data type of self is the same as that of end, weight, and out. The shapes of self, end, and weight must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `end` (aclTensor\*, compute input): input `end` in the formula. It is an aclTensor on the device. The data type of self is the same as that of end, weight, and out. The shapes of self, end, and weight must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `weight` (aclTensor\*, compute input): input `weight` in the formula. It is an aclTensor on the device. The data type of self and end, weight, and out is the same. The shapes of self and end, weight, and out must meet the broadcast relationship (../../../docs/en/context/broadcast_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `out` (aclTensor\*, compute output): `out` in the formula. It is an aclTensor on the device. The data type of self is the same as that of end, weight, and out. The shape of out is the same as that of the tensor after self, end, and weight are broadcast. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, end, weight, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, end, weight, or out is not supported.
                                      2. The data types of self, end, weight, and out are inconsistent.
                                      3. Broadcasting cannot be performed for self, end, and weight. 
                                      4. The shapes of self, end, and weight after broadcasting are different from that of out.
  ```

## aclnnLerp

- **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnLerpGetWorkspaceSize.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLerpGetWorkspaceSize

- **Parameters:**
  - `selfRef` (aclTensor\*, compute input/output): `start` and `out` in the formula, aclTensor on the device. The data types of selfRef, end, and weight are the same. The shapes of selfRef, end, and weight meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md), and the shape after broadcasting is the same as that of `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `end` (aclTensor\*, compute input): `end` in the formula, aclTensor on the device. The data types of selfRef, end, and weight are the same. The shapes of selfRef, end, and weight meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). The shape after broadcasting is the same as that of `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `weight` (aclTensor\*, compute input): `weight` in the formula, aclTensor on the device. The data types of selfRef, end, weight, and out are the same. The shapes of selfRef, end, and weight meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef, end, or weight is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef, end, or weight is not supported.
                                        2. The data types of selfRef, end, and weight are inconsistent.
                                        3. Broadcasting cannot be performed for selfRef, end, and weight. 
                                        4. The shapes of selfRef, end, and weight after broadcasting are different from that of selfRef.
  ```

## aclnnInplaceLerp

- **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnInplaceLerpGetWorkspaceSize.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLerp&aclnnInplaceLerp defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

aclnnLerp

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lerp_tensor.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // (Boilerplate) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> endShape = {4, 2};
    std::vector<int64_t> weightShape = {1};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* endDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* end = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> endHostData = {4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> weightHostData = {2};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an end aclTensor.
    ret = CreateAclTensor(endHostData, endShape, &endDeviceAddr, aclDataType::ACL_FLOAT, &end);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnLerp.
    ret = aclnnLerpGetWorkspaceSize(self, end, weight, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLerpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnLerp.
    ret = aclnnLerp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLerp failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar.
    aclDestroyTensor(self);
    aclDestroyTensor(end);
    aclDestroyTensor(weight);
    aclDestroyTensor(out);
    
    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(endDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

aclnnInplaceLerp

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lerp_tensor.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // (Boilerplate) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> endShape = {4, 2};
    std::vector<int64_t> weightShape = {1};
    void* selfDeviceAddr = nullptr;
    void* endDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* end = nullptr;
    aclTensor* weight = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> endHostData = {4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> weightHostData = {2};
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an end aclTensor.
    ret = CreateAclTensor(endHostData, endShape, &endDeviceAddr, aclDataType::ACL_FLOAT, &end);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    
    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnInplaceLerp.
    ret = aclnnInplaceLerpGetWorkspaceSize(self, end, weight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLerpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnInplaceLerp.
    ret = aclnnInplaceLerp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLerp failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host.
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar.
    aclDestroyTensor(self);
    aclDestroyTensor(end);
    aclDestroyTensor(weight);
    
    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(endDeviceAddr);
    aclrtFree(weightDeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
