# aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/bitwise_xor)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Computes the bitwise XOR between elements in the input tensor `self` and corresponding elements in the input tensor `other`. The input `self` and `other` must be of the INT or BOOL type. For the BOOL type, the operator computes the logical XOR.
- Formula:

  $$
  \text{out}_i =
  \text{self}_i \, \bigoplus\, \text{other}_i
  $$

## Prototype

- `aclnnBitwiseXorTensor` and `aclnnInplaceBitwiseXorTensor` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnBitwiseXorTensor`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceBitwiseXorTensor`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBitwiseXorTensorGetWorkspaceSize` or `aclnnInplaceBitwiseXorTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnBitwiseXorTensor` or `aclnnInplaceBitwiseXorTensor` is called to perform computation.

  - `aclnnStatus aclnnBitwiseXorTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnBitwiseXorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceBitwiseXorTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceBitwiseXorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBitwiseXorTensorGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): aclTensor on the device. Its data type can be BOOL, INT8, INT16, INT32, INT64, or UINT8. The data type must follow the deduction relationship with `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shapes of 'self' and `other` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND, and the data dimensions cannot exceed 8.
  - `other` (aclTensor*, computation input): aclTensor on the device. Its data type can be BOOL, INT8, INT16, INT32, INT64, or UINT8. The data type must follow the deduction relationship with `self` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shapes of 'other' and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND, and the data dimensions cannot exceed 8.
  - `out` (aclTensor *, computation output): aclTensor on the device. Its data type must be one convertible from the type deduced from `self` and `other`. The shape must be the shape obtained by broadcasting `self` and `other`. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND, and the data dimensions cannot exceed 8.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT, FLOAT16, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  - `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or other is not supported.
                                   2. self and other do not meet the deduction relationship.
                                   3. The data type deduced from self and other cannot be cast to the data type of out.
                                   4. The shapes of self and other are not broadcastable.
                                   5. The shape of out does not match the shape obtained by broadcasting self and other.
                                   6. self, other, and out each have more than 8 dimensions.
  ```

## aclnnBitwiseXorTensor

- **Parameters:**

  - `workspace` (void \*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnBitwiseXorTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceBitwiseXorTensorGetWorkspaceSize

- **Parameters:**

  - `selfRef` (aclTensor \*, computation input/output): input/output tensor, that is, `self` and `out` in the formula. It is an aclTensor on the device. Its data type can be BOOL, INT8, INT16, INT32, INT64, or UINT8. The data type must follow the deduction relationship with `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). A data type inferred from `selfRef` and `other` must be convertible to the data type of `selfRef` itself. The shapes of 'selfRef' and `other` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). A shape obtained through broadcasting must be the same as that of `selfRef`. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND, and the data dimensions cannot exceed 8.
  - `other` (aclTensor*, computation input): aclTensor on the device. Its data type can be BOOL, INT8, INT16, INT32, INT64, or UINT8. The data type must follow the deduction relationship with `selfRef` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shapes of 'other' and `selfRef` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND, and the data dimensions cannot exceed 8.
  - `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or other is not supported.
                                   2. selfRef and other do not meet the deduction relationship.
                                   3. The data type deduced from selfRef and other cannot be cast to the data type of selfRef.
                                   4. The shapes of selfRef and other are not broadcastable.
                                   5. The shape of selfRef is not the shape after broadcasting is performed on selfRef and other.
                                   6. selfRef and other each have more than 8 dimensions.
  ```

## aclnnInplaceBitwiseXorTensor

- **Parameters:**

  - `workspace` (void \*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnInplaceBitwiseXorTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnBitwiseXorTensor` and `aclnnInplaceBitwiseXorTensor` each default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnBitwiseXorTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bitwise_xor_tensor.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate code for resource initialization.
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

  // Call aclrtMemcpy to copy the data from the host to the device.
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
  // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> otherHostData = {0, 1, 1, 9, 3, 4, 5, 6};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnBitwiseXorTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBitwiseXorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API.
  ret = aclnnBitwiseXorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBitwiseXorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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

**aclnnInplaceBitwiseXorTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bitwise_xor_tensor.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate code for resource initialization.
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

  // Call aclrtMemcpy to copy the data from the host to the device.
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
  // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* other = nullptr;
  std::vector<int64_t> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> otherHostData = {0, 1, 1, 9, 3, 4, 5, 6};
  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT64, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnInplaceBitwiseXorTensorGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBitwiseXorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API.
  ret = aclnnInplaceBitwiseXorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBitwiseXorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(other);

  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
