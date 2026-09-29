# aclnnFmodScalar&aclnnInplaceFmodScalar

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/mod)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Returns the remainder of `self` divided by `other`.

- Formula:

  For the input parameter `self` and the comparison scalar `other`, `Fmod` can be expressed by the following formula:

  $$
  out_{i} = self_{i} - (other *\left \lfloor (self_{i}/other) \right \rfloor)
  $$

## Prototype

- `aclnnFmodScalar` and `aclnnInplaceFmodScalar` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnFmodScalar`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceFmodScalar`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFmodScalarGetWorkspaceSize` or `aclnnInplaceFmodScalarGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFmodScalar` or `aclnnInplaceFmodScalar` is called to perform computation.

  - `aclnnStatus aclnnFmodScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnFmodScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceFmodScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnFmodScalarGetWorkspaceSize

- **Parameters:**

  * `self` (aclTensor*, compute input): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape cannot be greater than 8D.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. The deduced data type must be within the supported data type range.
     * <term>Atlas training products </term>: The data type can be DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. The deduced data type must be within the supported data type range.
  * `other` (aclScalar*, compute input): aclScalar on the host. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`. The deduced data type must be within the supported data type range.
     * <term>Atlas training products </term>: The data type can be DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`. The deduced data type must be within the supported data type range.
  * `out` (aclTensor \*, compute output): aclTensor on the device. The data type must be convertible from the deduced data type of `self` and `other`. The shape must be the same as that of `self`. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape cannot be greater than 8D.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8.
     * <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API performs input parameter validation. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or shape of self or other is not supported.
                                    2. self or other do not meet the data type deduction rules.
                                    3. The deduced data type or shape cannot be converted to the data type or shape of out.
                                    4. The shapes of self and out are inconsistent.
  ```

## aclnnFmodScalar

- **Parameters:**

  - `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnFmodScalarGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceFmodScalarGetWorkspaceSize

- **Parameters:**

  * `selfRef` (aclTensor \*, compute input | compute output): input and output tensor, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape cannot be greater than 8D.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. The deduced data type must be convertible (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).
     * <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. The deduced data type must be convertible (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).
  * `other` (aclScalar*, compute input): aclScalar on the host. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`.
     * <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, or UINT8, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API performs input parameter validation. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or shape of selfRef or other is not supported.
                                    2. selfRef or other do not meet the data type deduction rules.
                                    3. The deduced data type cannot be converted to the type of selfRef.
  ```

## aclnnInplaceFmodScalar

- **Parameters:**

  - `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceFmodScalarGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnFmodScalar` and `aclnnInplaceFmodScalar` default to a deterministic implementation.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnFmodScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fmod_scalar.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
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
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Perform the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* other = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float otherValue = 1.0f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnFmodScalar.
  ret = aclnnFmodScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFmodScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnFmodScalar.
  ret = aclnnFmodScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFmodScalar failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. Free device resources.
  aclrtFree(selfDeviceAddr);
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

**aclnnInplaceFmodScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fmod_scalar.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
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
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Perform the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  float otherValue = 1.0f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceFmodScalar.
  ret = aclnnInplaceFmodScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFmodScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnInplaceFmodScalar.
  ret = aclnnInplaceFmodScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFmodScalar failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);

  // 7. Free device resources.
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
