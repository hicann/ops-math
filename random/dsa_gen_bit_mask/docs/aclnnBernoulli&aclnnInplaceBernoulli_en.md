# aclnnBernoulli&aclnnInplaceBernoulli

[📄 View Source Code](https://gitcode.com/cann/ops-math/tree/master/random/dsa_gen_bit_mask)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Draws binary random numbers (`0` or `1`) from a Bernoulli distribution. `prob` indicates the probability of generating `1`. The input tensor is used to specify the shape.

- Formula:

  $$
  out∼Bernoulli(prob)
  $$

   When `aclnnBernoulli` is used, `prob` and `out` in the formula correspond to those in the prototype of the first-phase API. When `aclnnInplaceBernoulli` is used, `prob` and `out` in the formula correspond to `prob` and `selfRef` respectively in the prototype of the first-phase API.

## Prototype

- `aclnnBernoulli` and `aclnnInplaceBernoulli` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnBernoulli`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceBernoulli`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBernoulliGetWorkspaceSize` or `aclnnInplaceBernoulliGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnBernoulli` or `aclnnInplaceBernoulli` is called to perform computation.

  - `aclnnStatus aclnnBernoulliGetWorkspaceSize(const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(const aclTensor* selfRef, const aclScalar* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnInplaceBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBernoulliGetWorkspaceSize

- **Parameters:**
  - `self` (aclTensor*, computation input): which specifies the shape of `out` and is an aclTensor on the device. The data type must be the same as that of `out`. The shape supports 0 to 8 dimensions and must be the same as that of `out`. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, or BFLOAT16.
  - `prob` (aclScalar*, computation input): `prob` in the formula, aclScalar on the host, which must meet the condition: $0 ≤ prob ≤ 1 $.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, or BFLOAT16.
  - `seed` (int64_t, computation input): integer on the host, used to set the seed of the random number generator.
  - `offset` (int64_t, computation input): integer on the host, used to set the random number offset.
  - `out` (aclTensor*, computation output): `out` in the formula, aclTensor on the device. The data type must be the same as that of `self`. The shape supports 0 to 8 dimensions and must be the same as that of `self`. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, or BFLOAT16.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, covering the operator computation process.

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```text
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input self, prob, or out is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or out is not supported.
                                          2. The data type of prob is not supported.
                                          3. The data types of self and out are inconsistent.
                                          4. prob does not meet the condition: 0 ≤ prob ≤ 1.
                                          5. self or out has more than 8 dimensions.
                                          6. The shapes of self and out are inconsistent.
    ```

## aclnnBernoulli

- **Parameters:**
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnBernoulliGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceBernoulliGetWorkspaceSize

- **Parameters:**
  - `selfRef` (aclTensor*, computation input/output): `out` in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, or BFLOAT16.
  - `prob` (aclScalar*, computation input): `prob` in the formula, aclScalar on the host, which must meet the condition: $0 ≤ prob ≤ 1 $.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, or BFLOAT16.
  - `seed` (int64_t, computation input): integer on the host, used to set the seed of the random number generator.
  - `offset` (int64_t, computation input): integer on the host, used to set the random number offset.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, covering the operator computation process.

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```text
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input selfRef or prob is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of selfRef is not supported.
                                          2. The data type of prob is not supported.
                                          3. prob does not meet the condition: 0 ≤ prob ≤ 1.
                                          4. selfRef has more than 8 dimensions.
    ```

## aclnnInplaceBernoulli

- **Parameters:**
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnInplaceBernoulliGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnBernoulli` and `aclnnInplaceBernoulli` each default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

aclnnBernoulli

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bernoulli.h"

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
  // Customize the error handling as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* prob = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData(16, 0);
  int64_t seed = 0;
  int64_t offset = 0;
  float probValue = 1.0f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a prob aclScalar.
  prob = aclCreateScalar(&probValue,aclDataType::ACL_FLOAT);
  CHECK_RET(prob != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBernoulli API call example
  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnBernoulli.
  ret = aclnnBernoulliGetWorkspaceSize(self, prob, seed, offset, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBernoulliGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBernoulli.
  ret = aclnnBernoulli(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBernoulli failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(prob);
  aclDestroyTensor(out);

  // 7. Free device resources. Modify the code based on the API definition.
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

aclnnInplaceBernoulli

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bernoulli.h"

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
  // Customize the error handling as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 4};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* prob = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int64_t seed = 0;
  int64_t offset = 0;
  float probValue = 1.0f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a prob aclScalar.
  prob = aclCreateScalar(&probValue,aclDataType::ACL_FLOAT);
  CHECK_RET(prob != nullptr, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnInplaceBernoulli API call example
  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnInplaceBernoulli.
  ret = aclnnInplaceBernoulliGetWorkspaceSize(self, prob, seed, offset, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBernoulliGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceBernoulli.
  ret = aclnnInplaceBernoulli(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBernoulli failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(prob);

  // 7. Free device resources. Modify the code based on the API definition.
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
