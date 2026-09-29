# aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/floor_mod)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Returns the remainder of each element in tensor `self` divided by scalar `other`. The sign of the result is the same as that of the divisor `other`, and the absolute value of the result is less than that of `other`.

- The actual computation of `remainder(self, other)` is equivalent to the following formula:

  $$
  out_i = self_i - floor(self_i / other) * other
  $$

- Example:

  ```text
  self = tensor([[-1, -2],
                 [-3, -4]]).type(int32)
  other = 3.5   # float
  result = remainder(self, other)

  # Value of result
  # tensor([[2.5000, 1.5000],
  #         [0.5000, 3.0000]])   float

  # For -1 in self, the computation result is -1 - floor(-1 / 3.5) * 3.5 = 2.5.
  # The absolute value of the final result 2.5 is less than that of other, which is 3.5.
  ```

## Prototype

- `aclnnRemainderTensorScalar` and `aclnnInplaceRemainderTensorScalar` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnRemainderTensorScalar`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceRemainderTensorScalar`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRemainderTensorScalarGetWorkspaceSize` or `aclnnInplaceRemainderTensorScalarGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRemainderTensorScalar` or `aclnnInplaceRemainderTensorScalar` is called to perform computation.

  * `aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnRemainderTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceRemainderTensorScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceRemainderTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRemainderTensorScalarGetWorkspaceSize

- **Parameters:**

  `self` (aclTensor*, computation input): input `self` in the formula. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND, and the data dimensions cannot exceed 8.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, DOUBLE, or BFLOAT16. Its data type and the data type of `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - <term>Atlas training products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, or DOUBLE. Its data type and the data type of `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).

  * `other` (aclScalar*, computation input): input `other` in the formula.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, DOUBLE, or BFLOAT16. Its data type and the data type of `self` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - <term>Atlas training products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, or DOUBLE. Its data type and the data type of `self` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).

  * `out` (aclTensor*, computation output): output `out` in the formula. The data type must be convertible to that after deduction between `self` and `other`. The shape must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The data dimensions cannot be greater than 8.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.
    - <term>Atlas training products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.

  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.

  * `executor` (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The shapes of self and out are inconsistent.
                                    2. Data type deduction cannot be performed for self and other.
                                    3. The data type deduced from self and other is not supported.
                                    4. The data type deduced from self and other cannot be converted to the data type of out.
                                    5. The dimensions of self or out are greater than 8.
                                    6. The data formats of self and out are inconsistent.
  ```

## aclnnRemainderTensorScalar

- **Parameters:**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.

  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnRemainderTensorScalarGetWorkspaceSize`.

  * `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.

  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceRemainderTensorScalarGetWorkspaceSize

- **Parameters:**

  * `selfRef` (aclTensor*, computation input|computation output): input and output tensors. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The data dimensions cannot be greater than 8.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, DOUBLE, or BFLOAT16. Its data type and the data type of `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) and must be convertible after deduction.
    - <term>Atlas training products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, or DOUBLE. Its data type and the data type of `other` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) and must be convertible after deduction.
  * `other` (aclScalar*, computation input): input `other` in the formula.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, DOUBLE, or BFLOAT16. Its data type and the data type of `selfRef` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - <term>Atlas training products</term>: The data type can be INT32, INT64, FLOAT16, FLOAT, or DOUBLE. Its data type and the data type of `selfRef` must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type cannot be deduced from selfRef and other.
                                    2. The data type deduced from selfRef and other is not supported.
                                    3. The data type deduced from selfRef and other cannot be converted to the data type of selfRef.
                                    4. The dimensions of selfRef are greater than 8.
  ```

## aclnnInplaceRemainderTensorScalar

- **Parameters:**
  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceRemainderTensorScalarGetWorkspaceSize`.
  * `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computation:
  - `aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar` defaults to deterministic implementation.

  * When the data type of `self` is INT32, the functionality and precision within the range of [–2^24, 2^24] are preferentially ensured.
  * When `other` is 0 and the data type of `self` is an integer, the result of `out` is `self`.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
**aclnnRemainderTensorScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_remainder.h"

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
  // Set device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> outShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t Other = 3;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&Other, aclDataType::ACL_INT64);
  CHECK_RET(other != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRemainderTensorScalar.
  ret = aclnnRemainderTensorScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRemainderTensorScalar.
  ret = aclnnRemainderTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderTensorScalar failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. Release device resources.
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

**aclnnInplaceRemainderTensorScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_remainder.h"

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
  // Set device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfRefShape = {3, 3};
  void* selfRefDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclScalar* other = nullptr;
  std::vector<int64_t> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  int64_t Other = 3;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT64, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&Other, aclDataType::ACL_INT64);
  CHECK_RET(other != nullptr, return ret);

  // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceRemainderTensorScalar.
  ret = aclnnInplaceRemainderTensorScalarGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRemainderTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceRemainderTensorScalar.
  ret = aclnnInplaceRemainderTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRemainderTensorScalar failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyScalar(other);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
