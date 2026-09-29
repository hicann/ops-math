# aclnnLtScalar&aclnnInplaceLtScalar

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/less)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Compares each element of `self` with the corresponding one of `other` to check if the former is less than the latter, and returns a tensor of the Boolean type.

- Formula:

  $$
  out_i = (self_i < other_i)  ?  [True] : [False]
  $$

## Prototype

- `aclnnLtScalar` and `aclnnInplaceLtScalar` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnLtScalar`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceLtScalar`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLtScalarGetWorkspaceSize` or `aclnnInplaceLtScalarGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLtScalar` or `aclnnInplaceLtScalar` is called to perform computation.

  - `aclnnStatus aclnnLtScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnLtScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceLtScalarGetWorkspaceSize(const aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceLtScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLtScalarGetWorkspaceSize

- **Parameters:**

  * `self` (aclTensor*, compute input): aclTensor on the device. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`.
  * `other` (aclScalar*, compute input): aclScalar on the host.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `self`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `self`.
  * `out` (aclTensor \*, compute output): aclTensor on the device. The shape is the same as that of `self`. The data type must be convertible to BOOL (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, COMPLEX64 or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, COMPLEX64 or COMPLEX128.
  * `workspaceSize` (uint64_t \*, output parameter): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, other, or out is not supported.
                                    2. The data types of self and other do not meet the type deduction rules.
                                    3. The shape of self is different from that of out.
                                    4. The dimensions of self and out are greater than 8.
  ```

## aclnnLtScalar

- **Parameters:**

  * `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLtScalarGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLtScalarGetWorkspaceSize

- **Parameters:**

  - `selfRef` (aclTensor*, compute input | compute output): input and output tensor, that is, `self` and `out` in the formula. aclTensor on the device. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, or BOOL, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`.
  - `other` (aclScalar*, compute input): aclScalar on the host.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, or BOOL, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `selfRef`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `selfRef`.
  - `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or other is not supported.
                                   2. The data types of selfRef and other do not meet the data type deduction rules.
                                   3. The dimensions of selfRef are greater than 8.
  ```

## aclnnInplaceLtScalar

- **Parameters:**

  * `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceLtScalarGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - `aclnnLtScalar` and `aclnnInplaceLtScalar` default to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnLtScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_scalar.h"

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

int ExecuteLtScalarOperator(aclrtStream stream) {
  auto ret = 0;

  // Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  double otherValue = 1.2;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(other != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLtScalar.
  ret = aclnnLtScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnLtScalar.
  ret = aclnnLtScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtScalar failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modification is required based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modification is required based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. Release device resources. Modification is required based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // Perform the InplaceLtScalar operation.
  ret = ExecuteLtScalarOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and terminate the ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**aclnnInplaceLtScalar sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_scalar.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
  if (!(cond)) {                     \
    return_expr;                     \
  }                                  \
 } while(0)

#define LOG_PRINT(message, ...)   \
 do {                             \
  printf(message, ##__VA_ARGS__); \
 } while(0)

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

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
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

int ExecuteInplaceLtScalarOperator(aclrtStream stream) {
  auto ret = 0;
  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 1.2, 0.3, 4.1, 5, 1.6, 7};
  double otherValue = 1.2;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(other != nullptr, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceLtScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnInplaceLtScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtScalar failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  aclDestroyTensor(self);
  aclDestroyScalar(other);

  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // Perform the InplaceLtScalar operation.
  ret = ExecuteInplaceLtScalarOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and terminate the ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
