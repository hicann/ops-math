# aclnnLtTensor&aclnnInplaceLtTensor

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

- `aclnnLtTensor` and `aclnnInplaceLtTensor` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnLtTensor`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceLtTensor`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLtTensorGetWorkspaceSize` or `aclnnInplaceLtTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLtTensor` or `aclnnInplaceLtTensor` is called to perform computation.

  * `aclnnStatus aclnnLtTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnLtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceLtTensorGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceLtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLtTensorGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): aclTensor on the device. The data type must meet the data type deduction rules with `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, or BOOL.
  - `other` (aclTensor*, compute input): aclTensor on the device. The data type must meet the data type deduction rules with `self` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, or BOOL.
  - `out` (aclTensor \*, compute output): aclTensor on the device. The data type must be convertible to BOOL (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape must be the same as the shape after `self` and `other` are broadcast (see [broadcast relationship](../../../docs/en/context/broadcast_relationship.md)). The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64 or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64 or COMPLEX128.
  - `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, other, or out is not supported.
                                    2. The dimensions of self, other, or out are greater than 8.
                                    3. Data type deduction cannot be performed for self and other.
                                    4. Broadcasting cannot be performed for the shapes of self and other.
                                    5. The shape of out before and after broadcasting is inconsistent.
  ```

## aclnnLtTensor

- **Parameters:**

  * `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLtTensorGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLtTensorGetWorkspaceSize

- **Parameters:**

  - `selfRef` (aclTensor*, compute input | compute output): input and output tensor, that is, `self` and `out` in the formula. aclTensor on the device. The input data type must meet the data type deduction rules with `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`, and the shape after broadcasting must be the same as that of `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, or BFLOAT16.
  - `other` (aclTensor*, compute input): aclTensor on the device. The data type must meet the data type deduction rules with `selfRef` (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`. The shape after broadcasting must be the same as that of `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, or BFLOAT16.
  - `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or other is not supported.
                                   2. Data type deduction cannot be performed for selfRef and other.
                                   3. Broadcasting cannot be performed for the shapes of selfRef and other.
                                   4. The shapes of selfRef and other after broadcasting are different from that of selfRef.
                                   5. The dimensions of selfRef and other are greater than 8.
  ```

## aclnnInplaceLtTensor

- **Parameters:**

  * `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceLtTensorGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - `aclnnLtTensor` and `aclnnInplaceLtTensor` default to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnLtTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_tensor.h"

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


struct LtTensorData {
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {5, 5, 5, 5, 5, 5, 5, 5};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
};

int CreateInputAndOutputTensors(LtTensorData& data) {
  auto ret = 0;
  
  // Create a self aclTensor.
  ret = CreateAclTensor(data.selfHostData, data.selfShape, &data.selfDeviceAddr, aclDataType::ACL_DOUBLE, &data.self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(data.otherHostData, data.otherShape, &data.otherDeviceAddr, aclDataType::ACL_DOUBLE, &data.other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(data.outHostData, data.outShape, &data.outDeviceAddr, aclDataType::ACL_DOUBLE, &data.out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  return ret;
}

int ExecuteLtTensorComputation(aclrtStream stream, LtTensorData& data) {
  auto ret = 0;
  aclOpExecutor* executor;
  
  // Call the first-phase API of aclnnLtTensor.
  ret = aclnnLtTensorGetWorkspaceSize(data.self, data.other, data.out, &data.workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  data.workspaceAddr = nullptr;
  if (data.workspaceSize > 0) {
    ret = aclrtMalloc(&data.workspaceAddr, data.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnLtTensor.
  ret = aclnnLtTensor(data.workspaceAddr, data.workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtTensor failed. ERROR: %d\n", ret); return ret);
  
  // Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  return ret;
}

int ProcessAndPrintResults(const LtTensorData& data) {
  auto ret = 0;
  auto size = GetShapeSize(data.outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), data.outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }
  return ret;
}

void ReleaseResources(LtTensorData& data) {
  // Release aclTensor and aclScalar.
  aclDestroyTensor(data.self);
  aclDestroyTensor(data.other);
  aclDestroyTensor(data.out);

  // Release device resources.
  aclrtFree(data.selfDeviceAddr);
  aclrtFree(data.otherDeviceAddr);
  aclrtFree(data.outDeviceAddr);
  if (data.workspaceSize > 0) {
    aclrtFree(data.workspaceAddr);
  }
}

int ExecuteLtTensorOperator(aclrtStream stream) {
  LtTensorData data;
  
  // Create input and output tensors.
  auto ret = CreateInputAndOutputTensors(data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Perform the LtTensor operator operation.
  ret = ExecuteLtTensorComputation(stream, data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Process and print the result.
  ret = ProcessAndPrintResults(data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Release resources.
  ReleaseResources(data);
  
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // Perform the InplaceLtScalar operation.
  ret = ExecuteLtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and terminate the ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**aclnnInplaceLtTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_tensor.h"

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

int ExecuteInplaceLtTensorOperator(aclrtStream stream) {
  auto ret = 0;
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> otherHostData = {1, 1, 1, 1, 0, 0, 0, 0};

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT32, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceLtTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnInplaceLtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtTensor failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(other);

  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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
  ret = ExecuteInplaceLtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and terminate the ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
