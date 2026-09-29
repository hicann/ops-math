# aclnnGtTensor&aclnnInplaceGtTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/greater)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Determines whether the elements in the `self` tensor are greater than those in the `other` tensor. Returns a tensor. If `self` > `other`, the value is `True` (`1`); otherwise, the value is `False` (`0`).
- Formula:

  $$
  out = (self_i > other_i)  ?  [True] : [False]
  $$

## Prototype

- `aclnnGtTensor` and `aclnnInplaceGtTensor` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnGtTensor`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceGtTensor`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnGtTensorGetWorkspaceSize` or `aclnnInplaceGtTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGtTensor` or `aclnnInplaceGtTensor` is called to perform computation.

  - `aclnnStatus aclnnGtTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnGtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceGtTensorGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceGtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGtTensorGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): aclTensor on the device. The data type must meet the data type deduction rules with `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, or BFLOAT16 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL is supported.
  - `other` (aclTensor*, computation input): aclTensor on the device. The data type must meet the data type deduction rules with `self` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, or BFLOAT16 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL is supported.
  - `out` (aclTensor\*, computation output): aclTensor on the device. The data type must be convertible to BOOL (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape must be the same as the shape after `self` and `other` are broadcast (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, BFLOAT16, COMPLEX64, or COMPLEX128 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, COMPLEX64, or COMPLEX128 is supported.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, other, or out is not supported.
                                   2. The dimension of self, other, or out is greater than 8.
                                   3. Data type deduction cannot be performed for self and other.
                                   4. Broadcasting cannot be performed for the shapes of self and other.
                                   5. The shape of out before and after broadcasting is inconsistent.
  ```

## aclnnGtTensor

- **Parameters:**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnGtTensorGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceGtTensorGetWorkspaceSize

- **Parameters:**

  * `selfRef` (aclTensor \*, computation input | computation output): input and output tensor, `self` and `out` in the formula, aclTensor on the device. The shapes of `selfRef` and `other` must meet the broadcast relationship. The shape after broadcasting is the same as that of `selfRef`. The data types of `selfRef` and `other` must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, or BFLOAT16 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL is supported.
  * `other` (aclTensor*, computation input): aclTensor on the device. The shapes of `selfRef` and `other` must meet the broadcast relationship. The shape after broadcasting is the same as that of `selfRef`. The data types of `selfRef` and `other` must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, or BFLOAT16 is supported.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, or BOOL is supported.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or other is not supported.
                                   2. Data type deduction cannot be performed for selfRef and other.
                                   3. Broadcasting cannot be performed for the shapes of selfRef and other.
                                   4. The shapes of selfRef and other after broadcasting are different from that of selfRef.
                                   5. The dimension of selfRef or other is greater than 8.
  ```

## aclnnInplaceGtTensor

- **Parameters:**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceGtTensorGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnGtTensor` and `aclnnInplaceGtTensor` default to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**Sample code of aclnnGtTensor:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gt_tensor.h"

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

struct GtTensorData {
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
  std::vector<double> otherHostData = {0, 1, 1, 2, 3, 4, 5, 6};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
};

int PrepareAndExecuteGtTensor(aclrtStream stream, GtTensorData& data, void*& workspaceAddr, uint64_t& workspaceSize) {
  auto ret = 0;
  // Create a self aclTensor.
  ret = CreateAclTensor(data.selfHostData, data.selfShape, &data.selfDeviceAddr, aclDataType::ACL_DOUBLE, &data.self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(data.otherHostData, data.otherShape, &data.otherDeviceAddr, aclDataType::ACL_DOUBLE, &data.other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(data.outHostData, data.outShape, &data.outDeviceAddr, aclDataType::ACL_BOOL, &data.out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Call the CANN operator library API.
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGtTensor.
  ret = aclnnGtTensorGetWorkspaceSize(data.self, data.other, data.out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGtTensor.
  ret = aclnnGtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGtTensor failed. ERROR: %d\n", ret); return ret);
  
  return 0;
}

int HandleGtTensorResult(aclrtStream stream, const GtTensorData& data, void* workspaceAddr, uint64_t workspaceSize) {
  auto ret = 0;
  // Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // Obtain the output value and copy the result from the device memory to the host. 
  auto size = GetShapeSize(data.outShape);
  std::vector<char> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), data.outDeviceAddr, size * sizeof(char),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // Release aclTensor and aclScalar.
  aclDestroyTensor(data.self);
  aclDestroyTensor(data.other);
  aclDestroyTensor(data.out);

  // Release device resources.
  aclrtFree(data.selfDeviceAddr);
  aclrtFree(data.otherDeviceAddr);
  aclrtFree(data.outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  
  return 0;
}

int ExecuteGtTensorOperator(aclrtStream stream) {
  GtTensorData data;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;

  auto ret = PrepareAndExecuteGtTensor(stream, data, workspaceAddr, workspaceSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = HandleGtTensorResult(stream, data, workspaceAddr, workspaceSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return 0;
}

int main() {
  // Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Perform the GtScalar operation.
  ret = ExecuteGtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteGtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and terminate the ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**Sample code of aclnnInplaceGtTensor:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gt_tensor.h"

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
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {0, 1, 1, 2, 3, 4, 5, 6};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_DOUBLE, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceGtTensor.
  ret = aclnnInplaceGtTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceGtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnInplaceGtTensor.
  ret = aclnnInplaceGtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceGtTensor failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
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
