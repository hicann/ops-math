# aclnnLeTensor & aclnnInplaceLeTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/less_equal)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Compare each element of `self` with the corresponding one of `other` to check if the former is less than or equal to the latter, and write each comparison result to `out`.
- Formula:

  $$
  out_i = (self_i <= other_i) ? [True] : [False]
  $$

## Prototype

- `aclnnLeTensor` and `aclnnInplaceLeTensor` implement the same function in different ways. Select a proper operator based on your requirements.

  - `aclnnLeTensor`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceLeTensor`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLeTensorGetWorkspaceSize` or `aclnnInplaceLeTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLeTensor` or `aclnnInplaceLeTensor` is called to perform computation.
  - `aclnnStatus aclnnLeTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnLeTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceLeTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceLeTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLeTensorGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, computation input): input `self` in the formula, which is an aclTensor on the device. The data type must meet the data type deduction rules with `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, BFLOAT16, FLOAT, DOUBLE, or BOOL.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, FLOAT, DOUBLE, or BOOL.
  - `other` (aclTensor*, computation input): input `other` in the formula, which is an aclTensor on the device. The data type must meet the data type deduction rules with `self` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`. The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, BFLOAT16, FLOAT, DOUBLE, or BOOL.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, FLOAT, DOUBLE, or BOOL.
  - `out` (aclTensor\*, computation output): out in the formula, aclTensor on the device. The data type must be convertible to BOOL (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape must be the same as the shape after `self` and `other` are broadcast (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). The shape dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>;: The data type can be FLOAT, INT32, INT64, FLOAT16, INT16, INT8, UINT8, DOUBLE, UINT32, UINT64, BOOL, UINT16, COMPLEX64, COMPLEX128 or BFLOAT16.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, INT32, INT64, FLOAT16, INT16, INT8, UINT8, DOUBLE, UINT32, UINT64, BOOL, UINT16, COMPLEX64 or COMPLEX128.
  - `workspaceSize` (output): size of the workspace to be allocated on the device.
  - `executor` (output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, other, or out is not supported.
                                    2. The dimensions of self, other, or out are greater than 8.
                                    3. Data type deduction cannot be performed for self and other.
                                    4. Broadcasting cannot be performed for the shapes of self and other.
                                    5. The shape of `out` is inconsistent with the broadcasted shape.
  ```

## aclnnLeTensor

- **Parameters**

  - `workspace` (void \*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnLeTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLeTensorGetWorkspaceSize

- **Parameters**

  * `selfRef` (aclTensor*, computation input | computation output): input and output tensor, that is, `self` and `out` in the formula. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`. The shape after broadcasting must be the same as that of `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, BFLOAT16, FLOAT, DOUBLE, or BOOL.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, FLOAT, DOUBLE, or BOOL.
  * `other` (aclTensor*, computation input): The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `selfRef`. The shape after broadcasting must be the same as that of `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, BFLOAT16, FLOAT, DOUBLE, or BOOL.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, UINT16, FLOAT, DOUBLE, or BOOL.
  * `workspaceSize` (output): size of the workspace to be allocated on the device.
  * `executor` (output): operator executor, containing the operator computation process.

- **Returns**

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

## aclnnInplaceLeTensor

- **Parameters**

  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * workspaceSize (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceLeTensorGetWorkspaceSize`.
  * executor (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnLeTensor` & `aclnnInplaceLeTensor` default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnLeTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_le_tensor.h"

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

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
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLeTensor.
  ret = aclnnLeTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnLeTensor.
  ret = aclnnLeTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLeTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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

  // 6. Destroy aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
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

**Sample code of aclnnInplaceLeTensor:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_le_tensor.h"

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
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> otherHostData = {1, 1, 1, 1, 0, 0, 0, 0};

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT32, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceLeTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnInplaceLeTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLeTensor failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // Destroy aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);

  // Release device resources. Modify the configuration based on the API definition.
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
