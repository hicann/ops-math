# aclnnPowTensorTensor&aclnnInplacePowTensorTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/pow)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Operator function: Uses each element of `exponent` as the power of the corresponding element of `input` to complete the computation.

- Formula:

  $$
  out_i = x_i^{exponent_i}
  $$
  
- Operator restrictions: If INT32 computation is performed beyond the following ranges, timeout occurs.

  | shape  | exponent_value|
  |----|----|
  |≤ 100000 (100 thousand)|–200000000 to +200000000 (200 million)|
  |≤ 1000000 (1 million)|–20000000 to +20000000 (20 million)|
  |≤ 10000000 (10 million)|–2000000 to +2000000 (2 million)|
  |≤ 100000000 (100 million)|–200000 to +200000 (200 thousand)|
  |≤ 1000000000 (1 billion)|–20000 to +20000 (20 thousand)|

## Function Prototype

- aclnnPowTensorTensor and aclnnInplacePowTensorTensor implement the same function in different ways. Select a proper operator based on your requirements.

  - aclnnPowTensorTensor: An output tensor object needs to be created to store the computation result.
  - aclnnInplacePowTensorTensor: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnPowTensorTensorGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnPowTensorTensor is called to perform computation.

  - `aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent, aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnPowTensorTensor(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplacePowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplacePowTensorTensor(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, aclrtStream stream)`

## aclnnPowTensorTensorGetWorkspaceSize

- **Parameters**:

  - `self` (aclTensor*, computation input): aclTensor on the device. The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `exponent`. The shape can be 0 to 8 dimensions. The data types of `self` and `exponent` cannot be BOOL at the same time. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `exponent`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, and BFLOAT16.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, and COMPLEX128.

  - `exponent` (aclTensor*, computation input): aclTensor on the device. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`. The data types of `self` and `exponent` cannot be BOOL at the same time. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, and BFLOAT16.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, and COMPLEX128.

  - `out` (aclTensor*, computation output): aclTensor on the device. The data type must be convertible from that after deduction between `self` and `exponent`. The shape must be that after `self` and `exponent` are broadcast. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, and BFLOAT16.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, and COMPLEX128.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, exponent, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of self and exponent are not supported.
                                        2. Data type deduction cannot be performed for self and exponent.
                                        3. The deduced data type cannot be converted to the data type of out.
                                        4. Broadcasting cannot be performed for the shapes of self and exponent.
                                        5. self and exponent are both of the bool type.
  ```

## aclnnPowTensorTensor

- **Parameters**:

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPowTensorTensorGetWorkspaceSize.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplacePowTensorTensorGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input | computation output): input and output tensor, that is, `x` and `out` in the formula. The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with exponent. The data type must be convertible from that after deduction between self and exponent (for details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with exponent. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, and BFLOAT16.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, and INT16.
  - `exponent` (aclTensor*, computation input): input `exponent` in the formula. The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with self. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, and BFLOAT16.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, COMPLEX64, COMPLEX128, and INT16.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or exponent is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of self and exponent are not supported.
                                        2. The shape of self or exponent is greater than 8D.
                                        3. self and exponent do not meet the type deduction rules.
                                        4. Broadcasting cannot be performed for the shapes of self and other.
                                        5. The shape after broadcasting between self and exponent is not equal to the shape of self.
                                        6. self and exponent are both of the bool data type.
  ```

## aclnnInplacePowTensorTensor

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnInplacePowTensorTensorGetWorkspaceSize.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPowTensorTensor and aclnnInplacePowTensorTensor default to deterministic implementation.

<term>Atlas training products</term> and <term>Atlas inference products</term>: If the computation result exceeds the value range of the specified data type, the boundary value of the data type is returned as the result.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
**aclnnPowTensorTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the `deviceId` based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> expShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an exp aclTensor.
  ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPowTensorTensor.
  ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPowTensorTensor.
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
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

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(expDeviceAddr);
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

**aclnnInplacePowTensorTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> expShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* exp = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an exp aclTensor.
  ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplacePowTensorTensor.
  ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfRef, exp, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplacePowTensorTensor.
  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),         selfRefDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(exp);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(expDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
