# aclnnSubs & aclnnInplaceSubs

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/sub)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

+ Operator function: Performs subtraction and scales the subtrahend by alpha.
+ Formula:

  $$
  out_{i} = self_{i} - alpha \times other
  $$

  $$
  selfRef_{i} = selfRef_{i} - alpha \times other
  $$

## Prototype

* `aclnnSubs` and `aclnnInplaceSubs` implement the same function in different ways. Select a proper operator based on your requirements.
  * `aclnnSubs`: An output tensor object needs to be created to store the computation result.
  * `aclnnInplaceSubs`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
* Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSubsGetWorkspaceSize` or `aclnnInplaceSubsGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the computation process. Then, `aclnnSubs` or `aclnnInplaceSubs` is called to perform computation.
  * `aclnnStatus aclnnSubsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnSubs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceSubsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceSubs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSubsGetWorkspaceSize

* **Parameter Description**:
  * `self` (aclTensor*, computation input): input `self` in the formula. The shape dimension cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`.

  * `other` (aclScalar*, computation input): input `other` in the formula.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `self`.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `self`.

  `alpha` (aclScalar*, computation input): `alpha` in the formula. The data type must be convertible to that after deduction between `self` and `other`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.

  * `out` (aclTensor*, computation output): `out` in the formula. The data type must be convertible to that after deduction between `self` and `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
SS
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

* **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, alpha, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                        2. self and other do not meet the type deduction rules.
                                        3. The deduced data type cannot be converted to the data type of out.
                                        4. The data type of alpha cannot be converted to the type deduced from self and other.
                                        5. The shapes of self and out are inconsistent.
                                        6. The shape dimension of self or out is greater than 8.
  ```

## aclnnSubs

* **Parameter Description**:
  + `workspace` (void*, input): address of the workspace to be allocated on the device.
  + `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnSubsGetWorkspaceSize`.
  + `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  + `stream` (aclrtStream, input): stream for executing the task.

* **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceSubsGetWorkspaceSize

* **Parameter Description**:
  * `selfRef` (aclTensor*, computation input | computation output): input `selfRef` in the formula. The number of dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`. In addition, the data type must be convertible to that after deduction (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `other`. In addition, the data type must be convertible to that after deduction (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).

  * `other` (aclScalar*, computation input): input `other` in the formula. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `selfRef`. In addition, the data type must be convertible to that after deduction (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `selfRef`. In addition, the data type must be convertible to that after deduction (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).

  * `alpha` (aclScalar*, computation input): `alpha` in the formula. The data type must be convertible to that after deduction between `selfRef` and `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - <term>Atlas inference products</term>, <term>Atlas training products</term>, and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.

  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

* **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef, other, or alpha is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef is not supported.
                                        2. selfRef and other do not meet the type deduction rules.
                                        3. The deduced data type cannot be converted to the type of selfRef.
                                        4. The data type of alpha cannot be converted to the type deduced from selfRef and other.
  ```

## aclnnInplaceSubs

* **Parameter Description**:
  + `workspace` (void*, input): address of the workspace to be allocated on the device.
  + workspaceSize (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnInplaceSubsGetWorkspaceSize`.
  + `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  + `stream` (aclrtStream, input): stream for executing the task.

* **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computation:
  - `aclnnSubs` and `aclnnInplaceSubs` default to deterministic implementation.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sub.h"

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
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  float otherValue = 2.0f;
  float alphaValue = 1.2f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // Create an alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnSubs API call example
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSubs.
  ret = aclnnSubsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSubs.
  ret = aclnnSubs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubs failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceSubs API call example
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceSubs\n");
  // Call the first-phase API of aclnnInplaceSubs.
  ret = aclnnInplaceSubsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceSubs.
  ret = aclnnInplaceSubs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSubs failed. ERROR: %d\n", ret); return ret);

  // 4. Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Copy the result from the device memory to the host.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("%ld acos(%f) = %f\n", i, selfHostData[i], resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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
