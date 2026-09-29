# aclnnSub & aclnnInplaceSub

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/sub)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √     |
| <term>Atlas inference products</term>                            |   ×     |
| <term>Atlas training products</term>                             |   √     |

## Function

* Operator function: Performs subtraction and scales the subtrahend by alpha.
* Formula:
  
  $$
  out_{i} = self_{i} - alpha \times other_{i}
  $$
  
  $$
  selfRef_{i}=selfRef_{i}-alpha \times other_{i}
  $$

## Prototype

* `aclnnSub` and `aclnnInplaceSub` implement the same function in different ways. Select a proper operator based on your requirements.
  * `aclnnSub`: An output tensor object needs to be created to store the computation result.
  * `aclnnInplaceSub`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
* Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSubGetWorkspaceSize` or `aclnnInplaceSubGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the computation process. Then, `aclnnSub` or `aclnnInplaceSub` is called to perform computation.
  * `aclnnStatus aclnnSubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnSub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceSubGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceSub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSubGetWorkspaceSize

* **Parameter Description**:
  * `self` (aclTensor*, computation input): input `self` in the formula. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `other`. The number of dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
     * <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
  * `other` (aclTensor*, computation input): input `other` in the formula. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self`. The number of dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
     * <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
  * `alpha` (aclScalar*, computation input): `alpha` in the formula. The data type must be convertible to that after deduction between `self` and `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
  * `out` (aclTensor\*, computation output): `out` in the formula. The data type must be convertible to that after deduction between `self` and `other` (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape must be that after `self` and `other` are [broadcast](../../../docs/en/context/broadcast_relationship.md), and the dimension must not exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
  * `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

* **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, alpha, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or other is not supported.
                                        2. self and other do not meet the type deduction rules.
                                        3. The deduced data type cannot be converted to the data type of out.
                                        4. The data type of alpha cannot be converted to the type deduced from self and other.
                                        5. Broadcasting cannot be performed for the shapes of self and other.
                                        6. The shape of out is not the shape after self and other are broadcast.
                                        7. The shape dimension of self or other is greater than 8.
  ```

## aclnnSub

- **Parameter Description**:
  
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnSubGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceSubGetWorkspaceSize

* **Parameter Description**:
  + `selfRef` (aclTensor*, computation input/output): input `selfRef` in the formula. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. In addition, the data type must be convertible to that after deduction (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape of `selfRef` must be the same as that of `other` after [broadcast](../../../docs/en/context/broadcast_relationship.md), and the number of dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
     * <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
  + `other` (aclTensor*, computation input): input `other` in the formula. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `selfRef`. The number of dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
     * <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
  + `alpha` (aclScalar*, computation input): `alpha` in the formula. The data type must be convertible to that after deduction between `selfRef` and `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
    - <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
  + `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  + `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

* **Returns**:

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef, other, or alpha is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or other is not supported.
                                       2. selfRef and other do not meet the type deduction rules.
                                       3. The deduced data type cannot be converted to the type of selfRef.
                                       4. The data type of alpha cannot be converted to the type deduced from selfRef and other.
                                       5. Broadcasting cannot be performed for the shapes of selfRef and other.
                                       6. The shape after broadcasting is not the same as that of selfRef.
                                       7. The shape dimension of selfRef or other is greater than 8.
  ```

## aclnnInplaceSub

* **Parameter Description**:
  * `workspace` (void\*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceSubGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

* **Returns**:
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computation:
  - `aclnnSub` and `aclnnInplaceSub` default to deterministic implementation.

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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float alphaValue = 1.2f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // aclnnSub API call example
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSub.
  ret = aclnnSubGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSub.
  ret = aclnnSub(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSub failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceSub API call example
  // 3. Call the CANN operator library API.
  LOG_PRINT("\nTest aclnnInplaceSub\n");
  // Call the first-phase API of aclnnInplaceSub.
  ret = aclnnInplaceSubGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSubGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceSub.
  ret = aclnnInplaceSub(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSub failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  // 7. Release device resources. Modify the code based on the API definition.
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
