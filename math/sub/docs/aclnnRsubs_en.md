# aclnnRsubs

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

- Description: Performs subtraction.
- Formula:

  $$
  out_i = other - self_i * alpha
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnRsubsGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRsubs` is called to perform computation.

- `aclnnStatus aclnnRsubsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRsubs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRsubsGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): input `self` in the formula. The shape dimension cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16. The data types of `self` and `other` must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). If `self` is FLOAT16, FLOAT32, BFLOAT16, or DOUBLE, the data type after deduction of `self` and `other` is the data type of `self`.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. If `self` is FLOAT16, FLOAT32, BFLOAT16, or DOUBLE, the data type after deduction of `self` and `other` is the data type of `self`.

  - `other` (aclScalar*, compute input): input `other` in the formula.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`. If the data type of `self` is not FLOAT16, FLOAT32, BFLOAT16, or DOUBLE and the data type of `other` is FLOAT16, FLOAT32, BFLOAT16, or DOUBLE, the data types of `self` and `other` must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). If the data types of `self` and `other` are not in FLOAT16, FLOAT32, BFLOAT16, or DOUBLE, the data type after derivation is that of `self`.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128. The data types of `self` and `other` must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). If the data type of `self` is not FLOAT16, FLOAT32, BFLOAT16, or DOUBLE and the data type of `other` is FLOAT16, FLOAT32, BFLOAT16, or DOUBLE, the data types of `self` and `other` must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). If the data types of `self` and `other` are not in FLOAT16, FLOAT32, BFLOAT16, or DOUBLE, the data type after derivation is that of `self`.

  - `alpha` (aclScalar*, input): `alpha` in the formula. The data type can be converted to the data type derived from `self` and `other`.
    - <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.

  - `out` (aclTensor*, compute output): `out` in the formula. The data type must be convertible from that after deduction of `self` and `other` (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, UINT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, alpha, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                    2. self and other do not meet the type deduction rules.
                                    3. The derived data type cannot be converted to the type of out.
                                    4. alpha cannot be converted to the data type deduced from self and other.
                                    5. The shapes of self and out are inconsistent.
                                    6. The dimensions of self and out are greater than 8.
  ```

## aclnnRsubs

- **Parameters:**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnRsubsGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnRsubs` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rsub.h"

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
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRsubs.
  ret = aclnnRsubsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRsubsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRsubs.
  ret = aclnnRsubs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRsubs failed. ERROR: %d\n", ret); return ret);
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
