# aclnnPowTensorScalar&aclnnInplacePowTensorScalar

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
out_i = self_i^{exponent_i}
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

- aclnnPowTensorScalar and aclnnInplacePowTensorScalar implement the same function in different ways. Select a proper operator based on your requirements.
  - aclnnPowTensorScalar: An output tensor object needs to be created to store the computation result.
  - aclnnInplacePowTensorScalar: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnPowTensorScalarGetWorkspaceSize or aclnnInplacePowTensorScalarGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnPowTensorScalar or aclnnInplacePowTensorScalar is called to perform computation.
  - `aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* exponent, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnPowTensorScalar(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* exponent, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplacePowTensorScalar(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, aclrtStream stream)`

## aclnnPowTensorScalarGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compution input): `self` in the formula, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The shape must be the same as that of `out`. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, INT16, COMPLEX64, COMPLEX128, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `exponent`.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, INT16, COMPLEX64, or COMPLEX128, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `exponent`.
  - `exponent` (aclScalar\*, compution input): `exponent` in the formula, aclScalar on the device. The data type cannot be BOOL at the same time as `self`. If the data type of `self` and `exponent` is integer after deduction, the value of `exponent` must be greater than or equal to 0. The value of `exponent` must be within the value range of the data type after deduction of `self` and `exponent`.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, INT16, COMPLEX64, COMPLEX128, or BFLOAT16, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `self`.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, INT16, COMPLEX64, or COMPLEX128, and must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `self`.
  - `out` (aclTensor\*, computation output): `out` in the formula, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The shape must be the same as that of self. The data type must be convertible from that after deduction between self and exponent (for details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The [data format](../../../docs/en/context/data_format.md) supports ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, INT16, COMPLEX64, COMPLEX128, and BFLOAT16.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, BOOL, INT8, UINT8, INT16, COMPLEX64, and COMPLEX128.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, exponent, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, exponent, or out is not supported.
                                       2. The shape of self exceeds eight dimensions.
                                       3. self and exponent do not meet the type deduction rules.
                                       4. The deduced data type cannot be converted to the type of out.
                                       5. The shapes of self and out are inconsistent.
                                       6. The data type of self and exponent is integer after deduction, and the value of exponent is less than 0.
                                       7. The value of exponent is beyond the value range of the data type after deduction of self and exponent.
  ```

## aclnnPowTensorScalar

- **Parameters:**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnPowTensorScalarGetWorkspaceSize.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplacePowTensorScalarGetWorkspaceSize

- **Parameters:**

  - `selfRef`(aclTensor\*): input `self/out` in the formula, aclTensor on the device. The data type must be convertible from that after deduction with that of `exponent` (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, and BFLOAT16, and the data type must satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with that of `exponent`.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, and INT16, and the data type must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `exponent`.
  - `exponent` (aclScalar\*, compution input): input `exponent` in the formula, `aclScalar` on the device. If the data type of `selfRef` and `exponent` is integer after deduction, the value of `exponent` must be greater than or equal to 0. The value of `exponent` must be within the value range of the data type after deduction of `selfRef` and `exponent`.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, and BFLOAT16, and the data type must satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with that of `selfRef`.
    * <term>Atlas training products </term>, <term>Atlas 200I/500 A2 inference products </term>, and <term>Atlas inference products </term>: The supported data types are FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, and INT16, and the data type must meet the [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `selfRef`.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. `selfRef` or `exponent` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or exponent is not supported.
                                       2. The shape of selfRef exceeds eight dimensions.
                                       3. selfRef and exponent cannot meet the type deduction rules.
                                       4. The deduced data type cannot be converted to the type of selfRef.
                                       5. The data type of selfRef and exponent is integer after deduction, and the value of exponent is less than 0.
                                       6. The value of exponent is beyond the value range of the data type after deduction of self and exponent.
  ```

## aclnnInplacePowTensorScalar

- **Parameters:**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnInplacePowTensorScalarGetWorkspaceSize.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPowTensorScalar and aclnnInplacePowTensorScalar default to deterministic implementation.

<term>Atlas training products</term> and <term>Atlas inference products</term>: If the computation result exceeds the value range of the specified data type, the boundary value of the data type is returned as the result.

In the exponent = 2 scenario, when the `square` operator is called and the input `self` is `int8`, the accuracy is ensured only when the result is within the range of (-2048, 1920).

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"

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
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* exponent = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float exponentVal = 4.1f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a threshold aclScalar.
  exponent = aclCreateScalar(&exponentVal, aclDataType::ACL_FLOAT);
  CHECK_RET(exponent != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Change the API name to the actual one.
  // aclnnPowTensorScalar API call example
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPowTensorScalar.
  ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPowTensorScalar.
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalar failed. ERROR: %d\n", ret); return ret);

  // aclnnInplacePowTensorScalar API call example
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplacePowTensorScalar.
  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplacePowTensorScalar.
  ret = aclnnInplacePowTensorScalar(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalar failed. ERROR: %d\n", ret); return ret);

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
    LOG_PRINT("aclnnPowTensorScalar result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), outDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplacePowTensorScalar result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(exponent);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (inplaceWorkspaceSize > 0) {
    aclrtFree(inplaceWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
