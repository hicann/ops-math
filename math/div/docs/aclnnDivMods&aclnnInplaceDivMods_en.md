# aclnnDivMods&aclnnInplaceDivMods

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/div)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Performs division and rounds the result based on the `mode` parameter.
- Formula:

$$
out_i = \frac{input_i}{other}
$$

## Prototype

- `aclnnDivMods` and `aclnnInplaceDivMods` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnDivMods`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceDivMods`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnDivModsGetWorkspaceSize` or `aclnnInplaceDivModsGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnDivMods` or `aclnnInplaceDivMods` is called to perform computation.
  - `aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, int mode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, int mode, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnDivModsGetWorkspaceSize

- **Parameters**
  
  * `self` (aclTensor*, computation input): dividend, `input` in the formula, `aclTensor` on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
     * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
  * `other` (aclScalar*, computation input): divisor, `other` in the formula, `aclScalar` on the device.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`.
     * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`.
  * `mode` (int, computation input): rounding mode of the quotient, input `mode` in the formula. It is an integer on the host. The valid values are:<br>`0` (corresponding to None): Rounding is not performed by default.<br>`1` (corresponding to `trunc`): The fractional part of the quotient is rounded to zero.<br>`2` (corresponding to `floor`): The quotient is rounded down.    
  * `out` (aclTensor\*, computation output): quotient, `out` in the formula, `aclTensor` on the device. The data type must be convertible from the deduced data type of `self` and `other`. The shape must be identical to that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16.
     * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or other is not supported.
                                    2. self and other do not meet the type deduction rules.
                                    3. The deduced data type cannot be converted to that of out.
                                    4. The number of dimensions of self or other exceeds 8.
                                    5. The value of mode is not 0, 1, or 2.
                                    6. When mode is 1 or 2, the deduced data type of self and other is COMPLEX64 or COMPLEX128.
  ```

## aclnnDivMods

- **Parameters**
  
  * `workspace` (void\*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnDivModsGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceDivModsGetWorkspaceSize

- **Parameters**
  
  * `selfRef` (aclTensor\*, computation input | computation output): dividend and quotient, `input` and `out` in the formula, `aclTensor` on the device. The data type must be convertible from the deduced data type of `selfRef` and `other` (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
     * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
  * `other` (aclScalar*, computation input): `other` in the formula, `aclScalar` on the device.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, or BFLOAT16, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`.
     * <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, or COMPLEX64, and must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`.
  * `mode` (int, computation input): rounding mode of the quotient, input `mode` in the formula. It is an integer on the host. The valid values are:<br>`0` (corresponding to None): Rounding is not performed by default.<br>`1` (corresponding to `trunc`): The fractional part of the quotient is rounded to zero.<br>`2` (corresponding to `floor`): The quotient is rounded down.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef, other, or mode is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of selfRef or other is not supported.
                                    2. selfRef and other do not meet the type deduction rules.
                                    3. The number of dimensions of selfRef or other exceeds 8.
                                    4. The value of mode is not 0, 1, or 2.
                                    5. When mode is 1 or 2, the deduced data type of selfRef and other is COMPLEX64 or COMPLEX128.
  ```

## aclnnInplaceDivMods

- **Parameters**
  * `workspace` (void\*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceDivModsGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnDivMods` and `aclnnInplaceDivMods` default to a deterministic implementation.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: When `mode` is set to `1` (`trunc` mode) or `2` (`floor` mode), the limited precision of FLOAT16/BFLOAT16 may result in errors during rounding or rounding down because not all fractional parts can be represented. You are advised to use data types with higher precision, such as FLOAT32.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_div.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID based on the actual device.
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
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  float otherValue = 2.0f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int mode = 2;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnDivMods API call example
  LOG_PRINT("test aclnnDivMods\n");

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // Call the first-phase API of aclnnDivMods.
  ret = aclnnDivModsGetWorkspaceSize(self, other, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivModsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDivMods.
  ret = aclnnDivMods(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivMods failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]), outDeviceAddr,
                    8 * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 8; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, outHostData[i]);
  }

  // aclnnInplaceDivMods API call example
  LOG_PRINT("\ntest aclnnInplaceDivMods\n");

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // Call the first-phase API of aclnnInplaceDivMods.
  ret = aclnnInplaceDivModsGetWorkspaceSize(self, other, mode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivModsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceDivMods.
  ret = aclnnInplaceDivMods(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivMods failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
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
