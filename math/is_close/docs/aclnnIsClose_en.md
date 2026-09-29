# aclnnIsClose

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/is_close)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Returns a new tensor with Boolean elements and determines whether the given values of `self` and `other` are close to each other, and returns `True` if the values are close to each other and returns `False` otherwise.
- Formula:
  Returns a new tensor `out` of type `BOOL`, indicating whether each element of the input `self` is close to the corresponding element of the input `other`.
  Closeness is defined as:
  
  $$
  \left | self_{i}-other_{i}\right | \le  atol + rtol\times \left | other_{i} \right |
  $$
    
  When both `self` and `other` are finite values, the foregoing formula is true. When both `self` and `other` are non-finite, the result is `True` only if they are equal. When `equal_nan` is `True`, `NaNs` is considered to be closed. When `equal_nan` is `False`, `NaNs` is considered not to be closed.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnIsCloseGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnIsClose` is called to perform computation.

  - `aclnnStatus aclnnIsCloseGetWorkspaceSize(const aclTensor* self, const aclTensor* other, double rtol, double atol, bool equal_nan, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnIsClose(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnIsCloseGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): aclTensor on the device of `self` in the formula. [non-contiguous tensor] (../../../docs/en/context/non_contiguous_tensor.md) are supported. The dtype must be the same as that of `other`. The `shape` and `other` must meet the broadcast relationship. The [data format](../../../docs/en/context/data_format.md) can be ND, and the data is less than or equal to 8D.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, or DOUBLE.
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, DOUBLE, or BFLOAT16.
  - `other` (aclTensor*, compute input): aclTensor on the device. The data type can be FLOAT, FLOAT16, INT32, or BFLOAT16. [non-contiguous tensor] (../../../docs/en/context/non_contiguous_tensor.md) are supported, the dtype must be the same as that of `self`. The `shape` and `other` must meet the broadcast relationship. The [data format](../../../docs/en/context/data_format.md) can be ND, and the data is less than or equal to 8D.
  - `rtol` (double, compute input): `rtol` in the formula, indicating the relative tolerance. The data type can be DOUBLE.
  - `atol` (double, compute input): `atol` in the formula, indicating the absolute difference. The data type can be DOUBLE.
  - `equal_nan` (bool, compute input): `equal_nan` in the formula, indicating the comparison of two NaN values. If the value is `True`, two `NaN` values are considered equal. If the value is `False`, two `NaN` values are considered unequal. The data type can be BOOL.
  - `out` (aclTensor*, computation output): `out` in the formula, the calculation result, which indicates whether the elements in the two inputs are close. `aclTensor` on the device. The data type is BOOL. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. Its `shape` should be the same as that of `self`. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape is the same as that of `self`, and the data is not greater than 8D.
  - `workspaceSize` (uint64_t*, compute output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, compute output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed `self`, `other`, or `out` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of `self` or `other` is not supported.
                                        2. The data types of `self` and `other` are different.
                                        3. Broadcasting cannot be performed for the shapes of `self` and `other`.
                                        4. The shape of `self` or `other` is greater than 8D.
                                        5. The shape of `out` is different from that of `other` after broadcasting.
  ```

## aclnnIsClose

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnIsCloseGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnIsClose` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_isclose.h"

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
  // 2. Construct the input and output based on the API.
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
  std::vector<float> otherHostData = {1, 1, 1, 2, 1, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  double rtol = 1.0;
  double atol = 1.0;
  bool equal_nan = false;
  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BOOL, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnIsClose.
  ret = aclnnIsCloseGetWorkspaceSize(self, other, rtol, atol, equal_nan, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsCloseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnIsClose.
  ret = aclnnIsClose(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIsClose failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(other);
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(otherDeviceAddr);
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
