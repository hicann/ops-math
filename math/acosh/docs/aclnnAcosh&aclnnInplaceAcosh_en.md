# aclnnAcosh&aclnnInplaceAcosh

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/acosh)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Performs the inverse hyperbolic cosine operation on each element of the input tensor and outputs the result.

- Formula:

  $$
  out=cosh^{-1}(self)
  $$

## Prototype

- `aclnnAcosh` and `aclnnInplaceAcosh` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnAcosh`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceAcosh`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAcoshGetWorkspaceSize` or `aclnnInplaceAcoshGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnAcosh` or `aclnnInplaceAcosh` is called to perform computation.
  * `aclnnStatus aclnnAcoshGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnAcosh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceAcoshGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceAcosh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnAcoshGetWorkspaceSize

- **Parameters:**

  * `self` (aclTensor*, computation input): `self` in the formula, which is aclTensor on the device. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, COMPLEX64, COMPLEX128, or BFLOAT16.
  * `out` (aclTensor*, computation output): `out` in the formula, which is aclTensor on the device. If the input is a complex number, the output must be a complex number. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                       2. The shapes of self and out are inconsistent.
                                       3. The data type of self cannot be cast to the specified type of out.
  ```

## aclnnAcosh

- **Parameters:**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnAcoshGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): operator executor, covering the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAcoshGetWorkspaceSize

- **Parameters:**

  * `selfRef` (aclTensor*, computation input/output): `self`/`out` in the formula, which is aclTensor on the device. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input selfRef is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef is not supported.
  ```

## aclnnInplaceAcosh

- **Parameters:**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnInplaceAcoshGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): operator executor, covering the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnAcosh` and `aclnnInplaceAcosh` each default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_acosh.h"

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
  // Boilerplate code for resource initialization.
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
  // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 1};
  std::vector<int64_t> outShape = {4, 1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1.3192, 1.9915, 1.9674, 1.7151};
  std::vector<float> outHostData = {0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAcosh.
  ret = aclnnAcoshGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcoshGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAcosh.
  ret = aclnnAcosh(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosh failed. ERROR: %d\n", ret); return ret);

  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceAcosh.
  ret = aclnnInplaceAcoshGetWorkspaceSize(self, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAcoshGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAcosh.
  ret = aclnnInplaceAcosh(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAcosh failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnAcosh result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr,
                    inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplaceAcosh result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

 // 7. Free device resources. Modify the code based on the API definition.
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
