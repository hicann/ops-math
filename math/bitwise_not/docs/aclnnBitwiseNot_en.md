# aclnnBitwiseNot

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/bitwise_not)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Performs the logical NOT operation when the input is a BOOL tensor; performs bitwise NOT operation when the input is of the INT type.
- Formula:

  $$
  out_i = \lnot self_i
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBitwiseNotGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnBitwiseNot` is called to perform computation.

- `aclnnStatus aclnnBitwiseNotGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnBitwiseNot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBitwiseNotGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): `self` in the formula. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md), and its [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT16, INT32, INT64, INT8, UINT8, or BOOL.
  - `out` (aclTensor\*, computation output): `out` in the formula. Its shape is the same as that of `self`. Its [data format](../../../docs/en/context/data_format.md) can be ND, and its data type must be the same as that of `self`.
     * <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT16, INT32, INT64, INT8, UINT8, or BOOL.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                   2. The data type of self is different from that of out.
                                   3. The shape of self is different from that of out.
  ```

## aclnnBitwiseNot

- **Parameters:**

  * `workspace` (void\*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnBitwiseNotGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, covering the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnBitwiseNot` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bitwise_not.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> selfHostData = {1, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnBitwiseNotGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBitwiseNotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnBitwiseNot(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBitwiseNot failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
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
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
