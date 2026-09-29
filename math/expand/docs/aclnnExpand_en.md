# aclnnExpand

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/expand)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Broadcasts the input tensor `self` to a tensor of a specified `size`.

- Example:

    $$
    \begin{aligned}
    & self=\begin{bmatrix}
    3
    \end{bmatrix}\\
    & size = \begin{bmatrix}
    2,2
    \end{bmatrix}\\
    & out =\begin{bmatrix}
    3 & 3\\
    3 & 3\\
    \end{bmatrix}\\
    \end{aligned}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnExpandGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnExpand` is called to perform computation.

- `aclnnStatus aclnnExpandGetWorkspaceSize(const aclTensor* self, const aclIntArray* size, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnExpand(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnExpandGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): target tensor to be broadcasted, that is, `self` in the formula and aclTensor on the device. It supports shapes from 0 to 8 dimensions, and the shape must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `size`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT16, FLOAT, UINT8, INT8, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, INT32, INT64, or BOOL.
  - `size` (aclIntArray*, compute input): size specified during broadcasting, that is, aclIntArray on the host. The data type can be INT.
  - `out` (aclTensor*, compute output): tensor after broadcasting, that is, aclTensor on the device. The data type must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape must match the shape of `self` derived based on `size`.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT16, FLOAT, UINT8, INT8, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, INT32, INT64, or BOOL.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API performs input parameter validation. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, size, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or out is not supported.
                                   2. The data types of self and out are inconsistent.
                                   3. The shape and size of self or out do not match.
                                      3.1 The number of elements in size is less than the number of dimensions of self.
                                      3.2 Any dimension of self is neither 1 nor equal to the corresponding dimension after size offset. (The offset is the difference between size and the number of dimensions of self.)
                                      3.3 The output shape is not equal to the expected shape (size).
                                   4. The maximum dimension of self exceeds 8.
  ```

## aclnnExpand

- **Parameters:**

  - `workspace` (void\*, input): memory address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnExpandGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- Deterministic computation:
  - `aclnnExpand` defaults to a deterministic implementation.

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

None

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_expand.h"

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
  // Set the device ID based on the actual device.
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
  aclIntArray* size = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t sizeValue[2] = {4, 2};
  size = aclCreateIntArray(&(sizeValue[0]), 2);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnExpand.
  ret = aclnnExpandGetWorkspaceSize(self, size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpandGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnExpand.
  ret = aclnnExpand(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpand failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto resultSize = GetShapeSize(outShape);
  std::vector<float> resultData(resultSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, resultSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < resultSize; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyIntArray(size);

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
