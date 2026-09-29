# aclnnLogAddExp

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/log_add_exp)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Computes the logarithm of the sum of exponents (power of *e*) of input tensors.

- Formula:

  $$
  out = \log(e^x+e^y) = \log[e^m(1+e^{-|x-y|})] = m+\log(1+e^{-|x-y|}) \quad, where m = \max(x, y)
  $$
    
  $x$ indicates self, and $y$ indicates other.

## Function Prototype

 Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnLogAddExpGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnLogAddExp is called to perform computation.

 - `aclnnStatus aclnnLogAddExpGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

 - `aclnnStatus aclnnLogAddExp(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLogAddExpGetWorkspaceSize

  - **Parameters:**

    - `self` (aclTensor*, compute input): $x$ in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. The data type must be convertible to that of out. For details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md). The shapes of self and other meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, or BOOL.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, BOOL, or BFLOAT16.
    - `other` (aclTensor*, compute input): $y$ in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. The data type must be convertible to that of out. For details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md). The shapes of self and other meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, or BOOL.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, BOOL, or BFLOAT16.
    - `out` (aclTensor*, compute output): $out$ in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions, and must be the same as that after broadcasting is performed for self and other. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
      - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, or BFLOAT16.
    - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor**, output): operator executor, covering the operator computation process.

  - **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```text
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, other, or out is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, other, or out is not supported.
                                          2. The data type of self or other cannot be converted to that of out.
                                          3. The dimensions of self, other, or out are greater than 8.
                                          4. Broadcasting cannot be performed for the shapes of self and other.
                                          5. The shape of out is different from that after broadcasting is performed for self and other.
    ```

## aclnnLogAddExp

  - **Parameters:**

    - `workspace` (void*, input): address of the workspace to be allocated on the device.
    - 'workspaceSize' (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnLogAddExpGetWorkspaceSize.
    - `executor` (aclOpExecutor*, input): operator executor, covering the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.

  - **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnLogAddExp` defaults to deterministic implementation.

## Calling Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_logaddexp.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

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
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLogAddExp.
  ret = aclnnLogAddExpGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogAddExpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnLogAddExp.
  ret = aclnnLogAddExp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogAddExp failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Destroy device resources. Modify the code based on the API definition.
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
