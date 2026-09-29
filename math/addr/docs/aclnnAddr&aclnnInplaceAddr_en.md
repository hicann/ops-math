# aclnnAddr&aclnnInplaceAddr

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/addr)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Computes the outer product of `vec1` and `vec2` to obtain a two-dimensional matrix, scales this product by a coefficient, and adds the result to `self` scaled by another coefficient.

- Formula:

  $$
  \text{out} = \beta\ \text{self} + \alpha\ (\text{vec1} \otimes\text{vec2})
  $$

## Prototype

- `aclnnAddr` and `aclnnInplaceAddr` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnAddr`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceAddr`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAddrGetWorkspaceSize` or `aclnnInplaceAddrGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the process. Then, `aclnnAddr` or `aclnnInplaceAddr` is called to perform computation.

  * `aclnnStatus aclnnAddrGetWorkspaceSize(const aclTensor *self, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *betaOptional, const aclScalar *alphaOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnAddr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  * `aclnnStatus aclnnInplaceAddrGetWorkspaceSize(aclTensor *selfRef, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *betaOptional, const aclScalar *alphaOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceAddr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnAddrGetWorkspaceSize

- **Parameters:**
  
  - `self` (aclTensor*, computation·input): a matrix to be broadcast to the shape of the outer product. It is an aclTensor on the device. Its shape cannot exceed 2 dimensions. The shapes of `self`, `vec1`, and `vec2` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `vec1` (aclTensor*, computation input): first input vector (1D) for the outer product, aclTensor on the device. The shapes of `vec1` and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `vec2` (aclTensor*, computation input): second input vector (1D) for the outer product, aclTensor on the device. The shapes of `vec2` and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `betaOptional` (aclScalar*, computation input): scaling factor `β` applied to `self`. It is an aclScalar on the host. If `betaOptional` is of the BOOL type, `self`, `vec1`, and `vec2` must be of the BOOL type. If `self`, `vec1`, or `vec2` is an integer, `betaOptional` or `alphaOptional` cannot be of the floating-point type. Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `alphaOptional` (aclScalar*, computation input): scaling factor for the outer product extended matrix, corresponding to `α` in the formula. It is an aclScalar on the host. If `alphaOptional` is of the BOOL type, `self`, `vec1`, and `vec2` must be of the BOOL type. If `self`, `vec1`, or `vec2` is an integer, `betaOptional` or `alphaOptional` cannot be of the floating-point type. Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `out` (aclTensor\*, computation output): output result, aclTensor on the device. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor\*\*, output): operator executor, covering the operator computation process.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input tensor or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self, vec1, or vec2 is not supported.
                                    2. vec1 and vec2 are not one-dimensional, and self has more than two dimensions.
                                    3. self cannot be broadcast to the shape of the outer product result of vec1 and vec2.
                                    4. When beta or alpha is of the BOOL type, but self, vec1, and vec2 are not of the BOOL type.
                                    5. When self, vec1, and vec2 are all integers, all BOOL, or a mix of integers and BOOL, but beta or alpha are floating-point.
  ```

## aclnnAddr

- **Parameters:**
  
  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnAddrGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, covering the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddrGetWorkspaceSize

- **Parameters:**
  
  - `selfRef` (aclTensor\*, computation input/output): outer product extended matrix and output matrix, aclTensor on the device. Its shape has two dimensions. Empty tensors are not supported. The shapes of `selfRef`, `vec1`, and `vec2` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `vec1` (aclTensor*, computation input): first input vector (1D) for the outer product, aclTensor on the device. The shapes of `vec1` and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `vec2` (aclTensor*, computation input): second input vector (1D) for the outer product, aclTensor on the device. The shapes of `vec2` and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `betaOptional` (aclScalar*, computation input): scaling factor for the outer product extended matrix, corresponding to `β` in the formula. It is an aclScalar on the host. If `betaOptional` is of the BOOL type, `self`, `vec1`, and `vec2` must be of the BOOL type. If `self`, `vec1`, or `vec2` is an integer, `betaOptional` or `alphaOptional` cannot be of the floating-point type. Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `alphaOptional` (aclScalar*, computation input): scaling factor for the outer product extended matrix, corresponding to `α` in the formula. It is an aclScalar on the host. If `alphaOptional` is of the BOOL type, `self`, `vec1`, and `vec2` must be of the BOOL type. If `self`, `vec1`, or `vec2` is an integer, `betaOptional` or `alphaOptional` cannot be of the floating-point type. Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, or BFLOAT16.

  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor\*\*, output): operator executor, covering the operator computation process.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input tensor is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of selfRef, vec1, or vec2 is not supported.
                                    2. vec1 and vec2 are not one-dimensional, and selfRef has more than two dimensions.
                                    3. selfRef cannot be broadcast to the shape of the outer product result of vec1 and vec2.
                                    4. When beta or alpha is of the BOOL type, but selfRef, vec1, and vec2 are not of the BOOL type.
                                    5. When selfRef, vec1, and vec2 are all integers, all BOOL, or a mix of integers and BOOL, but beta or alpha are floating-point.
  ```

## aclnnInplaceAddr

- **Parameters:**
  
  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceAddrGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, covering the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnAddr` and `aclnnInplaceAddr` each default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addr.h"

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
  // Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Construct the input and output according to the API.
  std::vector<int64_t> inputShape = {3, 2};
  std::vector<int64_t> vec1Shape = {3};
  std::vector<int64_t> vec2Shape = {2};
  std::vector<int64_t> outShape = {3, 2};

  void* inputDeviceAddr = nullptr;
  void* vec1DeviceAddr = nullptr;
  void* vec2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* vec1 = nullptr;
  aclTensor* vec2 = nullptr;
  aclScalar* beta = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> inputHostData = {6, 0};
  std::vector<float> vec1HostData = {1, 2, 3};
  std::vector<float> vec2HostData = {4, 5};
  std::vector<float> outHostData = {6, 0};
  float betaValue = 1.5f;
  float alphaValue = 1.5f;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vec1HostData, vec1Shape, &vec1DeviceAddr, aclDataType::ACL_FLOAT, &vec1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vec2HostData, vec2Shape, &vec2DeviceAddr, aclDataType::ACL_FLOAT, &vec2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create beta and alpha scalar values.
  beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);

 
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
   
   // aclnnAddr API call example
   //Call the first-phase API of aclnnAddr.
  ret = aclnnAddrGetWorkspaceSize(input, vec1, vec2, beta, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnAddr.
  ret = aclnnAddr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddr failed. ERROR: %d\n", ret); return ret);

  // (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceAddr API call example
  // Call the first-phase API of aclnnInplaceAddr.
  ret = aclnnInplaceAddrGetWorkspaceSize(input, vec1, vec2, beta, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnInplaceAddr.
  ret = aclnnInplaceAddr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddr failed. ERROR: %d\n", ret); return ret);

  // (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // Destroy aclTensor and aclScalar.
  aclDestroyTensor(input);
  aclDestroyTensor(vec1);
  aclDestroyTensor(vec2);
  aclDestroyTensor(out);

  // Free device resources.
  aclrtFree(inputDeviceAddr);
  aclrtFree(vec1DeviceAddr);
  aclrtFree(vec2DeviceAddr);
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
