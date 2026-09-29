# aclnnAtan&aclnnInplaceAtan

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/atan)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Computes the element-wise inverse tangent (arctan) of the input matrix.

- Formula:

  $$
  out_{i}=tan^{-1}(input_{i})
  $$

## Prototype

- `aclnnAtan` and `aclnnInplaceAtan` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnAtan`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceAtan`: No output tensor object needs to be created, and the computation result is written in place to the input tensor's memory.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAtanGetWorkspaceSize` or `aclnnInplaceAtanGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the computation process. Then, `aclnnAtan` or `aclnnInplaceAtan` is called to perform computation.

  - `aclnnStatus aclnnAtanGetWorkspaceSize(const aclTensor *input, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnAtan(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceAtanGetWorkspaceSize(aclTensor *inputRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceAtan(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnAtanGetWorkspaceSize

- **Parameters:**
  - `input` (aclTensor*, computation input): input tensor, aclTensor on the device. If the data type is INT8, INT16, INT32, INT64, UINT8, or BOOL, the data type is cast to FLOAT32 for computation, and the output is of the FLOAT32 type. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND. For a non-contiguous tensor, its dimensions cannot exceed 8, and its shape must be the same as that of `out`.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  - `out` (aclTensor\*, computation output): output tensor, aclTensor on the device. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND. For a non-contiguous tensor, its dimensions cannot exceed 8, and its shape must be the same as that of `input`.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of input or out is not supported.                                  
                                   2. The shapes of input and out are inconsistent.
                                   3. input or out has more than 8 dimensions.
  ```

## aclnnAtan

- **Parameters:**
  
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnAtanGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAtanGetWorkspaceSize

- **Parameters:**
  - `inputRef` (aclTensor*, computation input/output): input or output tensor, aclTensor on the device. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND. For a non-contiguous tensor, its dimensions cannot exceed 8.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. inputRef is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of inputRef is not supported.
                                   2. inputRef has more than 8 dimensions.
  ```

## aclnnInplaceAtan

- **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceAtanGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnAtan` and `aclnnInplaceAtan` each default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_atan.h"

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
  // Customize the error handling as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. aclnnAtan API calling example
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAtan.
  ret = aclnnAtanGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAtanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAtan.
  ret = aclnnAtan(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAtan failed. ERROR: %d\n", ret); return ret);

  // 3. aclnnInplaceAtan calling example
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceAtan.
  ret = aclnnInplaceAtanGetWorkspaceSize(self, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAtanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAtan.
  ret = aclnnInplaceAtan(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAtan failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
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
