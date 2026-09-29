# aclnnComplex

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/complex)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Takes two tensors (`real` and `imag`) with identical shapes and data types as the real and imaginary parts of a complex number, and returns a complex-type tensor `out`.
- Formula:

  $$
  \text{out}[i] = \text{real}[i] + \text{imag}[i]*\mathrm{j}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnComplexGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnComplex` is called to perform computation.

- `aclnnStatus aclnnComplexGetWorkspaceSize(const aclTensor *real, const aclTensor *imag, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnComplex(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnComplexGetWorkspaceSize

- **Parameters**

  - `real` (aclTensor*, computation input): input `real` in the formula. The data type can be FLOAT, FLOAT16, or DOUBLE. The shape must be broadcastable with that of `imag` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `imag` (aclTensor*, computation input): input `imag` in the formula. The data type can be FLOAT, FLOAT16, or DOUBLE. The shape must be broadcastable with that of `real` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `out` (aclTensor*, computation output): `out` in the formula. The shape must be the broadcast result of `real` and `imag`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: For `out` in the formula, the data type can be COMPLEX64 (input must be FLOAT), COMPLEX32 (input must be FLOAT16), or COMPLEX128 (input must be DOUBLE).
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: For `out` in the formula, the data type can be COMPLEX64 (input must be FLOAT) or COMPLEX128 (input must be DOUBLE).
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed real, imag, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type and data format of real and imag are not supported.
                                    2. The shapes of real and imag are not broadcastable.
                                    3. The number of dimensions of real or imag exceeds 8.
                                    4. The data types of real and imag are different.
  ```

## aclnnComplex

- **Parameters**
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnComplexGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnComplex` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <complex>
#include "acl/acl.h"
#include "aclnnop/aclnn_complex.h"

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
  std::vector<int64_t> realShape = {4, 2};
  std::vector<int64_t> imagShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* realDeviceAddr = nullptr;
  void* imagDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* real = nullptr;
  aclTensor* imag = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> realHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> imagHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<std::complex<float>> outHostData(8, 0);
  // Create a real aclTensor.
  ret = CreateAclTensor(realHostData, realShape, &realDeviceAddr, aclDataType::ACL_FLOAT, &real);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an imag aclTensor.
  ret = CreateAclTensor(imagHostData, imagShape, &imagDeviceAddr, aclDataType::ACL_FLOAT, &imag);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnComplex.
  ret = aclnnComplexGetWorkspaceSize(real, imag, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnComplex.
  ret = aclnnComplex(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplex failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(real);
  aclDestroyTensor(imag);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(realDeviceAddr);
  aclrtFree(imagDeviceAddr);
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
