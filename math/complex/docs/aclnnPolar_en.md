# aclnnPolar

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/complex)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Operator function: Computes the absolute value of input and computes the Cartesian coordinates for the polar coordinates of angle, which returns a complex tensor out.

- Formula:

  $$
  out = abs*cos(angle) + abs*sin(angle)*j
  $$

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnPolarGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnPolar is called to perform computation.

- `aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor* input, const aclTensor* angle, aclTensor* out,uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnPolar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnPolarGetWorkspaceSize

- **Parameters:**

  - `input` (aclTensor*, compution input): input tensor, `abs` in the formula. The data type can be FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape cannot be greater than 8D.

  - `angle` (aclTensor*, compution input): input tensor, `angle` in the formula. The data type must be the same as that of input. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape must be the same as that of `input`.

  - `out` (aclTensor*, computation output): output tensor, `out` in the formula. The data type can be COMPLEX64. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape must be the same as that of `input`.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown.
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed input, angle, or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of input or angle is not supported.
                                      2. The data types of input and angle are inconsistent.
                                      3. The data type of out is not COMPLEX64.
                                      4. The shape of input, angle, or out is greater than 8D, or their shapes are not consistent.                              
```

## aclnnPolar

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPolarGetWorkspaceSize.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnPolar` defaults to deterministic implementation.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_polar.h"
#include <complex>

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
  std::vector<int64_t> inputShape = {1, 2};
  std::vector<int64_t> angleShape = {1, 2};
  std::vector<int64_t> outShape = {1, 2};
  void* inputDeviceAddr = nullptr;
  void* angleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* angle = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> inputHostData = {1, 2};
  std::vector<float> angleHostData = {1, 2};
  std::vector<std::complex<float>> outHostData = {1, 2};
  // Create an input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an angle aclTensor.
  ret = CreateAclTensor(angleHostData, angleShape, &angleDeviceAddr, aclDataType::ACL_FLOAT, &angle);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPolar.
  ret = aclnnPolarGetWorkspaceSize(input, angle, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPolarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPolar.
  ret = aclnnPolar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPolar failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<std::complex<float>> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
   // In the output, the first part is the real part, and the second part is the imaginary part.
   std::cout << "result[" << i << "]: " <<resultData[i] << std::endl;
  }

  // 6. Release aclTensor and aclIntArray. Modify the configuration based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(angle);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(inputDeviceAddr);
  aclrtFree(angleDeviceAddr);
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
