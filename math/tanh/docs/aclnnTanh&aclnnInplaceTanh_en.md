# aclnnTanh & aclnnInplaceTanh

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/tanh)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     ×     |
| <term>Atlas inference products</term>                            |   ×     |
| <term>Atlas training products</term>                             |   √     |

## Function

- Operator function: Activation function. Returns a tensor with the same shape as the input tensor by performing elementwise computation on the input tensor.
- Formula:

$$
tanh(self)=\frac{e^{self}-e^{-self}}{e^{self} + e^{-self}}
$$

## Prototype

- `aclnnTanh` and `aclnnInplaceTanh` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnTanh`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceTanh`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTanhGetWorkspaceSize` or `aclnnInplaceTanhGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the computation process. Then, `aclnnTanh` or `aclnnInplaceTanh` is called to perform computation.

  - `aclnnStatus aclnnTanhGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnTanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceTanhGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnInplaceTanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnTanhGetWorkspaceSize

- **Parameter Description**

  - `self` (aclTensor\*, computation input): input `self` in the formula, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape dimension cannot exceed 8, and must be the same as that of `out`. The data type must meet the type deduction rules with `out`.
    - <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, BOOL, UINT8, INT8, INT16, INT32, or INT64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, BOOL, UINT8, INT8, INT16, INT32, INT64, or BFLOAT16.
  - `out` (aclTensor\*, computation output): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The shape must be the same as that of `self`. The [data format](../../../docs/en/context/data_format.md) can be ND. The data type must meet the type deduction rules with `self`.
    - <term>Atlas training products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or out is not supported.
                                        2. The data types of self and out do not meet the type deduction rules.
                                        3. The dimension of self or out is greater than 8.
                                        4. The shapes of self and out are inconsistent.
  ```

## aclnnTanh

- **Parameter Description**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnTanhGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceTanhGetWorkspaceSize

- **Parameter Description**

  - `selfRef` (aclTensor*, computation input/output): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef is not supported.
  ```

## aclnnInplaceTanh

- **Parameter Description**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceTanhGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computation:
  - `aclnnTanh` and `aclnnInplaceTanh` default to deterministic implementation.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_tanh.h"

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
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Change the API name to the actual one.
  // aclnnTanh API call example
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnTanh.
  ret = aclnnTanhGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTanh.
  ret = aclnnTanh(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanh failed. ERROR: %d\n", ret); return ret);

  // aclnnInplaceTanh API call example
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceTanh.
  ret = aclnnInplaceTanhGetWorkspaceSize(self, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTanhGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceTanh.
  ret = aclnnInplaceTanh(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTanh failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnTanh result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplaceTanh result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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
