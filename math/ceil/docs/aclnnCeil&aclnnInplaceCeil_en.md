# aclnnCeil&aclnnInplaceCeil

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/ceil)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Returns the round-up result of each element in the input tensor.

- Formula:

$$
out_i =⌈self_i⌉
$$

## Prototype

- `aclnnCeil` and `aclnnInplaceCeil` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnCeil`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceCeil`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnCeilGetWorkspaceSize` or `aclnnInplaceCeilGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnCeil` or `aclnnInplaceCeil` is called to perform computation.
  - `aclnnStatus aclnnCeilGetWorkspaceSize(const aclTensor *self,  aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnCeil(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceCeilGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceCeil(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnCeilGetWorkspaceSize

- **Parameters**
  
  * `self` (aclTensor*, computation input): `self` in the formula, `aclTensor` on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The shape must not exceed 8D. The shape and data type must be the same as those of `out`. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, or BFLOAT16.  
  * `out` (aclTensor \*, computation output): `out` in the formula, `aclTensor` on the device. The shape must not exceed 8D. The shape and data type must be the same as those of `self`. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, or BFLOAT16. 
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter validation. The following error codes may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self is not supported.
                                  2. The data types of self and out are inconsistent.
                                  3. The shapes of self and out are inconsistent.
                                  4. The number of dimensions of self exceeds 8.
```

## aclnnCeil

- **Parameters**
  
  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnCeilGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceCeilGetWorkspaceSize

- **Parameters**
  
  * `selfRef` (aclTensor \*, computation input | computation output): `self` in the formula, `aclTensor` on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The shape must not exceed 8D. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, or DOUBLE.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, or BFLOAT16.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter validation. The following error codes may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of selfRef is not supported.
                                  2. The number of dimensions of selfRef exceeds 8.
```

## aclnnInplaceCeil

- **Parameters**
  
  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnCeilGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnCeil` and `aclnnInplaceCeil` default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ceil.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& inputShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, void** outDeviceAddr,
    aclTensor** self, aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  std::vector<double> outHostData(8, 0);

  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, inputShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_DOUBLE, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize, void* selfDeviceAddr,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);

  // aclnnCeil API call example
  LOG_PRINT("test aclnnCeil\n");

  // Call the first-phase API of aclnnCeil.
  auto ret = aclnnCeilGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCeilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the second-phase API of aclnnCeil.
  ret = aclnnCeil(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCeil failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the output result.
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  // aclnnInplaceCeil API call example
  LOG_PRINT("\ntest aclnnInplaceCeil\n");

  // Call the first-phase API of aclnnInplaceCeil.
  ret = aclnnInplaceCeilGetWorkspaceSize(self, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCeilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Re-allocate device memory based on workspaceSize (consistent with the original logic).
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the second-phase API aclnnInplaceCeil.
  ret = aclnnInplaceCeil(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCeil failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy the self result.
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  // 1. Initialize the device and stream.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output.
  std::vector<int64_t> inputShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(inputShape, outShape, &selfDeviceAddr, &outDeviceAddr, &self, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator APIs (aclnnCeil and aclnnInplaceCeil).
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, out, &workspaceAddr, workspaceSize, selfDeviceAddr, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 6. Release aclTensors.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources.
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
