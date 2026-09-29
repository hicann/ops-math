# aclnnClampMaxTensor&aclnnInplaceClampMaxTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/clip_by_value_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Limits all input elements within the range of [-inf, max].

- Formula:

  $$
  {out}_{i} = min({{self}_{i}},{max}_{i})
  $$

## Prototype

- `aclnnClampMaxTensor` and `aclnnInplaceClampMaxTensor` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnClampMaxTensor`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceClampMaxTensor`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnClampMaxTensorGetWorkspaceSize` or `aclnnInplaceClampMaxTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnClampMaxTensor` or `aclnnInplaceClampMaxTensor` is called to perform computation.

  - `aclnnStatus aclnnClampMaxTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* max, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnClampMaxTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceClampMaxTensorGetWorkspaceSize(aclTensor* selfRef, const aclTensor* max, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnInplaceClampMaxTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnClampMaxTensorGetWorkspaceSize

- **Parameters**
  
  - `self` (aclTensor*, computation input): input tensor, `aclTensor` on the device. The shape must be 1D to 8D. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `max`. The shape must be broadcastable with that of `max` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, or INT64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16.

  - `max` (aclTensor*, computation input): input upper limit tensor. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`. The shape must be broadcastable with that of `self` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BFLOAT16, or BOOL.

  - `out` (aclTensor*, computation output): output tensor. The shape must be the broadcast result of `self` and `max`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, or INT64. The data type must be the same as that of `self` and must be convertible from the deduced data type of `self` and `max`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16. The data type must be the same as that of `self` and must be convertible from the deduced data type of `self` and `max`.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, max, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type deduced from self and max is not supported.
                                    2. The shapes of self and max are not broadcastable, or the broadcast result does not match the shape of out.
                                    3. The data type deduction of self and max failed, or the deduced type cannot be converted to the type of out.
                                    4. The number of dimensions of self, max, or out is greater than 8.
  ```

## aclnnClampMaxTensor

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnClampMaxTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceClampMaxTensorGetWorkspaceSize

- **Parameters**

  - `selfRef` (aclTensor*, computation input | computation output): input and output tensor, `self` and `out` in the formula. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `max`. The data type must be convertible from the deduced data type of `selfRef` and `max` (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)). The shape must be broadcastable with that of `max` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, or INT64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16.

  - `max` (aclTensor*, computation input): input upper limit tensor. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`. The shape must be broadcastable with that of `selfRef` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, or INT64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef or max is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type deduced from selfRef and max is not supported.
                                    2. The shapes of selfRef and max are not broadcastable.
                                    3. The data type deduction of selfRef and max failed.
                                    4. The number of dimensions of selfRef or max exceeds 8.
                                    5. The data type deduction of selfRef and max failed, or the deduced data type cannot be converted to that of selfRef.
  ```

## aclnnInplaceClampMaxTensor

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceClampMaxTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnClampMaxTensor` and `aclnnInplaceClampMaxTensor` default to a deterministic implementation.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

**aclnnClampMaxTensor Sample Code**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clamp.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<int64_t>& maxShape,
    void** selfDeviceAddr, aclTensor** self, void** maxDeviceAddr, aclTensor** max, void** outDeviceAddr,
    aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> maxHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<double> outHostData(8, 0);

  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a max aclTensor.
  ret = CreateAclTensor(maxHostData, maxShape, maxDeviceAddr, aclDataType::ACL_DOUBLE, max);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_DOUBLE, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclTensor* max, aclTensor* out)
{
  aclDestroyTensor(self);
  aclDestroyTensor(max);
  aclDestroyTensor(out);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* maxDeviceAddr, void* outDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
  aclrtFree(selfDeviceAddr);
  aclrtFree(maxDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> maxShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* maxDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* max = nullptr;
  aclTensor* out = nullptr;

  ret = PrepareInputAndOutput(
      selfShape, maxShape, outShape, &selfDeviceAddr, &self, &maxDeviceAddr, &max, &outDeviceAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnClampMaxTensor.
  ret = aclnnClampMaxTensorGetWorkspaceSize(self, max, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampMaxTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnClampMaxTensor.
  ret = aclnnClampMaxTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampMaxTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition. 
  ReleaseTensorAndScalar(self, max, out);
  
  // 7. Release device resources. Modify the code based on the API definition.
  ReleaseDevice(selfDeviceAddr, maxDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```

**aclnnInplaceClampMaxTensor Sample Code**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clamp.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& maxShape, void** selfDeviceAddr, aclTensor** self,
    void** maxDeviceAddr, aclTensor** max)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> maxHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a max aclTensor.
  ret = CreateAclTensor(maxHostData, maxShape, maxDeviceAddr, aclDataType::ACL_DOUBLE, max);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclTensor* max)
{
  aclDestroyTensor(self);
  aclDestroyTensor(max);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* maxDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
  aclrtFree(selfDeviceAddr);
  aclrtFree(maxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> maxShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* maxDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* max = nullptr;
  
  ret = PrepareInputAndOutput(selfShape, maxShape, &selfDeviceAddr, &self, &maxDeviceAddr, &max);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceClampMaxTensor.
  ret = aclnnInplaceClampMaxTensorGetWorkspaceSize(self, max, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceClampMaxTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceClampMaxTensor.
  ret = aclnnInplaceClampMaxTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceClampMaxTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition. 
  ReleaseTensorAndScalar(self, max);

  // 7. Release device resources. Modify the code based on the API definition.
  ReleaseDevice(selfDeviceAddr, maxDeviceAddr,  workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```
