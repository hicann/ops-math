# aclnnReciprocal&aclnnInplaceReciprocal

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/reciprocal)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    √    |

## Function

- Description: Returns a new tensor with the reciprocal of each input element.

- Formula:

$$
out = {\frac{1} {input}}
$$

- Example:

  input = tensor([1,2,3,4]). After reciprocal computation, out = tensor([1.00, 0.50, 0.33, 0.25]).

## Prototype

- `aclnnReciprocal` and `aclnnInplaceReciprocal` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnReciprocal`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceReciprocal`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnReciprocalGetWorkspaceSize` or `aclnnInplaceReciprocalGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnReciprocal` or `aclnnInplaceReciprocal` is called to perform computation.
  - `aclnnStatus aclnnReciprocalGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnReciprocal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceReciprocalGetWorkspaceSize(const aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceReciprocal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnReciprocalGetWorkspaceSize

- **Parameters:**

  * `self` (aclTensor*, computation input): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. An empty tensor can be passed in. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, COMPLEX64, COMPLEX128, FLOAT32, FLOAT16, INT8, INT16, INT32, INT64, UINT8, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, COMPLEX64, COMPLEX128, FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, or BOOL.
  * `out` (aclTensor \*, computation output): aclTensor on the device. The shape must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) and empty input tensors are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, COMPLEX64, COMPLEX128, FLOAT32, or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, COMPLEX64, COMPLEX128, FLOAT32, FLOAT16, and BFLOAT16.
  * `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  
  * `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed `self` or `out` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of `self` or `out` is not supported.
                                   2. The data format of `self` or `out` is not supported.
                                   3. The dimensions of `self` are greater than 8.
                                   4. The data format of `self` is different from that of `out`.
                                   5. The shape of `self` is different from that of `out`.
  ```

## aclnnReciprocal

- **Parameters:**

  * `workspace` (void \*, input): memory address of the workspace to be allocated on the device.

  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnReciprocalGetWorkspaceSize`.

  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.

  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceReciprocalGetWorkspaceSize

- **Parameters:**

  * `selfRef` (aclTensor*, computation input|computation output): aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. An empty tensor can be passed in. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be DOUBLE, COMPLEX64, COMPLEX128, FLOAT32, or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, COMPLEX64, COMPLEX128, FLOAT32, FLOAT16, and BFLOAT16.
  * `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed `selfRef` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of `selfRef` is not supported.
                                   2. The data format of `selfRef` is not supported.
                                   3. The dimensions of `selfRef` are greater than 8.
  ```

## aclnnInplaceReciprocal

- **Parameters:**

  * `workspace` (void \*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnInplaceReciprocalGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnReciprocal` and `aclnnInplaceReciprocal` default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
**Sample code of aclnnReciprocal:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_reciprocal.h"

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

  // Compute the strides of contiguous tensors.
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

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> outShape = {1, 2, 4};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnReciprocal.
  ret = aclnnReciprocalGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReciprocalGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnReciprocal.
  ret = aclnnReciprocal(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReciprocal failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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

**Sample code of aclnnInplaceReciprocal:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_reciprocal.h"

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

  // Compute the strides of contiguous tensors.
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

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> selfRefShape = {1, 2, 4};
  void* selfRefDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceReciprocal.
  ret = aclnnInplaceReciprocalGetWorkspaceSize(selfRef, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceReciprocalGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceReciprocal.
  ret = aclnnInplaceReciprocal(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceReciprocal failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(selfRef);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfRefDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
