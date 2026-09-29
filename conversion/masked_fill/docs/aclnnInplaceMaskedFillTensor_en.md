# aclnnInplaceMaskedFillTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/masked_fill)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Uses `value` to fill in the element corresponding to the position whose value is `true` in the `mask` matrix in `selfRef`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnInplaceMaskedFillTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnInplaceMaskedFillTensor` is called to perform computation.

* `aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnInplaceMaskedFillTensor(void* workspace, uint64_t workspace_size, aclOpExecutor* executor, aclrtStream stream)`

## aclnnInplaceMaskedFillTensorGetWorkspaceSize

- **Parameters:**

  - `selfRef` (aclTensor \*, computation input | computation output): input and output tensor, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be BOOL, INT8, INT32, INT64, FLOAT, or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BOOL, INT8, INT32, INT64, FLOAT, FLOAT16, or BFLOAT16.
  - `mask` (aclTensor*, input): aclTensor on the device. The data type can be BOOL. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `selfRef`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `value` (aclTensor*, input): aclTensor on the device. Only 0D is supported. The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md) with `selfRef`. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be BOOL, INT8, INT32, INT64, FLOAT, or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BOOL, INT8, INT32, INT64, FLOAT, FLOAT16, or BFLOAT16.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ````text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input selfRef, mask, and value are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types and formats of selfRef and mask are not supported.
                                        2. Broadcasting cannot be performed for the shapes of selfRef and mask.
                                        3. The data type of value cannot be converted into that of selfRef.
  ````

## aclnnInplaceMaskedFillTensor

- **Parameters:**

  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * `workspace_size` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceMaskedFillTensorGetWorkspaceSize`.
  * `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnInplaceMaskedFillTensor` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_masked_fill_tensor.h"

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

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> maskShape = {4, 2};
  std::vector<int64_t> valueShape = {};
  void* selfRefDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* mask = nullptr;
  aclTensor* value = nullptr;
  std::vector<float> selfHostData = {1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1};
  std::vector<char> maskHostData = {0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<float> valueHostData = {3.3};

  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a mask aclTensor.
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceMaskedFillTensor.
  ret = aclnnInplaceMaskedFillTensorGetWorkspaceSize(selfRef, mask, value, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMaskedFillTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceMaskedFillTensor.
  ret = aclnnInplaceMaskedFillTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMaskedFillTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(mask);
  aclDestroyTensor(value);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(maskDeviceAddr);
  aclrtFree(valueDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
