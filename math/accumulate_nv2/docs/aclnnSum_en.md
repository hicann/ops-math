# aclnnSum

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/accumulate_nv2)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    √    |

## Function

Operator function: Returns the sum of each input tensor in the input tensors list.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSumGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSum` is called to perform computation.

- `aclnnStatus aclnnSumGetWorkspaceSize(const aclTensorList* tensors, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnSumGetWorkspaceSize

- **Parameter Description:**

  - `tensors` (aclTensorList*, computation input): list of input tensors to be computed, aclTensorList on the device. The data type can be FLOAT16, FLOAT, INT8, INT32 or UINT8, and must be the same as the data type of `out`. The shape of each tensor in tensors must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `out`, and the number of dimensions cannot exceed 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `out` (aclTensor*, computation output): output tensor, aclTensor on the device. The data type can be FLOAT, FLOAT16, INT8, INT32 or UINT8. The data type must be the same as that of `tensors`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with each tensor in `tensors`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed tensors or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of tensors or out is not supported.
                                        2. The data types of tensors and out are inconsistent.
                                        3. The shapes of tensors and out do not meet the broadcast rules, or the shape of tensors after broadcasting is different from that of out.
  ```

## aclnnSum

- **Parameter Description:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnSumGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computation:
  - `aclnnSum` defaults to deterministic implementation.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sum.h"

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
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {2, 3};
  std::vector<int64_t> outShape = {2, 3};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9, 10, 11, 12};
  std::vector<float> outHostData(6, 0);
  // Create an input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an aclTensorList.
  std::vector<aclTensor*> tmp{input1, input2};
  aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSum.
  ret = aclnnSumGetWorkspaceSize(tensorList, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSum.
  ret = aclnnSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSum failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensorList(tensorList);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
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
