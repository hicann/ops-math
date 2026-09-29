# aclnnBincount

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/bincount)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Description: Counts the occurrence of each non-negative integer in an array. `minlength` is the minimum size of the output tensor. When `weights` is a null pointer, each occurrence of element `self[i]` increments the corresponding `out` value by 1. When weights is provided, the corresponding `out` value is incremented by `weights[i]`. Consequently, the size of `out` is the greater of (the maximum value in `self` + 1) and `minlength`.

- Formulas:

  If `n` is the value of `self` at position `i` and `weights` is specified, then:
  
  $$
  out[self_i] = out[self_i] + weights_i
  $$
  
  Otherwise:
  
  $$
  out[self_i] = out[self_i] + 1
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBincountGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor covering the operator computation process. Then, `aclnnBincount` is called to perform computation.

* `aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor* self, const aclTensor* weights, int64_t minlength,aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnBincountGetWorkspaceSize

- **Parameters:**

  * `self` (aclTensor*, computation input): aclTensor on the device. The data type can be INT8, INT16, INT32, INT64, or UINT8. The value must be a non-negative integer. The [data format](../../../docs/en/context/data_format.md) can be 1-dimensional ND. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md).
  * `weights` (aclTensor*, computation input): aclTensor on the device, which is the weight of each `self` value. It can be a null pointer. The data type can be FLOAT, FLOAT16, FLOAT64, INT8, INT16, INT32, INT64, UINT8, or BOOL. The [data format](../../../docs/en/context/data_format.md) can be 1-dimensional ND. The shape must be the same as that of `self`. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md).
  * `minlength` (int64_t, computation input): an integer on the host, which specifies the minimum length of the output tensor. This parameter ensures the minimum length of the `out`. If the maximum value of `self` is less than `minlength`, the length of `out` is `minlength`. Otherwise, the length of `out` is the maximum value of `self` + 1.
  * `out` (aclTensor*, computation output): aclTensor on the device. The data type can be INT32, INT64, FLOAT, or DOUBLE. The [data format](../../../docs/en/context/data_format.md) can be 1-dimensional ND. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). The length of `out` is the maximum value of `self` + 1 or `minlength`, whichever is greater.
  * `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor \**, output): operator executor, covering the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. self or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self, out, or weights is not supported.
                                  2. When weights is not empty, the shapes of self and weights are inconsistent.
```

## aclnnBincount

- **Parameters:**

  * `workspace` (void \*, input): address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnBincountGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, covering the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnBincount` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max.h"
#include "aclnnop/aclnn_bincount.h"

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
  // Initialize the device and stream. For details, see the ACL API manual.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Call max to compute the maximum element value in self, and then use the greater of (the maximum value + 1) and minlength as the output tensor size.
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> maxOutShape = {1};

  void* selfDeviceAddr = nullptr;
  void* maxOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* maxOut = nullptr;
  std::vector<int32_t> selfHostData = {8, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> maxOutHostData(1, 0);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a maxOut aclTensor.
  ret = CreateAclTensor(maxOutHostData, maxOutShape, &maxOutDeviceAddr, aclDataType::ACL_INT32, &maxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Call the CANN operator library API.
  uint64_t workspaceSizeMax = 0;
  aclOpExecutor* executorMax;
  // Call the first-phase API of aclnnMax.
  ret = aclnnMaxGetWorkspaceSize(self, maxOut, &workspaceSizeMax, &executorMax);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddrMax = nullptr;
  if (workspaceSizeMax > 0) {
    ret = aclrtMalloc(&workspaceAddrMax, workspaceSizeMax, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMax.
  ret = aclnnMax(workspaceAddrMax, workspaceSizeMax, executorMax, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMax failed. ERROR: %d\n", ret); return ret);

  // Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result from the device to the host.
  std::vector<int32_t> resultData(1, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), maxOutDeviceAddr,
                    sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  aclDestroyTensor(maxOut);

  // Call bincount.
  int64_t minlength = 0;
  int64_t outSize = (resultData[0] < minlength) ? minlength : resultData[0] + 1;
  std::vector<int64_t> weightsShape = {8};
  std::vector<int64_t> outShape = {outSize};

  void* weightsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* weights = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> weightsHostData = {1, 1, 1.1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(outSize, 0);
  // Create a weights aclTensor.
  ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, aclDataType::ACL_FLOAT, &weights);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnBincount.
  ret = aclnnBincountGetWorkspaceSize(self, weights, minlength, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBincountGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBincount.
  ret = aclnnBincount(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBincount failed. ERROR: %d\n", ret); return ret);

  // Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result from the device to the host.
  auto size = GetShapeSize(outShape);
  std::vector<float> bincountResultData(size, 0);
  ret = aclrtMemcpy(bincountResultData.data(), bincountResultData.size() * sizeof(bincountResultData[0]), outDeviceAddr,
                    size * sizeof(bincountResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, bincountResultData[i]);
  }
  // Destroy aclTensor.
  aclDestroyTensor(self);
  aclDestroyTensor(weights);
  aclDestroyTensor(out);

  // Free resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSizeMax > 0) {
    aclrtFree(workspaceAddrMax);
  }

  aclrtFree(weightsDeviceAddr);
  aclrtFree(maxOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
