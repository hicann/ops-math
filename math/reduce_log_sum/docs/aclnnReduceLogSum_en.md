# aclnnReduceLogSum

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/reduce_log_sum)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    ×    |

## Function

Description: Computes the log sum of the input tensor's elements along the provided axes.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnReduceLogSumGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnReduceLogSum` is called to perform computation.

* `aclnnStatus aclnnReduceLogSumGetWorkspaceSize(const aclTensor* data, const aclIntArray* axes, bool keepDims, bool noopWithEmptyAxes, aclTensor* reduce, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnReduceLogSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnReduceLogSumGetWorkspaceSize

- **Parameters:**

  * `data` (aclTensor*, computation input): the target tensor involved in computation, aclTensor on the device. The shape supports 0D to 8D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The data type can be FLOAT16 or FLOAT32. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `axes` (aclIntArray*, computation input): axes for computation, aclIntArray on the host. The data type can be INT64. The value range is [–self.dim(), self.dim() – 1].
  * `keepDims` (bool, computation input): whether to retain the dimensions of the input tensor in the output tensor, BOOL value on the host.
  * `noopWithEmptyAxes` (bool, computation input): behavior when `axes` is empty, BOOL value on the host. `false` indicates that all axes are computed. `true` indicates that no computation is performed, and the output tensor is equal to the input tensor.
  * `reduce` (aclTensor*, computation output): computation result, aclTensor on the device. The shape supports 0D to 8D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The data type can be FLOAT16 or FLOAT32, which must be the same as that of `data`. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed `data`, `axes`, or `reduce` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of `data` or `reduce` is not supported.
                                   2. The shape of `reduce` does not match the actual shape.
                                   3. The dimensions specified in the `axes` array exceed the range of dimensions of the input tensor.
                                   4. The axes in the `axes` array are duplicate.
```

## aclnnReduceLogSum

- **Parameters:**

  * `workspace` (void*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnReduceLogSumGetWorkspaceSize`.
  * `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnReduceLogSum` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_log_sum.h"

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
  // 1. (Boilerplate) Initialize the device, context, and stream. For details, see the ACL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> dataShape = {4, 2};
  std::vector<int64_t> outShape = {2};
  void* dataDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* data = nullptr;
  aclIntArray* axes = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> dataHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0};
  std::vector<int64_t> axesData = {0};
  bool keepDims = false;
  bool noopWithEmptyAxes = false;
  // Create a data aclTensor.
  ret = CreateAclTensor(dataHostData, dataShape, &dataDeviceAddr, aclDataType::ACL_FLOAT, &data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an axes aclIntArray.
  axes = aclCreateIntArray(axesData.data(), 1);
  CHECK_RET(axes != nullptr, return ret);
  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnReduceLogSum.
  ret = aclnnReduceLogSumGetWorkspaceSize(data, axes, keepDims, noopWithEmptyAxes, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceLogSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnReduceLogSum.
  ret = aclnnReduceLogSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceLogSum failed. ERROR: %d\n", ret); return ret);

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

  // 6. Release the aclTensor and aclIntArray. Modify the code based on the API definition.
  aclDestroyTensor(data);
  aclDestroyIntArray(axes);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(dataDeviceAddr);
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
