# aclnnNormalTensorTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/random/stateless_random_normal_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

Operator function: Returns a random number obtained from the independent normal distribution of a given mean (tensor) and standard deviation (tensor). The shapes of mean and std do not need to match, but the total number of elements in each tensor must be the same.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNormalTensorTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNormalTensorTensor` is called to perform computation.

- `aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(const aclTensor *mean, const aclTensor *std, int64_t seed, int64_t offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnNormalTensorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNormalTensorTensorGetWorkspaceSize

- **Parameters:**

  - `mean` (aclTensor*, computation input): tensor for generating the mean value of random number distribution, which is an aclTensor on the device. The data type can be FLOAT16, FLOAT, or DOUBLE. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `std`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `std`. The shape cannot exceed eight dimensions. The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `std` (aclTensor*, computation input): tensor for generating the standard deviation of random number distribution, which is an aclTensor on the device. The data type can be FLOAT16, FLOAT, or DOUBLE. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `mean`. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `mean`. The shape cannot exceed eight dimensions. The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `seed` (int64_t, computation output): seed for sampling the pseudo-random number generator. The data type is INT64.

  - `offset` (int64_t, computation output): offset for sampling the pseudo-random number generator. The data type is INT64.

  - `out` (aclTensor*, computation output): output tensor, which is an aclTensor on the device. The data type can be FLOAT, FLOAT16, or DOUBLE, and must be convertible from that after deduction between `mean` and `std`. The shape must be the same as that after broadcasting between `mean` and `std`. The [data format](../../../docs/en/context/data_format.md) can be ND.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input mean, std, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of the input mean, std, and out are not supported.
                                       2. The shapes of mean and std cannot be broadcast.
                                       3. The shape of mean, std, or out exceeds eight dimensions.
                                       4. After the shapes of mean and std are broadcast, the shapes are not equal to the shape of out.
                                       5. The data types of mean and std cannot be deduced.
                                       6. The data types deduced from mean and std cannot be converted to the type of out.
  ```

## aclnnNormalTensorTensor

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnNormalTensorTensorGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnNormalTensorTensor` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_normal_out.h"

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

  // Calculate the strides of consecutive tensors.
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
    // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> meanShape = {1, 4};
  std::vector<int64_t> stdShape = {1, 4};
  std::vector<int64_t> outShape = {1, 4};
  void* meanDeviceAddr = nullptr;
  void* stdDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* std = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> meanHostData = {1.1, 1.2, 1.3, 1.4};
  std::vector<float> stdHostData = {0.5, 0.6, 0.4, 0.5};
  std::vector<float> outHostData = {0.0, 0.0, 0.0, 0.0};
  int64_t seed = 1;
  int64_t offset = 1;

  // Create a mean aclTensor.
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an std aclTensor.
  ret = CreateAclTensor(stdHostData, stdShape, &stdDeviceAddr, aclDataType::ACL_FLOAT, &std);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNormalTensorTensor.
  ret = aclnnNormalTensorTensorGetWorkspaceSize(mean, std, seed, offset, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormalTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNormalTensorTensor.
  ret = aclnnNormalTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormalTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
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

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(mean);
  aclDestroyTensor(std);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(meanDeviceAddr);
  aclrtFree(stdDeviceAddr);
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
