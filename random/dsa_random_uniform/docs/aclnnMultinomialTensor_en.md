# aclnnMultinomialTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/random/dsa_random_uniform)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Operator function: Draws numsamples samples from the input tensor based on the probability distribution of each object, and stores the indices of these samples in the output tensor.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMultinomialTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMultinomialTensor` is called to perform computation.

- `aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor* self, int64_t numsamples, bool replacement, const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMultinomialTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMultinomialTensorGetWorkspaceSize

- **Parameters:**
  
  - `self` (aclTensor*, computation input): input tensor, indicating the probability distribution of each object. It is the aclTensor on the device. The shape is (N, C) or (C). The value of `self` must be greater than or equal to 0, and the dimensions of `self` and `out` must be the same. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The data type can be BFLOAT16, FLOAT16, FLOAT, or DOUBLE.
  - `numsamples` (int64_t, computation input): integer on the host, indicating the number of samples to draw from each multinomial distribution. The value of `numsamples` is a non-negative integer. When `replacement` is false, `numsamples` is not greater than C.
  - `replacement` (bool, computation input): boolean type on the host, indicating whether to draw with replacement or not
  - `seedTensor` (aclTensor*, computation input): aclTensor on the device. The shape is [1]. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND, and the data type can be INT64. It can be used to set the seed value of the random number generator, which affects the generated random number sequence.
  - `offsetTensor` (aclTensor*, computation input): aclTensor on the device. The shape is [1]. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND, and the data type can be INT64. The sum of this value and the scalar offset is used as the offset of the random number operator. It indicates the offset of the random number, which affects the position of the generated random number sequence. After the offset is set, the generated random number sequence starts from the specified position.
  - `offset` (int64_t, computation input): integer on the host, which is used as the accumulation of offsetTensor.
  - `out` (aclTensor*, computation output): aclTensor on the device. The data type can be INT64. The shape is (N, numsamples) or (numsamples). The dimensions of `self` and `out` are the same. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                        2. The data type of out is not INT64.
                                        3. The shape of self is not 1D or 2D.
                                        4. The shapes of self and out are inconsistent.
                                        5. The value of numsamples is less than or equal to 0.
                                        6. replacement is false and numsamples is greater than the size of the last dimension of self.
                                        7. The size of the last dimension of self cannot exceed 2^24.
                                        8. The size of the last dimension of out must be the same as that of numsamples.
                                        9. When self is a 2D tensor, the first dimension of out must have the same size as the first dimension of self.
  ```

## aclnnMultinomialTensor

- **Parameters:**
  
  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnMultinomialTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMultinomialTensor` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multinomial.h"

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
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> outShape = {4};
  std::vector<int64_t> seedShape = {1};
  std::vector<int64_t> offsetShape = {1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  void* seedDeviceAddr = nullptr;
  aclTensor* seed = nullptr;
  void* offsetDeviceAddr = nullptr;
  aclTensor* offset = nullptr;
  int64_t offset2 = 102;
  std::vector<float> selfHostData = {0, 10, 3, 0, 1, 2, 1, 1};
  std::vector<int64_t> outHostData = {2, 0, 1, 2};
  std::vector<int64_t> seedHostData = {0};
  std::vector<int64_t> offsetHostData = {392};
  int64_t numsamples = 4;
  bool replacement = false;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a seed aclTensor.
  ret = CreateAclTensor(seedHostData, seedShape, &seedDeviceAddr, aclDataType::ACL_INT64, &seed);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_INT64, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMultinomialTensor.
  ret = aclnnMultinomialTensorGetWorkspaceSize(self, numsamples, replacement, seed, offset, offset2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultinomialTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMultinomialTensor.
  ret = aclnnMultinomialTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultinomialTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(seed);
  aclDestroyTensor(offset);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(seedDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
