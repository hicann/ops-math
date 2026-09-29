# aclnnStdMeanCorrection

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/reduce_std_with_mean)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |   ×     |
| <term>Atlas training products</term>                             |   √     |

## Function

- Description: Computes the standard deviation and mean of the sample.
- Formula:
  If `dim` is $i$, the dimension is computed. $N$ is the shape. $self_{i}$ is used to compute the average value $\bar{x_{i}}$ in the dimension.

  $$
  \left\{
  \begin{array} {rcl}
  meanOut& &= \bar{x_{i}}\\
  stdOut& &= \sqrt{\frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2}
  \end{array}
  \right.
  $$

  If `keepdim = true`, the dimension is retained after reduction, and the dimension value in the output shape is 1. If `keepdim = false`, the dimension is not retained.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnStdMeanCorrectionGetWorkspaceSize` is called to obtain the input, the workspace size required for computation, and the executor that contains the operator computation process. Then, `aclnnStdMeanCorrection` is called to perform computation.

- `aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim, aclTensor* stdOut, aclTensor* meanOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnStdMeanCorrection(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnStdMeanCorrectionGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor\*, computation input): `self` in the formula, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, BFLOAT16, or FLOAT16.
  - `dim` (aclIntArray*, computation input): `dim` in the formula. It is aclIntArray on the host, indicating the dimension involved in computation. The value range is [-self.dim(), self.dim()-1], and the data must be unique. The data type can be INT64. When `dim` is nullptr or [], all dimensions are calculated.
  - `correction` (int64_t, calculation input): $\delta N$ value in the formula, which is an integer on the host side.
  - `keepdim` (bool, computation input): `keepdim` in the formula, Boolean value on the host, whether to retain the dimension in the output tensor.
  - `stdOut` (aclTensor\*, computation output): `stdOut` in the formula, aclTensor on the device. [non-contiguous tensor] (../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, BFLOAT16, or FLOAT16.
  - `meanOut` (aclTensor\*, computation output): `meanOut` in the formula, aclTensor on the device. [non-contiguous tensor] (../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, BFLOAT16, or FLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_INVALID): 1. self, stdOut, or meanOut is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, stdOut, or meanOut is not supported.
                                        2. The dimension in the dim array is out of the dimension range of self.
                                        3. Duplicate elements exist in the dim array.
                                        4. An error occurs when the shape of stdOut is as follows:
                                          If keepdim is set to true, stdOut.shape != self.shape (the shape when the specified dim is set to 1).
                                          If keepdim is set to false, stdOut.shape != self.shape (the shape after the specified dim is removed).
                                        5. If the shape of meanOut is as follows, an error occurs:
                                          If keepdim is set to true, meanOut.shape != self.shape (the shape when the specified dim is set to 1).
                                          If keepdim is set to false, meanOut.shape != self.shape (the shape after the specified dim is removed).
  ```

## aclnnStdMeanCorrection

- **Parameters**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnStdMeanCorrectionGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnStdMeanCorrection` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_std_mean_correction.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> stdOutShape = {2, 4};
  std::vector<int64_t> meanOutShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* stdOutDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* stdOut = nullptr;
  aclTensor* meanOut = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24};
  std::vector<float> stdOutHostData = {1, 2, 3, 4, 5, 6, 7, 8.0};
  std::vector<float> meanOutHostData = {1, 2, 3, 4, 5, 6, 7, 8.0};
  std::vector<int64_t> dimData = {1};
  int64_t correction = 1;
  bool keepdim = false;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a stdOut aclTensor.
  ret = CreateAclTensor(stdOutHostData, stdOutShape, &stdOutDeviceAddr, aclDataType::ACL_FLOAT, &stdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const aclIntArray *dim = aclCreateIntArray(dimData.data(), dimData.size());
  CHECK_RET(dim != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnStdMeanCorrection.
  ret = aclnnStdMeanCorrectionGetWorkspaceSize(self, dim, correction, keepdim, stdOut, meanOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStdMeanCorrectionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnStdMeanCorrection.
  ret = aclnnStdMeanCorrection(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStdMeanCorrection failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(stdOutShape);
  std::vector<float> stdResultData(size, 0);
  ret = aclrtMemcpy(stdResultData.data(), stdResultData.size() * sizeof(stdResultData[0]), stdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("stdResultData[%ld] is: %f\n", i, stdResultData[i]);
  }

  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(stdOut);
  aclDestroyTensor(meanOut);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(stdOutDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
