# aclnnVarMean

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/reduce_var)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     ×     |
| <term>Atlas inference products</term>                            |   ×     |
| <term>Atlas training products</term>                             |   √     |

## Function

- Description: Returns the mean and variance of the dimension specified by the input tensor.
- Calculation formula: If `dim` is `$i$`, the dimension is calculated. `$N$` is the shape. Take `$self_{i}$` and calculate the mean value `$meanOut = \bar{self_{i}}$` in this dimension.
  The formula for computing the variance is as follows:

  $$
  varOut = \frac{1}{max(0, N - correction)}\sum_{j=0}^{N-1}(self_{ij}-\bar{self_{i}})^2
  $$

  If `keepdim = true`, the dimension is retained after the reduce operation, and the value of the dimension in the output shape is 1. If `keepdim = false`, the dimension is not retained.
  When `dim` is `nullptr` or `[]`, all dimensions are calculated.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnVarMeanGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnVarMean` is called to perform computation.

  - `aclnnStatus aclnnVarMeanGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, int64_t correction, bool keepdim, aclTensor* varOut, aclTensor* meanOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnVarMean(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnVarMeanGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, compute input): input `self` in the formula. The shape supports 0 to 8 dimensions. The data types of `self`, `meanOut`, and `varOut` must meet the [type deduction rules](../../../docs/en/context/deduction_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16 or FLOAT.
  - `dim` (aclIntArray*, input): `dim` in the formula. It is `aclIntArray` on the host, indicating the dimension involved in computation. The value range is [-self.dim(), self.dim()-1], and the data must be unique. The supported data type is INT64. When `dim` is `nullptr` or `[]`, all dimensions are calculated.
  - `correction` (int64_t, input): input `correction` in the formula. The data type is int64_t.
  - `keepdim` (bool, input): whether to retain the dimension of the reduced axis. The data type is bool.
  - `meanOut` (aclTensor*, compute output): mean output result. The data types of `self` and `meanOut` must meet the [type deduction rules](../../../docs/en/context/deduction_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16 or FLOAT.
  - varOut (aclTensor*, compute output): output `varOut` in the formula. It is the variance calculation result. The data types of self and varOut must meet the [type deduction rules](../../../docs/en/context/deduction_relationship.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16 or FLOAT.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  `161001` (ACLNN_ERR_PARAM_NULLPTR): 1. The input `self`, `meanOut`, and `varOut` are null pointers.
  `161002` (ACLNN_ERR_PARAM_INVALID): 1. The data type of `self`, `meanOut`, and `varOut` is not supported.
                                        2. The shape of `self` exceeds eight dimensions.
                                        3. The value of `dim` is invalid. (The data in `dim` points to the same dimension, or `dim` exceeds the dimension range of `self`.)
                                        4. The shapes of `self`, `meanOut`, and `varOut` do not meet the deduction rules in the calculation formula.
  ```

## aclnnVarMean

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnVarMeanGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnVarMean` defaults to a deterministic implementation.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_var_mean.h"

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
  // Call `aclrtMalloc` to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call `aclrtMemcpy` to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call `aclCreateTensor` to create an aclTensor.
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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> selfShape = {2,4};
  std::vector<int64_t> outShape = {2,1};
  void* selfDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* varDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dim = nullptr;
  aclTensor* var = nullptr;
  aclTensor* mean = nullptr;
  std::vector<float> selfHostData = {0.0, 1.1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> dimData = {1};
  int64_t correction = 1;
  bool keepdim = true;
  std::vector<float> varHostData = {0.0, 0};
  std::vector<float> meanHostData = {0.0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a dim aclIntArray.
  dim = aclCreateIntArray(dimData.data(), dimData.size());
  CHECK_RET(dim != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(varHostData, outShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  ret = CreateAclTensor(meanHostData, outShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of `aclnnVarMean`.
  ret = aclnnVarMeanGetWorkspaceSize(self, dim, correction, keepdim, var, mean, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnVarMeanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed `workspaceSize`.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of `aclnnVarMean`.
  ret = aclnnVarMean(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnVarMean failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> meanData(size, 0);
  ret = aclrtMemcpy(meanData.data(), meanData.size() * sizeof(meanData[0]), meanDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResult[%ld] is: %f\n", i, meanData[i]);
  }
  std::vector<float> varData(size, 0);
  ret = aclrtMemcpy(varData.data(), varData.size() * sizeof(varData[0]), varDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("varResult[%ld] is: %f\n", i, varData[i]);
  }

  // 6. Release `aclTensor` and `aclScalar`. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(dim);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(meanDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
