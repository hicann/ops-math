# aclnnMeanV2

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

Description: Computes the mean value of tensors on a specified dimension.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMeanV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMeanV2` is called to perform computation.

- `aclnnStatus aclnnMeanV2GetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, bool keepDim, bool noopWithEmptyAxes, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMeanV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMeanV2GetWorkspaceSize

- **Parameter description:**

  - `self` (aclTensor*, compute input): aclTensor on the device. If the input is an empty tensor, the output type cannot be COMPLEX64 or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, or COMPLEX128.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, or COMPLEX128.
  - dim (aclIntArray\*, input): aclIntArray on the host. The data type can be INT64. The value range is [–r, r – 1], where `r` is the dimension count of the input data.
  - `keepDim` (bool, input): whether to keep the reduced dimension. The data type is BOOL.
  - `noopWithEmptyAxes` (bool, input): behavior defined when dim is set to []. If this parameter is set to `false`, all axes are reduced when `dim` is set to []. If this parameter is set to `true`, all axes are retained when `dim` is set to [], and the output tensor is the same as the input tensor.
  - `out` (aclTensor*, compute output): aclTensor on the device. The data type must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, or COMPLEX128.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, or COMPLEX128.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, out, or dim is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                    2. The data format of self or out is not supported.
                                    3. The dimension range of the dim array exceeds that of the input tensor.
                                    4. Elements duplicate in the dim array.
  ```

## aclnnMeanV2

- **Parameter description:**

  - `workspace` (void \*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnMeanV2GetWorkspaceSize`.
  - `executor` (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnReduceMeanV2` defaults to a deterministic implementation.

## Example

For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mean.h"

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
  // (Fixed writing) Initialize resources.
  auto  ret = aclInit(nullptr);
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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, aclTensor** self, aclIntArray** dim,
    void** outDeviceAddr, aclTensor** out)
{
    std::vector<int64_t> selfHostData = {2, 3, 5, 8, 4, 12, 6, 7};
    std::vector<int64_t> outHostData = {2, 3, 5, 8};
    std::vector<int64_t> dimData = {1, 2};

    // Create a self aclTensor.
    auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_INT64, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dim aclIntArray.
    *dim = aclCreateIntArray(dimData.data(), 1);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_INT64, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

void ReleaseTensorAndIntArray(aclTensor* self, aclIntArray* dim, aclTensor* out)
{
    aclDestroyTensor(self);
    aclDestroyIntArray(dim);
    aclDestroyTensor(out);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* outDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {2, 2, 2};
  std::vector<int64_t> outShape = {2, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dim = nullptr;
  aclTensor* out = nullptr;
  
  ret = PrepareInputAndOutput(selfShape, outShape, &selfDeviceAddr, &self, &dim, &outDeviceAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  bool keepdim = false;
  bool noopWithEmptyAxes = true;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMeanV2.
  ret = aclnnMeanV2GetWorkspaceSize(self, dim, keepdim, noopWithEmptyAxes, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMeanV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnMeanV2.
  ret = aclnnMeanV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMeanV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
  ReleaseTensorAndIntArray(self, dim, out);

  // 7. Release device resources. Modify the code based on the API definition.
  ReleaseDevice(selfDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```
