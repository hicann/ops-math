# aclnnAminmaxDim

## Supported Products
- Ascend 910 AI Processor
- Ascend 910B AI Processor
- Ascend 910_93 AI Processor

## Prototype
Each operator has [two-phase API](common/two_phase_api.md) calls. First, aclnnAminmaxDimGetWorkspaceSize is called to obtain the input parameters and compute the required workspace size based on the process. Then, aclnnAminmaxDim is called to perform computation.

* `aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor *self, const int64_t dim, bool keepDim, aclTensor *minOut, aclTensor *maxOut, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnAminmaxDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## Function

Returns the minimum and maximum values of the input tensor in the specified dimension.

## aclnnAminmaxDimGetWorkspaceSize

- **Parameters**:

  * self (aclTensor*, compute input): aclTensor on the device. [Non-contiguous tensors](common/non_contiguous_tensors.md) are supported. The [data format](common/data_format.md) can be ND.
     * Ascend 910B AI Processor, Ascend 910_93 AI Processor: FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL
     * Ascend 910 AI Processor: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL
  * dim (const int64_t compute input): INT64 on the host, which specifies the dimension to be reduced. The value range is [–self.dim(), self.dim()].
  * keepDim (bool, compute input): BOOL on the host, whether to retain the dimension of the reduced axis.
  * minOut (aclTensor\*, compute output): aclTensor on the device. The data type must be the same as that of self. [Non-contiguous tensors](common/non_contiguous_tensors.md) are supported. The [data format](common/data_format.md) can be ND.
     * Ascend 910B AI Processor, Ascend 910_93 AI Processor: FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL
     * Ascend 910 AI Processor: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL
  * maxOut (aclTensor\*, compute output): aclTensor on the device. The data type must be the same as that of self. [Non-contiguous tensors](common/non_contiguous_tensors.md) are supported. The [data format](common/data_format.md) can be ND.
     * Ascend 910B AI Processor, Ascend 910_93 AI Processor: FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL
     * Ascend 910 AI Processor: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL
  * workspaceSize (uint64_t\*, output): size of the workspace to be allocated on the device.
  * executor (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

```
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input self, minOut, or maxOut is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                  2. The data type of minOut or maxOut is different from that of self.
                                  3. The shape of self, minOut, or maxOut exceeds eight dimensions.
                                  4. The value of dim is out of the dimension range of self.
                                  5. The axis specified by dim is empty.
```

## aclnnAminmaxDim
- **Parameters**:

  * workspace (void\*, input): address of the workspace to be allocated on the device.
  * workspaceSize (uint64_t, input): workspace size allocated on the device, which is obtained by the first-phase API aclnnAminmaxDimGetWorkspaceSize.
  * executor (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * stream (aclrtStream, input): AscendCL stream for executing the task.

- **Returns**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

## Constraints
None

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](common/compilation_running_sample.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_aminmax_dim.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Use CHECK as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {2, 3, 2};
  std::vector<int64_t> outShape = {1, 3, 2};
  void* selfDeviceAddr = nullptr;
  void* minOutDeviceAddr = nullptr;
  void* maxOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* minOut = nullptr;
  aclTensor* maxOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> minOutHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> maxOutHostData = {0, 0, 0, 0, 0, 0};
  int64_t dim = 0;
  bool keepDim = true;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(minOutHostData, outShape, &minOutDeviceAddr, aclDataType::ACL_FLOAT, &minOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maxOutHostData, outShape, &maxOutDeviceAddr, aclDataType::ACL_FLOAT, &maxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnAminmaxDim API.
  ret = aclnnAminmaxDimGetWorkspaceSize(self, dim, keepDim, minOut, maxOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAminmaxDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAminmaxDim.
  ret = aclnnAminmaxDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAminmaxDim failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> minResultData(size, 0);
  ret = aclrtMemcpy(minResultData.data(), minResultData.size() * sizeof(minResultData[0]), minOutDeviceAddr,
                    size * sizeof(minResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, minResultData[i]);
  }
  std::vector<float> maxResultData(size, 0);
  ret = aclrtMemcpy(maxResultData.data(), maxResultData.size() * sizeof(maxResultData[0]), maxOutDeviceAddr,
                    size * sizeof(maxResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, maxResultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(minOut);
  aclDestroyTensor(maxOut);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(minOutDeviceAddr);
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
