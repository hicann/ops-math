# aclnnRoundDecimals&aclnnInplaceRoundDecimals

## Supported Products

- Ascend 910 AI Processor
- Ascend 910B AI Processor
- Ascend 910_93 AI Processor

## Prototype

- **aclnnRoundDecimals** and **aclnnInplaceRoundDecimals** implement the same function in different ways. Select a proper operator based on your requirements.

  - **aclnnRoundDecimals**: An output tensor object needs to be created to store the computation result.
  - **aclnnInplaceRoundDecimals**: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](common/two_phase_api.md) calls. First, **aclnnRoundDecimalsGetWorkspaceSize** or **aclnnInplaceRoundDecimalsGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRoundDecimals** or **aclnnInplaceRoundDecimals** is called to perform computation.

  - `aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(const aclTensor* self, int64_t decimals, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

  - `aclnnStatus aclnnRoundDecimals(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

  - `aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(aclTensor* selfRef, int64_t decimals, uint64_t* workspaceSize, aclOpExecutor** executor)`

  - `aclnnStatus aclnnInplaceRoundDecimals(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## Function

Description: Rounds the elements of the input tensor to the specified number of decimal places.

## aclnnRoundDecimalsGetWorkspaceSize

- **Parameters**:

  * self (aclTensor*, compute input): input tensor, aclTensor on the device. [Non-contiguous tensors](common/non_contiguous_tensors.md) are supported. The [data format](common/data_format.md) supports ND. The number of dimensions cannot exceed 8, and the shape must be the same as that of out.
    - Ascend 910 AI Processor: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, or INT64.
    - Ascend 910B AI Processor, Ascend 910_93 AI Processor: The data type can be FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, or INT64.
  * decimals (int64_t, compute input): number of decimal places to round.

  * out (aclTensor *, compute output): output tensor, aclTensor on the device. [Non-contiguous tensors](common/non_contiguous_tensors.md) are supported. The [data format](common/data_format.md) supports ND. The number of dimensions cannot exceed 8, and the shape must be the same as that of self.
    - Ascend 910 AI Processor: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, or INT64.
    - Ascend 910B AI Processor, Ascend 910_93 AI Processor: The data type can be FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, or INT64.
  * workspaceSize (uint64_t *, output): size of the workspace to be allocated on the device.

  * executor (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.


- **Returns**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

```
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 ACLNN_ERR_PARAM_NULLPTR: 1. The input self or out is a null pointer.
161002 ACLNN_ERR_PARAM_INVALID: 1. The data type of self or out is not supported.
                         2. The data types of self and out are inconsistent.
                         3. The shapes of self and out are inconsistent.
                         4. The dimensions of self or out are greater than 8.
```

## aclnnRoundDecimals

- **Parameters**:

  * workspace (void *, input): address of the workspace memory allocated on the device.

  * workspaceSize (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnRoundDecimalsGetWorkspaceSize.

  * executor (aclOpExecutor *, input): operator executor, containing the operator computation process.

  * stream (aclrtStream, input): AscendCL stream for executing the task.


- **Returns**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

## Constraints

- When decimals is not 0:
  If the input data exceeds the range of (–347000, 347000), the precision may be affected.

## aclnnInplaceRoundDecimalsGetWorkspaceSize

- **Parameters**:

  * selfRef (aclTensor*, compute input): input tensor, aclTensor on the device. [Non-contiguous tensors](common/non_contiguous_tensors.md) are supported. The [data format](common/data_format.md) supports ND. The number of dimensions cannot exceed 8.
    - Ascend 910 AI Processor: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, or INT64.
    - Ascend 910B AI Processor, Ascend 910_93 AI Processor: The data type can be FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, or INT64.
  * decimals(int64_t, compute input): Specifies the number of digits to be rounded off.

  * workspaceSize (uint64_t *, output): size of the workspace to be allocated on the device.

  * executor (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.


- **Returns**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

```
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 ACLNN_ERR_PARAM_NULLPTR: 1. The input selfRef is a null pointer.
161002 ACLNN_ERR_PARAM_INVALID: 1. The data type of selfRef is not supported.
                         2. The shape of selfRef is greater than 8D.
```

## aclnnInplaceRoundDecimals

- **Parameters**:

  * workspace (void *, input): address of the workspace memory allocated on the device.

  * workspaceSize (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnInplaceRoundDecimalsGetWorkspaceSize.

  * executor (aclOpExecutor *, input): operator executor, containing the operator computation process.

  * stream (aclrtStream, input): AscendCL stream for executing the task.


- **Returns**:

  aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

## Constraints

- When decimals is not 0:
  If the input data exceeds the range of (–347000, 347000), the precision may be affected.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](common/compilation_running_sample.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_round.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int decimals = 0;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoundDecimals.
  ret = aclnnRoundDecimalsGetWorkspaceSize(self, decimals, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoundDecimalsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRoundDecimals.
  ret = aclnnRoundDecimals(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoundDecimals failed. ERROR: %d\n", ret); return ret);

    
    
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceRoundDecimals.
  ret = aclnnInplaceRoundDecimalsGetWorkspaceSize(self, decimals, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRoundDecimalsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceRoundDecimals.
  ret = aclnnInplaceRoundDecimals(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRoundDecimals failed. ERROR: %d\n", ret); return ret);

    
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
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
