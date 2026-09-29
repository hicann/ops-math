# aclnnAffineGrid

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/affine_grid)

## Supported Product Models

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    √     |

## Function Description

Description: Based on a given set of 3D affine transformation matrices (`theta`) and the desired output image size (`size`), generates a 2D or 3D grid. This grid represents the coordinates of points in the transformed image with respect to the original image.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAffineGridGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnAffineGrid` is called to perform computation.

- `aclnnStatus aclnnAffineGridGetWorkspaceSize(const aclTensor* theta, const aclIntArray* size, bool alignCorners, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAffineGrid(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAffineGridGetWorkspaceSize

- **Parameters:**
  
  - `theta` (aclTensor\*, computation input): aclTensor on the device, affine transformation parameter for controlling rotation, scaling, and translation. The shape is (N, 2, 3) or (N, 3, 4). It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT32 or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT32, FLOAT16, or BFLOAT16.
  - size (aclIntArray\*, computation input): aclIntArray on the host, size of the output image. The size is 4 (N, C, H, W) or 5 (N, C, D, H, W).
  - alignCorners (bool, computation input): whether to align corner pixels. If True, the output grid's corner pixels align with the input grid's corner pixels. If False, the output grid's center pixels align with the input grid's center pixels. Defaults to False.
  - out (aclTensor\*, computation output): coordinates of the affine image on the original image. The data type is the same as that of theta. It supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md). Its [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT32 or FLOAT16.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT32, FLOAT16, or BFLOAT16.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, covering the operator computation process.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input theta, size, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of theta is not supported.
                                        2. The data types of out and theta are inconsistent.
                                        3. The dimensionality of theta is not 3.
                                        4. The size of the size array is neither 4 nor 5.
                                        5. The shape of theta is not (N, 2, 3) when the size of the size array is 4, or not (N, 3, 4) when the size of the size array is 5.
                                        6. The shape of out is not (N, H, W, 2) when the size of the size array is 4, or not (N, D, H, W, 3) when the size of the size array is 5.
  ```

## aclnnAffineGrid

- **Parameters:**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnAffineGridGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, covering the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnAffineGrid` defaults to a deterministic implementation.

- The values of `N`, `H`, and `W` in `size` must be within the value range of (0, 100000].

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_affine_grid.h"

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
  // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Customize the error handling as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> thetaShape = {1, 2, 3};
  std::vector<int64_t> outShape = {1, 2, 3, 2};
  void* thetaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* theta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> thetaHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> sizeData = {1, 1, 2, 3};
  bool alignCorners = false;

  // Create a theta aclTensor.
  ret = CreateAclTensor(thetaHostData, thetaShape, &thetaDeviceAddr, aclDataType::ACL_FLOAT, &theta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a size aclIntArray.
  aclIntArray *size = aclCreateIntArray(sizeData.data(), sizeData.size());
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAffineGrid.
  ret = aclnnAffineGridGetWorkspaceSize(theta, size, alignCorners, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAffineGridGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnAffineGrid.
  ret = aclnnAffineGrid(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAffineGrid failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate code) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto length = GetShapeSize(outShape);
  std::vector<float> resultData(length, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    length * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < length; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(theta);
  aclDestroyIntArray(size);
  aclDestroyTensor(out);

  // 7. Free device resources. Modify the code based on the API definition.
  aclrtFree(thetaDeviceAddr);
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
