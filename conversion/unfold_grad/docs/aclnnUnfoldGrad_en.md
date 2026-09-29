# aclnnUnfoldGrad

## Supported Products

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/unfold_grad)

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |   ×     |
| <term>Atlas training products</term>                             |   ×    |

## Function

Description: Performs backpropagation of the Unfold operator and computes its gradient.

The Unfold operator computes all slices whose size is `$size$` in dimension `$dim$` based on the input parameter `self`. The step between two slices is given by `$step$`. If `$sizedim$` is the size of dimension `$dim$` of the input parameter self, the size of dimension `$dim$` in the returned tensor is $(sizedim – size)/step + 1$. An additional dimension whose size is `$size$` is added to the returned tensor.

The shape of the input `gradOut` of the UnfoldGrad operator is the shape of the forward output of the Unfold operator. The shape of the input `inputSizes` is the shape of the forward input `self` of the Unfold operator. The shape of the output `gradIn` of the UnfoldGrad operator is the shape of the forward input `self` of the Unfold operator.

Examples:

```Python
>>> x = torch.arange(1., 8)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
>>> x.unfold(0, 2, 1)
tensor([[ 1.,  2.],
        [ 2.,  3.],
        [ 3.,  4.],
        [ 4.,  5.],
        [ 5.,  6.],
        [ 6.,  7.]])
>>> x.unfold(0, 2, 2)
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.]])
>>> res = torch.ops.aten.unfold_backward(grad, [7], 0, 2, 2)
tensor([1, 2, 3, 4, 5, 6, 0])
```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnUnfoldGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnUnfoldGrad` is called to perform computation.

- `aclnnStatus aclnnUnfoldGradGetWorkspaceSize(const aclTensor *gradOut, const aclIntArray *inputSizes, int64_t dim, int64_t size, int64_t step, const aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUnfoldGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnUnfoldGradGetWorkspaceSize

- **Parameters**

  - `gradOut` (aclTensor*, compute input): `aclTensor` on the device, indicating the gradient update coefficient. The shape is (..., (sizedim – size)/step + 1, size). The `dim` dimension of `gradOut` must be equal to $(inputSizes[dim] – size)/step + 1$, and the size of `gradOut` must be equal to size of inputSizes plus 1. The data type can be FLOAT, FLOAT16, or BFLOAT16. The [data format] (../../../docs/en/context/data_format.md) can be ND.
  - `inputSizes` (aclIntArray*, compute input): `aclIntArray` on the host, indicating the shape of the output tensor. The value is (..., sizedim). The size of `inputSizes` is less than or equal to 8. The data type can be INT64. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `dim` (int64_t, compute input): `$dim$` in the formula. Dimension. `$dim$` must be greater than or equal to 0 and less than the size of `inputSizes`.
  - `size` (int64_t, compute input): `$size$` in the formula. Size of each slice. `$size$` must be greater than 0 and less than or equal to the `dim` dimension of `inputSizes`.
  - `step` (int64_t, compute input): `$step$` in the formula. Indicates the step between slices. The value of `$step$` must be greater than 0.
  - `gradIn` (aclTensor*, computation output): gradient of Unfold. It is `aclTensor` on the device. The shape is `inputSizes`. The data type can be FLOAT, FLOAT16, or BFLOAT16, and must be the same as that of gradOut.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  `161001` (ACLNN_ERR_PARAM_NULLPTR): 1. The input or output tensor is a null pointer.
  `161002` (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of the input or output is not supported.
  `561002` (ACLNN_ERR_INNER_TILING_ERROR): 1. The `dim` dimension of `gradOut` is not equal to (inputSizes[dim] – size)/step + 1.
                                              2. The size of `gradOut` is not equal to the size of `inputSizes` plus 1.
                                              3. The value of `dim` is less than 0 or greater than or equal to the size of `inputSizes`.
                                              4. The value of `size` is less than or equal to 0 or greater than the `dim` dimension of `inputSizes`.
                                              5. The value of `step` is less than or equal to 0.
  ```

## aclnnUnfoldGrad

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnUnfoldGradGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnUnfoldGrad` defaults to a deterministic implementation.

1. The shape of `gradOut` must meet the following constraints:
    (1) The `dim` dimension of `gradOut` is equal to (inputSizes[dim] – size)/step + 1.
    (2) The `size` of `gradOut` is equal to the size of `inputSizes` plus 1.
2. Requirements for `dim`, `size`, and `step`:
    (1) `dim` must be greater than or equal to 0 and less than the `size` of `inputSizes`.
    (2) `size` must be greater than 0 and less than or equal to the `dim` dimension of `inputSizes`.
    (3) `step` is greater than 0.
    (4) `step` and `size` are less than or equal to 100.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unfold_grad.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> gradOutShape = {3, 2, 3};
  std::vector<int64_t> gradInShape = {8, 2};

  void* gradOutDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* gradIn = nullptr;

  std::vector<float> gradOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<int64_t> inputSizesData = {8, 2};
  std::vector<float> gradInHostData(16, 0);

  // Create a gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an aclIntArray.
  auto inputSizes = aclCreateIntArray(inputSizesData.data(), 2);
  CHECK_RET(inputSizes != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of `aclnnUnfoldGrad`.
  ret = aclnnUnfoldGradGetWorkspaceSize(gradOut, inputSizes, 0, 3, 2, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnfoldGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed `workspaceSize`.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of `aclnnUnfoldGrad`.
  ret = aclnnUnfoldGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnfoldGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release `aclTensor` and `aclIntArray`. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOut);
  aclDestroyIntArray(inputSizes);
  aclDestroyTensor(gradIn);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
