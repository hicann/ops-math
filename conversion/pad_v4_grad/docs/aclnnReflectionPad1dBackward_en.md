# aclnnReflectionPad1dBackward

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/pad_v4_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    √    |

## Function

- Description: Performs backward propagation of reflection_pad1d. For details about the forward propagation, see [aclnnReflectionPad1d](../../../conversion/mirror_pad/docs/aclnnReflectionPad1d_en.md).
- Example:

  ```text
  Input: gradOutput([[1, 1, 1, 1, 1]])
  self([[0, 1, 2]])
  padding([1, 1])
  Output: ([[1, 3, 1]])
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnReflectionPad1dBackwardGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnReflectionPad1dBackward` is called to perform computation.

- `aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnReflectionPad1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnReflectionPad1dBackwardGetWorkspaceSize

- **Parameters:**

  - `gradOutput` (aclTensor*, computation input): input gradient, `aclTensor` on the device. The data type can be BFLOAT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, or COMPLEX128, which is the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape supports 2D or 3D, with dimensions the same as those of `self` and `gradInput`. The shape must be the same as that of output in forward propagation `reflection_pad1d`.
  - `self` (aclTensor*, computation input): tensor to be padded, aclTensor on the device. The data type can be BFLOAT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape supports 2D or 3D, with dimensions the same as those of `gradOutput` and `gradInput`. The shape must be the same as that of `gradInput`.
  - `padding` (aclIntArray*, computation input): padding range, aclIntArray array on the device. The data type is INT64 and the length is 2. The two values of `padding` must be less than the value of the last dimension of `self`.
  - `gradInput` (aclTensor*, computation output): gradient of computed `self`, whose data type can be BFLOAT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, or COMPLEX128, which is the same as that of `self`. The shape is the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. `gradOutput`, `self`, `padding`, or `gradInput` is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of `gradOutput`, `self`, `padding`, or `gradInput` is not supported.
                           2. The input shape of `gradOutput`, `self`, `padding`, or `gradInput` is not supported.
                           3. `self` is an empty tensor, and the values of the last two dimensions of `self` are 0.
                           4. The value in `padding` is greater than or equal to the dimensions of `self`.
                           5. The shape of `gradOutput` must be the same as the output of reflection_pad1d forward propagation.
  ```

## aclnnReflectionPad1dBackward

- **Parameters:**

  - `workspace` (void*, input): memory address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnReflectionPad1dBackwardGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnReflectionPad1dBackward` defaults to a deterministic implementation.

If the number of elements in `gradOutput` is greater than 300\*1024\*1024, the execution may time out.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_reflection_pad1d_backward.h"
#include <iostream>
#include <vector>

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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> gradOutputShape = {1, 4};
    std::vector<int64_t> selfShape = {1, 2};
    std::vector<int64_t> gradInputShape = {1, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutputHostData = {1, 1, 1, 1};
    std::vector<float> selfHostData = {1, 2};
    std::vector<int64_t> paddingData = {1, 1};
    std::vector<float> gradInputHostData = {0, 0};
    // Create a gradOutput aclTensor.
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a padding aclIntArray.
    padding = aclCreateIntArray(paddingData.data(), 2);
    CHECK_RET(padding != nullptr, return ret);
    // Create a gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnReflectionPad1dBackward.
    ret = aclnnReflectionPad1dBackwardGetWorkspaceSize(gradOutput, self, padding, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReflectionPad1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnReflectionPad1dBackward.
    ret = aclnnReflectionPad1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReflectionPad1dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensors. Modify the code based on the API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(self);
    aclDestroyIntArray(padding);
    aclDestroyTensor(gradInput);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
