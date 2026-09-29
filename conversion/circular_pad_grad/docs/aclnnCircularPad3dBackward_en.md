# aclnnCircularPad3dBackward

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/circular_pad_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Description: Performs backward propagation of [aclnnCircularPad3d](../../../conversion/circular_pad/docs/aclnnCircularPad3d_en.md).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnCircularPad3dBackwardGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnCircularPad3dBackward` is called to perform computation.

- `aclnnStatus aclnnCircularPad3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnCircularPad3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnCircularPad3dBackwardGetWorkspaceSize

- **Parameters**

  - `gradOutput` (aclTensor*, computation input): gradient data input during backward propagation, `aclTensor` on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32 and must be the same as that of `self`. The shape must be 4D or 5D and must be identical to that of the output of `aclnnCircularPad3d`. The number of dimensions must be the same as those of `self` and `gradInput`. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `self` (aclTensor*, computation input): original input data to be padded during forward propagation, `aclTensor` on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be 4D or 5D and must be identical to that of `gradInput`. The number of dimensions must be the same as those of `gradOutput` and `gradInput`. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `padding` (aclIntArray*, computation input): padding dimension in forward propagation, `aclIntArray` on the device. The shape is 1D, the data type is INT64, the [data format](../../../docs/en/context/data_format.md) can be ND, and the length is 6. The values represent the padding values on the left, right, top, bottom, front, and back respectively. The first two values of `padding` must be less than the size of the last dimension of `self`. The middle two values must be less than the size of the second-to-last dimension of `self`. The last two values must be less than the size of the third-to-last dimension of `self`.
  - `gradInput` (aclTensor*, computation output): gradient data output during backward propagation, `aclTensor` on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32 and must be the same as that of `self`. The shape must be 4D or 5D and must be identical to that of `self`. The number of dimensions must be the same as those of `gradOutput` and `self`. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The tensor is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of gradOutput, self, padding, or gradInput is not supported.
                                    2. The input shape of gradOutput, self, padding, or gradInput is not supported.
                                    3. self is an empty tensor and the size of a dimension other than the first dimension is 0.
                                    4. The value in padding is greater than or equal to the dimension size of self.
                                    5. The shape of gradOutput must be identical to that of the output of aclnnCircularPad3d.
  ```

## aclnnCircularPad3dBackward

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnCircularPad3dBackwardGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns** 

- Deterministic computation:
  - `aclnnCircularPad3dBackward` defaults to a deterministic implementation.

- The size of the last dimension of `gradOutput` must meet the following requirements:
  - FLOAT16/BFLOAT16: (0, 16362)
  - FLOAT32: (0, 24544)

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_circular_pad3d_backward.h"
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
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute strides of the contiguous tensor.
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
    // Set the device ID based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> gradOutputShape = {1, 1, 4, 4, 4};
    std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
    std::vector<int64_t> gradInputShape = {1, 1, 2, 2, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutputHostData(64);
    for (int64_t i = 0; i < 64; i++) {
        gradOutputHostData[i] = 1;
    }
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> paddingData = {1, 1, 1, 1, 1, 1};
    std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create a gradOutput aclTensor.
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a padding aclIntArray.
    padding = aclCreateIntArray(paddingData.data(), 6);
    CHECK_RET(padding != nullptr, return ret);
    // Create a gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnCircularPad3dBackward.
    ret = aclnnCircularPad3dBackwardGetWorkspaceSize(gradOutput, self, padding, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCircularPad3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnCircularPad3dBackward.
    ret = aclnnCircularPad3dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCircularPad3dBackward failed. ERROR: %d\n", ret); return ret);
    
    // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
    
    // 7. Release device resources.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0){
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
