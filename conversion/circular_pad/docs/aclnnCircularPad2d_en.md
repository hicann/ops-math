# aclnnCircularPad2d

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/circular_pad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Pads the last two dimensions of the input tensor by using circular padding.
- Example:

  ```text
  - Input: tensor([[[[0,1,2],
    [3,4,5],
    [6,7,8]]]])
  - padding([2,2,2,2])
  - Output: ([[[[4,5,3,4,5,3,4],
    [7,8,6,7,8,6,7],
    [1,2,0,1,2,0,1],
    [4,5,3,4,5,3,4],
    [7,8,6,7,8,6,7],
    [1,2,0,1,2,0,1],
    [4,5,3,4,5,3,4]]]])
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnCircularPad2dGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnCircularPad2d` is called to perform computation.

- `aclnnStatus aclnnCircularPad2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnCircularPad2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnCircularPad2dGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, computation input): original input data to be padded, `aclTensor` on the device. The shape must be 3D or 4D. The data type can be FLOAT16, BFLOAT16, FLOAT32, INT8, or INT32. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `padding` (aclIntArray*, computation input): dimensions to be padded in the input, `aclIntArray` on the host. The shape must be 1D, the data type is INT64, and the [data format](../../../docs/en/context/data_format.md) can be ND. The length is 4, and the values represent the padding values on the left, right, top, and bottom, respectively. The first two values of `padding` must be less than the size of the last dimension of `self`, and the last two values must be less than the size of the second-to-last dimension of `self`.
  - `out` (aclTensor*, computation output): output result after padding, `aclTensor` on the device. The shape must be 3D or 4D. The data type can be FLOAT16, BFLOAT16, FLOAT32, INT8, or INT32. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The size of the second-to-last dimension of `out` is equal to the size of the second-to-last dimension of `self` plus the last two values of `padding`. The size of the last dimension of `out` is equal to the size of the last dimension of `self` plus the first two values of `padding`.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, padding, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or out is not supported.
                                    2. The data types of self and out are inconsistent.
                                    3. The input shape of self, padding, or out is not supported.
                                    4. self is an empty tensor, and the values of the last three dimensions of self are 0.
                                    5. The value of padding is greater than or equal to the size of the corresponding dimension of self.
                                    6. The size of the last two dimensions of out is not equal to the size of the last two dimensions of self plus the corresponding padding.
  ```

## aclnnCircularPad2d

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnCircularPad2dGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnCircularPad2d` defaults to a deterministic implementation.

- The size of the last dimension of `out` must meet the following requirements:
  - INT8: (0, 98176)
  - FLOAT16/BFLOAT16: (0, 49088)
  - INT32/FLOAT32: (0, 24544)

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_circular_pad2d.h"
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
    std::vector<int64_t> selfShape = {1, 1, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 4, 4};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData = {1, 2, 3, 4};
    std::vector<int64_t> paddingData = {1, 1, 1, 1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a padding aclIntArray.
    padding = aclCreateIntArray(paddingData.data(), 4);
    CHECK_RET(padding != nullptr, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnCircularPad2d.
    ret = aclnnCircularPad2dGetWorkspaceSize(self, padding, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCircularPad2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnCircularPad2d.
    ret = aclnnCircularPad2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCircularPad2d failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(self);
    aclDestroyIntArray(padding);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the code based on the API definition.
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
