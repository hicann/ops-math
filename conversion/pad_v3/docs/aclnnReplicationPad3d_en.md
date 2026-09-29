# aclnnReplicationPad3d

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/pad_v3)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Pads the last three dimensions of the input tensor using replication of the input boundary.
- Example:

```text
Input tensor ([[[[[0,1],
               [2,3]],
              [[4,5],
               [6,7]]]]])
padding([1,1,1,1,1,1])
Output:
([[[[[0,0,1,1],
[0,0,1,1],
[2,2,3,3],
[2,2,3,3]],
[[0,0,1,1],
[0,0,1,1],
[2,2,3,3],
[2,2,3,3]],
[[4,4,5,5],
[4,4,5,5],
[6,6,7,7],
[6,6,7,7]],
[[4,4,5,5],
[4,4,5,5],
[6,6,7,7],
[6,6,7,7]]]]])
```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnReplicationPad3dGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnReplicationPad3d` is called to perform computation.

- `aclnnStatus aclnnReplicationPad3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnReplicationPad3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnReplicationPad3dGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, compute input): original input data to be padded, aclTensor on the device. The shape supports 4 or 5 dimensions, and the data type can be BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8, COMPLEX64, or COMPLEX128. The [data format](../../../docs/en/context/data_format.md) supports ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.

  - `padding` (aclIntArray*, compute input): dimensions to be padded in the input. It is an aclIntArray on the host. The length is 6, and the values indicate respectively the padding values on the left, right, top, bottom, front, and back.

  - `out` (aclTensor*, compute output): output after padding, which is an aclTensor on the device. The shape and dtype must be the same as those of `self`. The data type supports BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8, COMPLEX64, and COMPLEX128. [data format](../../../docs/en/context/data_format.md). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. In the output shape, all dimensions except the last three dimensions to be padded must be the same. In the last three dimensions, the third to last dimension of out is equal to that of `self` plus the last two values of `padding`; the last but one dimension of `out` is equal to that of `self` plus the middle two values of `padding`; the last dimension of `out` is equal to that of `self` plus the first two values of `padding`.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input self, padding, and out are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or data format of self and out is not supported.
                                      2. The data types of self and out are inconsistent.
                                      3. The input shape of self, padding, or out is not supported.
                                      4. self or out is an empty tensor, and the values of the last four dimensions of self are 0.
                                      5. The values of padding are greater than or equal to the corresponding dimension values of self.
                                      6. The values of the last three dimensions of out are not equal to those of the last three dimensions of self plus corresponding padding.
  ```

## aclnnReplicationPad3d

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnReplicationPad3dGetWorkspaceSize.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnReplicationPad3d` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_replication_pad3d.h"
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
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 4, 4, 4};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> paddingData = {1, 1, 1, 1, 1, 1};
    std::vector<float> outHostData(GetShapeSize(outShape) * 2, 0);

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a padding aclIntArray.
    padding = aclCreateIntArray(paddingData.data(), 6);
    CHECK_RET(padding != nullptr, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnReplicationPad3d.
    ret = aclnnReplicationPad3dGetWorkspaceSize(self, padding, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReplicationPad3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnReplicationPad3d.
    ret = aclnnReplicationPad3d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReplicationPad3d failed. ERROR: %d\n", ret); return ret);
    
    // 4. (Boilerplate) Wait until the task execution is complete.
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

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
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
