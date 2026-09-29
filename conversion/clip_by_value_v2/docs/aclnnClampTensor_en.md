# aclnnClampTensor

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/clip_by_value_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Limits all input elements within the range of [min, max]. If `min` is not set, there is no lower limit. If `max` is not set, there is no upper limit.

- Formula:

  $$
  {y}_{i} = max(min({{x}_{i}},{max\_value}_{i}),{min\_value}_{i})
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnClampTensorGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnClampTensor` is called to perform computation.

- `aclnnStatus aclnnClampTensorGetWorkspaceSize(const aclTensor *self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnClampTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnClampTensorGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, computation input): input tensor, `aclTensor` on the device. The shape must be 1D to 8D. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `clipValueMin` and `clipValueMax`. The shape must be broadcastable with those of `min` and `max` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16.
  - `clipValueMin` (aclTensor*, computation input): input lower limit tensor. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self` and `clipValueMax`. The shape must be broadcastable with those of `self` and `max` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64, BFLOAT16, or BOOL.
  - `clipValueMax` (aclTensor*, computation input): input upper limit tensor. The data type must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self` and `clipValueMin`. The shape must be broadcastable with those of `self` and `clipValueMin` (see [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64, BFLOAT16, or BOOL.
  - `out` (aclTensor *, computation output): output tensor. The shape must be identical to that of the broadcast result of `self`, `clipValueMin`, and `clipValueMax`. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, or INT64. The data type must be the same as that of `self` and must be convertible from the deduced data type of `self`, `clipValueMin`, and `clipValueMax` (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16. The data type must be the same as that of `self` and must be convertible from the deduced data type of `self`, `clipValueMin`, and `clipValueMax` (see [conversion relationship](../../../docs/en/context/conversion_relationship.md)).
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer, or clipValueMin and clipValueMax are both null pointers.
                                         
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or out is not supported.
                                    2. The shapes of self, clipValueMin, and clipValueMax are not broadcastable, or the broadcast result does not match the shape of out.
                                    3. The data type deduction of self, clipValueMin, and clipValueMax fails, or the deduced data type cannot be converted to that of out.
  ```

## aclnnClampTensor

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnClampTensorGetWorkspaceSize`.
  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnClampTensor` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clamp.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& shape, void** selfDeviceAddr, aclTensor** self, void** minDeviceAddr,
    aclTensor** clipValueMin, void** maxDeviceAddr, aclTensor** clipValueMax, void** outDeviceAddr, aclTensor** out)
{
    std::vector<int8_t> selfHostData = {0, 1, 0, 3, 0, 5, 0, 7};
    std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int8_t> minHostData = {1, 3, 0, 0, 0, 0, 0, 0};
    std::vector<int8_t> maxHostData = {5, 5, 3, 3, 4, 5, 6, 6};

    // Create a self aclTensor.
    auto ret = CreateAclTensor(selfHostData, shape, selfDeviceAddr, aclDataType::ACL_INT8, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a min aclTensor.
    ret = CreateAclTensor(minHostData, shape, minDeviceAddr, aclDataType::ACL_INT8, clipValueMin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a max aclTensor.
    ret = CreateAclTensor(maxHostData, shape, maxDeviceAddr, aclDataType::ACL_INT8, clipValueMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, shape, outDeviceAddr, aclDataType::ACL_INT8, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclTensor* max, aclTensor* min, aclTensor* out)
{
  aclDestroyTensor(self);
  aclDestroyTensor(max);
  aclDestroyTensor(min);
  aclDestroyTensor(out);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* minDeviceAddr, void* maxDeviceAddr, void* outDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
  aclrtFree(selfDeviceAddr);
  aclrtFree(minDeviceAddr);
  aclrtFree(maxDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> shape = {4, 2};

    void* selfDeviceAddr = nullptr;
    void* minDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* clipValueMin = nullptr;
    aclTensor* clipValueMax = nullptr;
    aclTensor* out = nullptr;

    ret = PrepareInputAndOutput(
        shape, &selfDeviceAddr, &self, &minDeviceAddr, &clipValueMin, &maxDeviceAddr, &clipValueMax, &outDeviceAddr, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnClampTensor.
    ret = aclnnClampTensorGetWorkspaceSize(self, clipValueMin, clipValueMax, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnClampTensor.
    ret = aclnnClampTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampTensor failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  ReleaseTensorAndScalar(self, clipValueMax, clipValueMin, out);

    // 7. Release device resources.
  ReleaseDevice(selfDeviceAddr, minDeviceAddr, maxDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

    return 0;
}
```
