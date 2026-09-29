# aclnnClampMin

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

- Description: Limits all input elements within the range of [min, inf].

- Formula:

  $$
  {y}_{i} = max({{x}_{i}},{min\_value}_{i})
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnClampMinGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnClampMin` is called to perform computation.

- `aclnnStatus aclnnClampMinGetWorkspaceSize(const aclTensor *self, const aclScalar* clipValueMin, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnClampMin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

## aclnnClampMinGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, computation input): input tensor, `aclTensor` on the device. The shape must be 1D to 8D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, or INT64.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16.
  - `clipValueMin` (aclScalar*, computation input): lower bound.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BFLOAT16, or BOOL.
  - `out` (aclTensor *, computation output): output tensor. The shape must be identical to that of `self`. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, or INT64. The data type must be the same as that of `self`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, or BFLOAT16. The data type must be the same as that of `self`.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter validation. The following error codes may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer, or min is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self or out is not supported.
                                       
  ```

## aclnnClampMin

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnClampMinGetWorkspaceSize`.
  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnClampMin` defaults to a deterministic implementation.

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
    std::vector<int64_t>& shape, void** selfDeviceAddr, aclTensor** self, aclScalar** min, void** outDeviceAddr, aclTensor** out)
{
    double min_v = 2;

    std::vector<double> selfHostData = {0, 1, 0, 3, 0, 5, 0, 7};
    std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

    // Create a self aclTensor.
    auto ret = CreateAclTensor(selfHostData, shape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create min.
    *min = aclCreateScalar(&min_v, aclDataType::ACL_DOUBLE);
    CHECK_RET(*min != nullptr, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, shape, outDeviceAddr, aclDataType::ACL_DOUBLE, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclScalar* min, aclTensor* out)
{
  aclDestroyTensor(self);
  aclDestroyScalar(min);
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
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* min = nullptr;
    aclTensor* out = nullptr;
  
    ret = PrepareInputAndOutput(shape, &selfDeviceAddr, &self, &min, &outDeviceAddr, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnClampMin.
    ret = aclnnClampMinGetWorkspaceSize(self, min, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampMinGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnClampMin.
    ret = aclnnClampMin(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClampMin failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(shape);
    std::vector<double> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(double),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition. 
  ReleaseTensorAndScalar(self, min, out);

    // 7. Release device resources.
  ReleaseDevice(selfDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

    return 0;
}
```
