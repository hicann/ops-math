# aclRfft1D 

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

* Description: Performs Real Fast Fourier Transform (RFFT) on the input tensor `self`, and outputs a complex tensor containing non-negative frequencies.
* Formula:

  $$
  {\displaystyle X_{k}=\sum _{n=0}^{N-1}x_{n}\cdot e^{-i2\pi {\tfrac {k}{N}}n}}
  $$
  
* Example:
  If `self` is {1, 2, 3, 4}, then `out` = {10, 0, -2, 2, -2, 0} = {10 + 0j, -2 + 2j, -2 + 0j} (customized).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclRfft1DGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclRfft1D` is called to perform computation.

* `aclnnStatus aclRfft1DGetWorkspaceSize(const aclTensor* self, int64_t n, int64_t dim, int64_t norm, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclRfft1D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclRfft1DGetWorkspaceSize 

* Parameters
    * `self` (aclTensor\*, compute input): input in the formula. The data type can be FLOAT, and the [data format](../../../docs/en/context/data_format.md) can be ND. The shape supports 1 to 7 dimensions.
    * `n` (int64_t, compute input): signal length. The data type is INT64. If the value is specified, the input will be zeroed or trimmed to this length before Rfft1D is computed. The value range of n is [1, 4096]. The maximum value of 2<sup>n</sup> is 262144.
    * `dim` (int64_t, compute input): dimension. The data type is INT64. If the value is specified, RFFT applies to the specified dimension. The supported values are [-self.dim(), self.dim()-1].
    * `norm` (int64_t, compute input): normalization mode. The data type is INT64. The value `1` indicates no normalization, `2` indicates normalization by 1/n, and `3` indicates normalization by 1/sqrt(n).
    * `out` (aclTensor\*, compute output): output in the formula. The data type can be FLOAT. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * `workspaceSize` (uint64_t, output): size of the workspace to be allocated on the device.
    * `executor` (aclOpExecutor\*, output): operator executor, containing the operator computation process.

* Returns:

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    ```text
    The first-phase API implements input parameter verification. The following errors may be thrown:
    `161001` (ACLNN_ERR_PARAM_NULLPTR): 1. The input or output tensor is empty.
    `161002` (ACLNN_ERR_PARAM_INVALID): 1. The data type and dimension of `self` are not supported.
    `561103` (ACLNN_ERR_INNER_NULLPTR): 1. The intermediate result is null.
    `561101` (ACLNN_ERR_INNER_CREATE_EXECUTOR): 1. The executor is null.                             
    ```

## aclRfft1D

* Parameters
    * `workspace` (void*, input): address of the workspace to be allocated on the device.
    * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclRfft1DGetWorkspaceSize`.
    * `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
    * `stream` (aclrtStream, input): stream for executing the task.
* Returns:

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclRfft1D` defaults to a deterministic implementation.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_rfft1d.h"
#define CHECK_RET(cond, return_expr)   \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)
#define LOG_PRINT(message, ...)       \
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

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> selfShape = {1, 1, 8};
    std::vector<int64_t> outShape = {1, 1, 5, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    int n = 8;
    int dim = -1;
    int norm = 1; // backward - 1, forward - 2, ortho - 3

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclRfft1DGetWorkspaceSize(self, n, dim, norm, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1DGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclRfft1D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1D failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);

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
