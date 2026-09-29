# aclnnPdistForward

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/pdist)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Operator function: Computes the p-norm distance of each pair of row vectors in the input `self`.
- Formula:
  Assume that the shape of the input `self` is [N, M], and $self_{in}$ indicates the element whose index is `n` in row `i` of `self`. The norm distance between row `i` and row `j` is calculated as follows:

  $$
  out_{ij} = \sum_{k=0}^{M-1}||self_{in}-self_{jn}||_p =  (\sum_{k=0}^{M-1}|self_{in}-self_{jn}|^p)^\frac{1}{p}
  $$

  The shape of out is [$\frac{1}{2}N(N-1)$], where the elements are arranged in the following order: $out_{01}$, ..., $out_{0,N-1}$,$out_{1,2}$, ... $out_{1,N-1}$, ..., $out_{N-2,N-1}$

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnPdistForwardGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnPdistForward is called to perform computation.

- `aclnnStatus aclnnPdistForwardGetWorkspaceSize(const aclTensor *self, const aclScalar *pScalar, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnPdistForward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnPdistForwardGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): `self` in the formula, aclTensor on the device. The data type can be FLOAT or FLOAT16, and must be the same as that of out. The shape can be 2D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `pScalar` (aclScalar*, input): `p` in the formula, aclScalar on the host. The data range is [0, +∞].
  - `out` (aclTensor*, computation output): `out` in the formula, aclTensor on the device. The data type can be FLOAT or FLOAT16. The shape can be 1D. If the shape of self is [N, M], the shape of out is [$\frac{1}{2}N(N-1)$]. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, pScalar, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                        2. The data types of self and out are inconsistent.
                                        3. The shape of self is not 2D.
                                        4. The shape of out is not 1D.
                                        5. The shape of out is inconsistent with the expected shape in the parameter description.
                                        6. The value of pScalar is less than 0.
  ```

## aclnnPdistForward

- **Parameters:**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPdistForwardGetWorkspaceSize.
  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnPdistForward` defaults to deterministic implementation.

## Examples

The following examples are for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pdist_forward.h"

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
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {3,3};
  std::vector<int64_t> outShape = {3};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* p = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 0, 0, 1, 1, 1, 1, 2, 3};
  float pHostData = 2.0f;
  std::vector<float> outHostData = {0, 0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a p aclScalar.
  p = aclCreateScalar(&pHostData, aclDataType::ACL_FLOAT);
  CHECK_RET(p != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPdistForward.
  ret = aclnnPdistForwardGetWorkspaceSize(self, p, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdistForwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnPdistForward.
  ret = aclnnPdistForward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdistForward failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
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

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(p);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
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
