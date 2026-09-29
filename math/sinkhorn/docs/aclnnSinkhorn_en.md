# aclnnSinkhorn

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/sinkhorn)

## Supported Product Models

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     ×     |
| <term>Atlas inference products</term>                             |   ×     |
| <term>Atlas training products</term>                             |   ×     |

## Function

- Description:

  Computes the Sinkhorn distance, which can be used for expert routing in MoE models.

- Formula:

  $$
  p=Sinkhorn(cost, tol)
  $$

  **Input**:

  cost(R, C): 2D cost matrix
  tol: tolerance

  **Initialization**:

  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1\_old= d1 \\
  $$

  **Repeating**:

  $$
  d0 = \frac{1}{R * (sum(d1 * cost, 1) + eps)} \\
  d1 = \frac{1}{C * (sum(d0.unsqueeze(1) * cost, 0) + eps)} \\
  error = mean(abs(d1\_old - d1)) \\
  d1\_old = d1
  $$

  Until:

  $$
  error <= tol
  $$

  **Output**:

  $$
  p = d1 * cost * d0.unsqueeze(1)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSinkhornGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnSinkhorn` is called to perform computation.

- `aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor *cost, const aclScalar *tol, aclTensor *p, uint64_t *workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSinkhornGetWorkspaceSize

- **Parameters:**

    - `cost` (aclTensor*, input): cost tensor, `cost` in the formula, aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT. The value must be in the range [0, 1] and can be normalized. [Data Format] (../../../docs/en/context/data_format.md) The ND format is supported. The input is a two-dimensional matrix, and the number of rows cannot exceed 10,000 and the number of columns cannot exceed 1,024. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.   
    - `tol` (aclScalar*, input): Sinkhorn tolerance. The data type can be FLOAT. If a null pointer is passed, the value of `tol` is 0.0001.
    - `p` (aclTensor*, output): optimal transport tensor, `p` in the formula, aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape is 2D. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The data type and shape are the same as those of the input `cost`.
    - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
    - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns:**

  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input cost or p is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of cost or p is not supported.
                                        2. The data types of cost and p cannot be deduced.
  ```

## aclnnSinkhorn

- **Parameters:**

    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnSinkhornGetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): AscendCL stream for executing the task.

- **Returns:**

  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnSinkhorn` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sinkhorn.h"

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

int Init(int32_t deviceId, aclrtStream *stream) {
  // (Boilerplate) Initialize AscendCL.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> costShape = {3, 2};
  std::vector<int64_t> pShape = {3, 2};
  void* costDeviceAddr = nullptr;
  void* pDeviceAddr = nullptr;
  aclTensor* cost = nullptr;
  aclScalar* tol = nullptr;
  aclTensor* p = nullptr;
  std::vector<float> costHostData = {45, 48, 65, 68, 68, 10};
  std::vector<float> pHostData(6, 0);

  float tolValue = 0.0001;

  // Create a cost aclTensor.
  ret = CreateAclTensor(costHostData, costShape, &costDeviceAddr, aclDataType::ACL_FLOAT, &cost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a p aclTensor.
  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a tol aclScalar.
  tol = aclCreateScalar(&tolValue, aclDataType::ACL_FLOAT);
  CHECK_RET(tol != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSinkhorn.
  ret = aclnnSinkhornGetWorkspaceSize(cost, tol, p, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhornGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSinkhorn.
  ret = aclnnSinkhorn(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhorn failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(pShape);
  std::vector<float> pData(size, 0);
  ret = aclrtMemcpy(pData.data(), pData.size() * sizeof(pData[0]), pDeviceAddr,
                    size * sizeof(pData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("p result[%ld] is: %e\n", i, pData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(cost);
  aclDestroyTensor(p);
  aclDestroyScalar(tol);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(costDeviceAddr);
  aclrtFree(pDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
