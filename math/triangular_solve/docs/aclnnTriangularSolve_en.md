# aclnnTriangularSolve

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/triangular_solve)

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |   √     |

## Function

- Description: Solves a system of equations with a square upper or lower triangular invertible matrix A and multiple right-hand sides b.
- Formula:

  $$
  AX = b
  $$
  
  $A$ is an upper triangular matrix (or a lower triangular matrix if upper=false) and does not have zeros on the main diagonal. $b,A$ can be two-dimensional matrices or batches of two-dimensional matrices. If the inputs are batches, then returns batched outputs X. If the main diagonal of $A$ contains `0` or elements close to `0`, and `unitriangular` is `false`, the output result may contain $NaN$.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnTriangularSolveGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnTriangularSolve` is called to perform computation.

- `aclnnStatus aclnnTriangularSolveGetWorkspaceSize(const aclTensor *self, const aclTensor *A, bool upper, bool transpose, bool unitriangular, aclTensor *xOut, aclTensor *mOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnTriangularSolve(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnTriangularSolveGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, compute input): $b$ in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128, and must be the same as that of `A`. The shape supports 2 to 8 dimensions. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. self[-2]=A[-2]. The dimensions (except the last two dimensions) of `A` and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md).

  - `A` (aclTensor*, compute input): $A$ in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128, and must be the same as that of `self`. The shape supports 2 to 8 dimensions. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The last two axes are equal. The dimensions (except the last two dimensions) of `A` and `self` must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md).

  - `upper` (bool, compute input): computation attribute. The default value is `true`, indicating that `A` is an upper triangular matrix. When `upper` is set to `false`, `A` is a lower triangular matrix.

  - `transpose` (bool, compute input): computation attribute. The default value is `false`. When `transpose` is `true`, $A^T X=b$ is computed.

  - `unitriangular` (bool, compute input): computation attribute. The default value is `false`. When `unitriangular` is set to `true`, the elements on the main diagonal of `A` are considered as 1 instead of being referenced from `A`. When `unitriangular` is set to `true`, the data types of the inputs `self` and `A` and the outputs `xOut` and `mOut` support only FLOAT.

  - `xOut` (aclTensor *, compute output): $X$ in the formula. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128, and must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shapes of `xOut` and broadcasted `A`,`b` must meet the $AX=b$ constraint. The shapes of `A` and `self` meet the broadcast relationship. The last axis dim=self[-1].

  - `mOut` (aclTensor *, compute output): copy of the upper triangle (lower triangle) of `A` after broadcasting. The data type can be FLOAT, DOUBLE, COMPLEX64, or COMPLEX128, and must be the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shapes of `A` and `self` meet the broadcast relationship. The last axis dim=A[-1].

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API performs input parameter validation. The following errors may be returned:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, A, xOut, or mOut is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self, A, xOut, or mOut is not supported.
                                   2. The shape of self, A, xOut, or mOut does not meet the constraint.
  ```

## aclnnTriangularSolve

- **Parameters**

  - `workspace` (void *, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnTriangularSolveGetWorkspaceSize`.

  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnTriangularSolve` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_triangular_solve.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
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
  // Set device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {3, 1};
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> xOutShape = {3, 1};
  std::vector<int64_t> mOutShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* xOutDeviceAddr = nullptr;
  void* mOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* xOut = nullptr;
  aclTensor* mOut = nullptr;
  bool upper = true;
  bool transpose = false;
  bool unitriangular = false;
  std::vector<float> selfHostData = {1, 2, 3};
  std::vector<float> otherHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  std::vector<float> xOutHostData = {-0.2500, -0.1250, 0.5000};
  std::vector<float> mOutHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an xOut aclTensor.
  ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT, &xOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an mOut aclTensor.
  ret = CreateAclTensor(mOutHostData, mOutShape, &mOutDeviceAddr, aclDataType::ACL_FLOAT, &mOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnTriangularSolve.
  ret = aclnnTriangularSolveGetWorkspaceSize(self, other, upper, transpose, unitriangular, xOut, mOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTriangularSolve.
  ret = aclnnTriangularSolve(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolve failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto xSize = GetShapeSize(xOutShape);
  std::vector<float> resultData(xSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xOutDeviceAddr,
                    xSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  auto mSize = GetShapeSize(mOutShape);
  std::vector<float> mResultData(mSize, 0);
  ret = aclrtMemcpy(mResultData.data(), mResultData.size() * sizeof(mResultData[0]), mOutDeviceAddr,
                    mSize * sizeof(mResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < mSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, mResultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(xOut);
  aclDestroyTensor(mOut);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(xOutDeviceAddr);
  aclrtFree(mOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
