# aclnnTriangularSolve

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/triangular_solve)

## Supported Products

| Product| Supported|
| :--- | :------: |
| Ascend 950PR/Ascend 950DT|    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>|    ×     |
| <term>Atlas inference products</term>|    ×     |
| <term>Atlas training products</term>|    √     |

## Function

- Function: Solves a system of equations with a square upper or lower triangular invertible matrix A and multiple right-hand sides b.
- Formulas:

  $$
  AX = b
  $$

  $A$ is an upper triangular square matrix (or a lower triangular square matrix when upper is false), whose main diagonal does not contain 0 elements. $b$ and $A$ are two-dimensional matrices or batches of two-dimensional matrices. When the input is a batch, the returned output X is also a corresponding batch. If the main diagonal of $A$ contains 0 or elements close to 0, and unitriangular is false, the output may contain $NaN$.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, the `aclnnTriangularSolveGetWorkspaceSize` API is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, the `aclnnTriangularSolve` API is called to perform computation.

```Cpp
aclnnStatus aclnnTriangularSolveGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* A,
  bool             upper,
  bool             transpose,
  bool             unitriangular,
  aclTensor*       xOut,
  aclTensor*       mOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTriangularSolve(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnTriangularSolveGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self (aclTensor*) </td>
      <td>Input</td>
      <td> b in the formula, right-hand side of the equation.</td>
      <td><ul><li> Has the same data type as A. </li><li>self[-2]=A[-2]. </li><li>Except for the last two dimensions, other dimensions of A and self must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>A (aclTensor*) </td>
      <td>Input</td>
      <td> A in the formula, coefficient matrix.</td>
      <td><ul><li>The data type is the same as that of self. </li><li> The last two axes are equal. </li><li>Except for the last two dimensions, the remaining dimensions of A and self must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>upper (bool) </td>
      <td>Input</td>
      <td> Controls whether A in the formula is used for computation based on the upper or lower triangular part.</td>
      <td>The default value is true. If upper is true, A is an upper triangular matrix. If upper is false, A is a lower triangular matrix.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose (bool) </td>
      <td>Input</td>
      <td>Computing attribute that controls whether A or A<sup>T</sup> is used in the formula for calculation.</td>
      <td>The default value is false. When transpose is true, A<sup>T</sup>X=b is calculated.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unitriangular (bool) </td>
      <td>Input</td>
      <td>Computing attribute that controls whether A in the formula is processed as a unit triangular matrix.</td>
      <td><ul><li>The default value is false. </li><li>When unitriangular is true, the elements on the main diagonal of A are considered to be 1 instead of being referenced from A. </li><li>When unitriangular is true, the data types of the input self and A and the output xOut and mOut support only FLOAT.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*) </td>
      <td>Output</td>
      <td>X in the formula, which is the result of equation solving.</td>
      <td><ul><li>The data type is the same as that of self. </li><li>Its shape and the shapes of A and b after the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> must meet the AX = b constraint. </li><li>The dimensions of A and self after the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> relationship is met. The last axis is dim = self[-1].</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mOut (aclTensor*) </td>
      <td>Output</td>
      <td>Upper (lower) triangular copy of A after broadcast.</td>
      <td><ul><li>The data type is the same as that of self. </li><li>A and self must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> for the dimensions following the last axis. The last axis dim = A[-1].</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>One of the input pointers self, A, xOut, and mOut is null.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and formats of self, A, xOut, and mOut are not supported.</td>
    </tr>
    <tr>
      <td>The shapes of self, A, xOut, and mOut do not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnTriangularSolve

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size obtained by the first API call <code>aclnnTriangularSolveGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism description: The default implementation of `aclnnTriangularSolve` is deterministic.

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
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
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
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

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
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
