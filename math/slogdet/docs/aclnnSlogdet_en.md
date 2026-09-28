# aclnnSlogdet

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/slogdet)

## Supported Products

| Product| Supported|
| :--- | :------: |
| Ascend 950PR/Ascend 950DT|    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>|    ×     |
| <term>Atlas inference products</term>|    √     |
| <term>Atlas training products</term>|    √     |

## Function

- This API is used to calculate the sign and natural logarithm of the determinant of the input self.
- Formula:

  $$
  signOut = sign(det(self))     \\
  logOut = log(abs(det(self)))
  $$

  `det` indicates determinant computation and `abs` indicates absolute value computation. If the result of `$det(self)$` is `0`, `$logOut` equals `-inf$`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, the `aclnnSlogdetGetWorkspaceSize` API is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, the `aclnnSlogdet` API is called to perform computation.

```Cpp
aclnnStatus aclnnSlogdetGetWorkspaceSize(
  const aclTensor* self,
  aclTensor*       signOut,
  aclTensor*       logOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSlogdet(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnSlogdetGetWorkspaceSize

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
      <td>The <code>self</code> in the formula, which is the input matrix.</td>
      <td>The shape is in the form of (*, n, n), where <code> *</code> indicates the batch of 0 or more dimensions, and n indicates any positive integer.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2 or higher</td>
      <td>√</td>
    </tr>
    <tr>
      <td>signOut (aclTensor*) </td>
      <td>Output</td>
      <td>Determinant result of <code>signOut</code> in the formula.</td>
      <td><ul><li>The value must be derived from <code>self</code>. </li><li>If <code>self</code> is of the COMPLEX type, <code>signOut</code> cannot be of a non-COMPLEX type. </li><li>The shape is the same as the batch of <code>self</code>.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>The same as the batch of self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>logOut (aclTensor*) </td>
      <td>Output</td>
      <td>Natural logarithm result of <code>logOut</code> in the formula.</td>
      <td><ul><li>The value must be derived from <code>self</code>. </li><li>If <code>self</code> is of the COMPLEX type, <code>logOut</code> cannot be of a non-COMPLEX type. </li><li>The batch of shape is the same as that of <code>self</code>.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Same as the batch of self</td>
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
  </tbody></table>

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

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
      <td>The input self, signOut, or logOut contains a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of self, signOut, and logOut are not supported.</td>
    </tr>
    <tr>
      <td>The shape of self does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The shapes of signOut and logOut do not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnSlogdet

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
      <td>Workspace size obtained by the first API call <code>aclnnSlogdetGetWorkspaceSize</code>.</td>
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

- Deterministic description: The default deterministic implementation of `aclnnSlogdet` is used.
- The input data cannot contain the overflow value `Inf`/`NaN`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_slogdet.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  //Calculate the strides of the contiguous tensors.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  //Call aclCreateTensor to create an ACL tensor.
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& signOutShape, std::vector<int64_t>& logOutShape,
    void** selfDeviceAddr, void** signOutDeviceAddr, void** logOutDeviceAddr, aclTensor** self, aclTensor** signOut,
    aclTensor** logOut)
{
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> signOutHostData = {0, 0, 0};
  std::vector<float> logOutHostData = {0, 0, 0};

  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a signOut aclTensor.
  ret = CreateAclTensor(signOutHostData, signOutShape, signOutDeviceAddr, aclDataType::ACL_FLOAT, signOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a logOut aclTensor.
  ret = CreateAclTensor(logOutHostData, logOutShape, logOutDeviceAddr, aclDataType::ACL_FLOAT, logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* signOut, aclTensor* logOut, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* signOutDeviceAddr, void* logOutDeviceAddr, std::vector<int64_t>& signOutShape,
    std::vector<int64_t>& logOutShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSlogdet.
  auto ret = aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the second-phase API of aclnnSlogdet.
  ret = aclnnSlogdet(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

  // Synchronize
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy signOut.
  auto sizeSign = GetShapeSize(signOutShape);
  std::vector<float> resultData(sizeSign, 0);
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), signOutDeviceAddr, sizeSign * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < sizeSign; i++) {
    LOG_PRINT("signout result[%ld] is: %f\n", i, resultData[i]);
  }

  // Copy logOut.
  auto sizeLog = GetShapeSize(logOutShape);
  std::vector<float> logResultData(sizeLog, 0);
  ret = aclrtMemcpy(
      logResultData.data(), logResultData.size() * sizeof(logResultData[0]), logOutDeviceAddr,
      sizeLog * sizeof(logResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < sizeLog; i++) {
    LOG_PRINT("logout result[%ld] is: %f\n", i, logResultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  // 1. Initialize the device and stream.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 2. Construct the input and output.
  std::vector<int64_t> selfShape = {3, 2, 2};
  std::vector<int64_t> signOutShape = {3};
  std::vector<int64_t> logOutShape = {3};

  void* selfDeviceAddr = nullptr;
  void* signOutDeviceAddr = nullptr;
  void* logOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* signOut = nullptr;
  aclTensor* logOut = nullptr;

  ret = CreateInputs(
      selfShape, signOutShape, logOutShape, &selfDeviceAddr, &signOutDeviceAddr, &logOutDeviceAddr, &self, &signOut,
      &logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator API.
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      self, signOut, logOut, &workspaceAddr, workspaceSize, signOutDeviceAddr, logOutDeviceAddr, signOutShape,
      logOutShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 6. Release aclTensor.
  aclDestroyTensor(self);
  aclDestroyTensor(signOut);
  aclDestroyTensor(logOut);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(signOutDeviceAddr);
  aclrtFree(logOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
