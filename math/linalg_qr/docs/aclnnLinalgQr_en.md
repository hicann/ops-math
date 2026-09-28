# aclnnLinalgQr

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/linalg_qr)

## Supported Products

| Product| Supported|
| :--- | :------: |
| Ascend 950PR/Ascend 950DT|    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>|    ×     |
| <term>Atlas inference products</term>|    ×     |
| <term>Atlas training products</term>|    √     |

## Function

- Function: performs orthogonal decomposition on the input tensor.
- Formula:

  $$
  A = QR
  $$

  $A$ is the input tensor, which has at least two dimensions. $A$ can be represented as the product of the orthogonal matrix $Q$ and the upper triangular matrix $R$.

- Example:

  ```text
  A = tensor([[1, 2], [3, 4]], dtype=torch.float)
  q,r = linalg_qr(A, mode='reduced')
  q = tensor([[-0.3162, -0.9487],
             [-0.9487, 0.3162]])
  r = tensor([[-3.1623, -4.4272],
             [0.0000, -0.6325]])
  ```

## Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). You must call `aclnnLinalgQrGetWorkspaceSize` to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call `aclnnLinalgQr` to perform computation.

```Cpp
aclnnStatus aclnnLinalgQrGetWorkspaceSize(
  const aclTensor* self,
  int64_t          mode,
  aclTensor*       Q,
  aclTensor*       R,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnLinalgQr(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnLinalgQrGetWorkspaceSize

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
      <td>A in the formula.</td>
      <td>The shape must meet the constraints with Q and R.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mode (int64_t) </td>
      <td>Input</td>
      <td>Computational attribute of the output forms of Q and R in the formula.</td>
      <td><ul><li>When mode is set to 0, the <code>reduced</code> mode is used. For the input A(*, m, n), the output is the simplified Q(*, m, k) and R(*, k, n), where k is the minimum value of m and n.</li><li>When mode is set to 1, the <code>complete</code> mode is used. For the input A(*, m, n), the output is the complete Q(*, m, m) and R(*, m, n). </li><li>When mode is set to 2, the <code>r</code> mode is used. Only R(*, k, n) in the <code>reduced</code> scenario is computed, where k is the minimum value of m and n. The returned Q is an empty tensor.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Q (aclTensor*) </td>
      <td>Output</td>
      <td>Q in the formula, which is the orthogonal matrix output by the orthogonal decomposition.</td>
      <td>The shape is Q(*, m, m), Q(*, m, k), or empty, where k is the minimum value of m and n. The data format must be the same as that of <code>self</code> and R.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Derived from mode</td>
      <td>√</td>
    </tr>
    <tr>
      <td>R (aclTensor*) </td>
      <td>Output</td>
      <td>R in the formula, which is the upper triangular matrix output by the orthogonal decomposition.</td>
      <td>The shape is R(*, m, n) or R(*, k, n), where k is the minimum value of m and n. The data format must be the same as that of <code>self</code> and Q.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Derived from mode</td>
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
      <td>One of the input pointers self, Q, and R is null.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of self, Q, and R are not supported.</td>
    </tr>
    <tr>
      <td>The shapes of self, Q, and R do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The mode is not in the supported range.</td>
    </tr>
  </tbody></table>

## aclnnLinalgQr

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
      <td>Workspace size obtained by the first segment of the API <code>aclnnLinalgQrGetWorkspaceSize</code>.</td>
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

- Deterministic description: The default deterministic implementation of `aclnnLinalgQr` is used.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_linalg_qr.h"

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
  // Call aclrtMemcpy to copy the data from the host to the device.
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
    std::vector<int64_t>& selfShape, std::vector<int64_t>& qOutShape, std::vector<int64_t>& rOutShape,
    void** selfDeviceAddr, void** qOutDeviceAddr, void** rOutDeviceAddr, aclTensor** self, aclTensor** qOut,
    aclTensor** rOut)
{
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> qOutHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> rOutHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(qOutHostData, qOutShape, qOutDeviceAddr, aclDataType::ACL_FLOAT, qOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(rOutHostData, rOutShape, rOutDeviceAddr, aclDataType::ACL_FLOAT, rOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* qOut, aclTensor* rOut, int64_t mode, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* qOutDeviceAddr, void* rOutDeviceAddr, std::vector<int64_t>& qOutShape, std::vector<int64_t>& rOutShape,
    aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnLinalgQrGetWorkspaceSize(self, mode, qOut, rOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgQrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnLinalgQr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgQr failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy qOut.
  auto size1 = GetShapeSize(qOutShape);
  std::vector<double> resultData1(size1, 0);
  ret = aclrtMemcpy(
      resultData1.data(), resultData1.size() * sizeof(resultData1[0]), qOutDeviceAddr, size1 * sizeof(resultData1[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size1; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
  }

  // Copy rOut.
  auto size2 = GetShapeSize(rOutShape);
  std::vector<float> resultData2(size2, 0);
  ret = aclrtMemcpy(
      resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rOutDeviceAddr, size2 * sizeof(resultData2[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size2; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> qOutShape = {1, 1, 4, 4};
  std::vector<int64_t> rOutShape = {1, 1, 4, 4};

  void* selfDeviceAddr = nullptr;
  void* qOutDeviceAddr = nullptr;
  void* rOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* qOut = nullptr;
  aclTensor* rOut = nullptr;

  ret = CreateInputs(
      selfShape, qOutShape, rOutShape, &selfDeviceAddr, &qOutDeviceAddr, &rOutDeviceAddr, &self, &qOut, &rOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t mode = 0;
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      self, qOut, rOut, mode, &workspaceAddr, workspaceSize, qOutDeviceAddr, rOutDeviceAddr, qOutShape, rOutShape,
      stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Free the tensor.
  aclDestroyTensor(self);
  aclDestroyTensor(qOut);
  aclDestroyTensor(rOut);

  // Free the device memory.
  aclrtFree(selfDeviceAddr);
  aclrtFree(qOutDeviceAddr);
  aclrtFree(rOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
