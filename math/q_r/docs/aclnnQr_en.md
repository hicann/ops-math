# aclnnQr

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/q_r)

## Supported Products

| Product| Supported|
| :--- | :------: |
| Ascend 950PR/Ascend 950DT|    ×    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>|    ×    |
| <term>Atlas inference products</term>|    ×    |
| <term>Atlas training products</term>|    √    |

## Function

- Description: Performs orthogonal decomposition on the input tensor.
- Formula:

  $$
  A = QR
  $$

  `$A$` is the input tensor with at least two dimensions. `A` can be expressed as the product of the orthogonal matrix `$Q$` and the upper triangular matrix `$R$`.

- Example:

  ```text
  A = tensor([[1, 2], [3, 4]], dtype=torch.float)
  Q, R = QR(A, some=False)
  Q = tensor([[-0.3162, -0.9487],
              [-0.9487, 0.3162]])
  R = tensor([[-3.1623, -4.4272],
              [0.0000, -0.6325]])
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQrGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnQr` is called to perform computation.

```Cpp
aclnnStatus aclnnQrGetWorkspaceSize(
  const aclTensor* self,
  bool             some,
  aclTensor*       Q,
  aclTensor*       R,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnQr(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnQrGetWorkspaceSize

- **Description**

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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>A in the formula.</td>
      <td>The shape dimension must be at least 2 and at most 8. The shape is in the form of [..., M, N], where `...` indicates dimensions 0 to 6.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>some (bool)</td>
      <td>Input</td>
      <td>Computational attribute that controls the output forms of Q and R in the formula.</td>
      <td><ul><li>If set to false, Q is a square matrix, for example, A[..., M, N], and the complete Q[..., M, M] and R[..., M, N] are output. </li><li>If set to true, Q is a thin matrix, for example, A[..., M, N], and the output is Q[..., M, K] and R[..., K, N], where K is the smaller value between M and N.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Q in the formula, which is the orthogonal matrix output by orthogonal decomposition.</td>
      <td>For details about the shape constraint, see the description of the <code>some</code> parameter. The data format must be the same as that of <code>self</code> and R.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Derived from some</td>
      <td>√</td>
    </tr>
    <tr>
      <td>R (aclTensor*)</td>
      <td>Output</td>
      <td>R in the formula, which is the upper triangular matrix output by orthogonal decomposition.</td>
      <td>For details about the shape constraint, see the description of the <code>some</code> parameter. The data format must be the same as that of <code>self</code> and Q.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Derived from some</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be returned:

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
      <td>The input self, Q, or R is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type or format of self, Q, or R is not supported.</td>
    </tr>
    <tr>
      <td>The shape of self, Q, or R does not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnQr

- **Description**

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
      <td>Workspace size obtained by the first-phase API <code>aclnnQrGetWorkspaceSize</code>.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism description: **aclnnQr** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_qr.h"

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

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& qShape, std::vector<int64_t>& rShape, void** selfDeviceAddr,
    aclTensor** self, void** qDeviceAddr, aclTensor** q, void** rDeviceAddr, aclTensor** r)
{
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> qHostData = {0, 0, 0, 0};
  std::vector<float> rHostData = {0, 0, 0, 0};
  // Create a self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create q and r aclTensors.
  ret = CreateAclTensor(qHostData, qShape, qDeviceAddr, aclDataType::ACL_FLOAT, q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(rHostData, rShape, rDeviceAddr, aclDataType::ACL_FLOAT, r);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

void ReleaseTensorAndScalar(aclTensor* self, aclTensor* q, aclTensor* r)
{
    aclDestroyTensor(self);
    aclDestroyTensor(q);
    aclDestroyTensor(r);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* qDeviceAddr, void* rDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
    aclrtFree(selfDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(rDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> selfShape = {2, 2};
  bool some = false;
  std::vector<int64_t> qShape = {2, 2};
  std::vector<int64_t> rShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* qDeviceAddr = nullptr;
  void* rDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* q = nullptr;
  aclTensor* r = nullptr;  
  
  ret = PrepareInputAndOutput(selfShape, qShape, rShape, &selfDeviceAddr, &self, &qDeviceAddr, &q, &rDeviceAddr, &r);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnQr.
  ret = aclnnQrGetWorkspaceSize(self, some, q, r, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnQr.
  ret = aclnnQr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQr failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(qShape);
  std::vector<float> resultQData(size, 0);
  ret = aclrtMemcpy(resultQData.data(), resultQData.size() * sizeof(resultQData[0]), qDeviceAddr,
                    size * sizeof(resultQData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result Q from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result Q[%ld] is: %f\n", i, resultQData[i]);
  }

  std::vector<float> resultRData(size, 0);
  ret = aclrtMemcpy(resultRData.data(), resultRData.size() * sizeof(resultRData[0]), rDeviceAddr,
                    size * sizeof(resultRData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result R from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result R[%ld] is: %f\n", i, resultRData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  ReleaseTensorAndScalar(self, q, r);

  // 7. Release device resources. Set the parameters based on the API definition.
  ReleaseDevice(selfDeviceAddr, qDeviceAddr, rDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```
