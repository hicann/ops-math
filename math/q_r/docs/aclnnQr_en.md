# aclnnQr

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/q_r)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    √    |

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
Q,R = QR(A, some =False)
Q = tensor([[-0.3162, -0.9487],
            [-0.9487, 0.3162]])
R = tensor([[-3.1623, -4.4272],
            [0.0000, -0.6325]])
```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQrGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnQr` is called to perform computation.

- `aclnnStatus aclnnQrGetWorkspaceSize(const aclTensor *self, bool some, aclTensor *Q, aclTensor *R, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnQr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnQrGetWorkspaceSize

- **Parameters:**

  - `self` (aclTensor*, computation input): `$A$` in the formula. The data type can be FLOAT, FLOAT16, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND. The shape supports 2 to 8 dimensions, and is represented as [..., M, N], where `...` indicates 0 to 6 dimensions.

  - `some` (bool, computation input): determines whether `Q` is a square matrix. If this parameter is set to `false`, `Q` is a square matrix. For example, if `A` is [..., M, N], the complete Q[..., M, M] and R[..., M, N] are output. If this parameter is set to `true`, `Q` is a skinny matrix. For example, if `A` is [..., M, N], Q[..., M, K] and R[..., K, N] are output, where `K` is the minimum value of `M` and `N`.

  - `Q` (aclTensor*, computation output): `$Q$` in the formula. The data type can be FLOAT, FLOAT16, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND and must be the same as that of `self` and `R`. For details about the shape restrictions, see the description of the `some` parameter.

  - `R` (aclTensor*, computation output): `$R$` in the formula. The data type can be FLOAT, FLOAT16, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND and must be the same as that of `self` and `Q`. For details about the shape restrictions, see the description of the `some` parameter.

  - `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter verification. The following errors may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed `self`, `Q`, or `R` is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of `self`, `Q`, or `R` is not supported.
                                 2. The shape of `self`, `Q`, or `R` does not comply with the constraints.
```

## aclnnQr

- **Parameters:**

  - `workspace` (void *, input): memory address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnQrGetWorkspaceSize`.

  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnQr` defaults to a deterministic implementation.

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

  // Compute the strides of contiguous tensors.
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

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  ReleaseTensorAndScalar(self, q, r);

  // 7. Release device resources. Set the parameters based on the API definition.
  ReleaseDevice(selfDeviceAddr, qDeviceAddr, rDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```
