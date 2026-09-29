# aclnnLinalgQr

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/linalg_qr)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

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
  q,r = linalg_qr(A, mode='reduced')
  q = tensor([[-0.3162, -0.9487],
             [-0.9487, 0.3162]])
  r = tensor([[-3.1623, -4.4272],
             [0.0000, -0.6325]])
  ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLinalgQrGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLinalgQr` is called to perform computation.

- `aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor *self, int64_t mode, aclTensor *Q, aclTensor *R, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLinalgQr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLinalgQrGetWorkspaceSize

- **Parameters**

  - `self` (aclTensor*, computation input): `$A$` in the formula. The data type can be FLOAT, FLOAT16, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports 2 to 8 dimensions, and must meet the constraints with Q and R.

  - `mode` (int64_t, computation input): computation attribute. When mode is `0`, the reduced mode is used. For the input `A(\*, m, n)`, the output is simplified `Q(\*, m, k)` and `R(\*, k, n)`, where `k` is the minimum value of `m` and `n`. When mode is `1`, the complete mode is used. For the input `A(\*, m, n)`, the output is complete `Q(\*, m, m)` and `R(\*, m, n)`. When mode is `2`, the `r` mode is used. Only `R(\*,k,n)` in the reduced scenario is calculated, where `k` is the minimum value of `m` and `n`. The returned `Q` is an empty tensor.

  - `Q` (aclTensor, computation output): `$Q$` in the formula. The data type can be FLOAT, FLOAT16, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND and must be the same as that of `self` and `R`. The shape is `Q(\, m, m)`, `Q(\*, m, k)`, or empty, where `k` is the minimum value of `m` and `n`.

  - `R` (aclTensor, computation output): `$R$` in the formula. The data type can be FLOAT, FLOAT16, DOUBLE, COMPLEX64, or COMPLEX128. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND and must be the same as that of `self` and `Q`. The shape is `R(\, m, n)` or `R(\*, k, n)`, where `k` is the minimum value of `m` and `n`.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor \**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input self, Q, or R is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type and format of self, Q, or R are not supported.
                                   2. The shape of self, Q, or R does not meet the requirements.
                                   3. mode is not within the optional range.
  ```

## aclnnLinalgQr

- **Parameters**

  - `workspace` (void *, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLinalgQrGetWorkspaceSize`.

  - `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnLinalgQr` defaults to deterministic implementation.

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

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
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

  // Destroy tensors.
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
