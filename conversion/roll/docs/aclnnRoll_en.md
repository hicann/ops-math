# aclnnRoll

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/roll)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Rolls data in a tensor based on the given size and dimension.

The following is an example:

```python
x = tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
After the torch.roll(x, 1, 0) computation, (moving down one row along the dim=0 dimension)

x = tensor([[7, 8],
            [1, 2],
            [3, 4],
            [5, 6]])
```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnRollGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRoll` is called to perform computation.

- `aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnRollGetWorkspaceSize

- **Parameters:**

  - x (aclTensor*, compute input): input original data, aclTensor on the device. The shape supports 0 to 8 dimensions. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT16, FLOAT32, INT8, UINT8, INT32, INT64, UINT32, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT32, INT64, UINT32, or BOOL.
  - `shifts` (aclIntArray*, compute input): number of steps by which elements are rolled in each dimension. It is `aclIntArray` on the host. The array length must be the same as that of `dims`.

  - `dims` (aclIntArray*, compute input): dimensions along which elements are rolled. It is `aclIntArray` on the host. The array length must be the same as that of `shifts`. The value range is [–x.dim(), x.dim() – 1]. For example, if x has 4 dimensions, the value range is [–4, 3].

  - `out` (aclTensor*, compute output): output data after rolling. It is `aclTensor` on the device. The shape supports 0–8 dimensions and must be the same as that of x. The data type must be the same as that of x. The [data format](../../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be FLOAT16, FLOAT32, INT8, UINT8, INT32, INT64, UINT32, or BOOL.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT32, INT64, UINT32, or BOOL.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input x or shifts is a null pointer.
  ACLNN_ERR_PARAM_INVALID (161002): 1. The data type of x is not supported.
                                      2. The data format of x is not supported.
                                      3. When dims is not null, the sizes of shifts and dims are different.
                                      4. If dims is empty, the length of the shifts array is not equal to 1.
                                      5. The value of dims is not within the range of [-x.dim(), x.dim() - 1].
                                      6. The shape of x exceeds 8 dimensions.
                                      7. The shapes of out and x are inconsistent.
  ```

## aclnnRoll

- **Parameters:**

  - `workspace` (void\*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRollGetWorkSpaceSize.

  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnRoll` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roll.h"

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
  // Call aclrtMemcpy to copy the data from the host to the device.
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

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> xShape = {1, 2, 3};
  std::vector<int64_t> outShape = {1, 2, 3};
  void* xDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclIntArray* shifts = nullptr;
  aclIntArray* dims = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> outHostData(6, 0);
  std::vector<int64_t> dimsData = {0, 0};
  std::vector<int64_t> shiftsData = {1, 2};
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a shifts aclIntArray.
  shifts = aclCreateIntArray(shiftsData.data(), 2);
  CHECK_RET(shifts != nullptr, return ret);
  // Create a dims aclIntArray.
  dims = aclCreateIntArray(dimsData.data(), 2);
  CHECK_RET(dims != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoll.
  ret = aclnnRollGetWorkspaceSize(x, shifts, dims, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRollGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRoll.
  ret = aclnnRoll(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoll failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyIntArray(shifts);
  aclDestroyIntArray(dims);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(xDeviceAddr);
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
