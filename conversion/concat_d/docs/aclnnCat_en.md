# aclnnCat

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/concat)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Concatenates all tensors in `tensors` by `dim`. The dimensions except those corresponding to `dim` must be the same.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnCatGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnCat` is called to perform computation.

- `aclnnStatus aclnnCatGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnCat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnCatGetWorkspaceSize

- **Parameters**

  - `tensors` (aclTensorList*, computation input): list of tensors to be concatenated, `aclTensor` list on the device. The list length cannot exceed 32. The shape of each `aclTensor` in the list must be 1D to 8D, and the data types of tensors in the list must meet the data type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16, DOUBLE, or COMPLEX64.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, DOUBLE, or COMPLEX64.

  - `dim` (int64_t, input): dimension to be concatenated. The value is of the int type and ranges from [-tensors.dim(), tensors.dim() - 1].

  - `out` (aclTensor*, computation output): output tensor, `aclTensor` on the device. The data type must be convertible from the deduced data type of the tensors. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16, DOUBLE, or COMPLEX64.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, DOUBLE, or COMPLEX64.

  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.

  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API implements input parameter validation. The following error codes may be returned:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed tensors or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types or formats of tensors in the list are not supported.
                                  2. Data type deduction cannot be performed in the list of tensors.
                                  3. The deduced data type cannot be converted to that of out.
                                  4. The shapes of the non-cascading dimensions are inconsistent.
                                  5. dim exceeds the tensor dimension range.
```

## aclnnCat

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.

  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnCatGetWorkspaceSize`.

  - `executor` (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnCat` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cat.h"

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

  // Compute strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> outShape = {3, 3};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> outHostData(9, 0);

  // Create an input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tmp{input1, input2};
  aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
  // 3. Call the CANN operator library API. Replace it with the actual API name.
  int64_t dim = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnCat.
  ret = aclnnCatGetWorkspaceSize(tensorList, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnCat.
  ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensorList(tensorList);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
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
