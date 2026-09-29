# aclnnSearchSorteds

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Searches for the position where a given scalar value (self) should be inserted in a sorted 1-dimensional tensor (sortedSequence). A returned tensor whose shape is (1,) indicates the position where the given scalar value should be inserted in the original tensor. If `self` is of the tensor type, see [aclnnSearchSorted](./aclnnSearchSorted_en.md).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSearchSortedsGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSearchSorteds` is called to perform computation.

- `aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor *sortedSequence, const aclScalar *self, const bool outInt32, const bool right, const aclTensor *sorter, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSearchSorteds(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSearchSortedsGetWorkspaceSize

- **Parameters:**

  - `sortedSequence` (aclTensor*, compute input): sorted tensor (which must be 1-dimensional), aclTensor on the device. The data type can be DOUBLE, FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, or INT64. The data types of `sortedSequence` and `self` must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `self` (aclScalar*, compute input): value to be inserted, aclScalar on the host. The data type can be DOUBLE, FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, or INT64. The data types of `self` and `sortedSequence` must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)).
  - `outInt32` (bool, compute input): BOOL type on the host, indicating whether the output tensor is of the INT32 type.
  - `right` (bool, compute input): BOOL type on the host, indicating whether to return the position on the right if the same value is found. If the value is `False`, the position on the left is returned.
  - `sorter` (aclTensor*, compute input): aclTensor on the device, specifying the element sequence in `sortedSequence`. The data type can be INT64. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `out` (aclTensor*, compute output): aclTensor on the device. The data type can be INT32 or INT64. The [data format](../../../docs/en/context/data_format.md) can be ND.
  - `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\**, output): operator executor, containing the operator computation process.
  
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed sortedSequence, self, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of sortedSequence or self is not supported.
                                        1. The data type of out does not comply with the meaning of outInt32.
                                        2. If the data types of sortedSequence and self are different, they cannot be deduced.
                                        3. The passed sorter is not of the INT64 type.
                                        4. The shapes of sorter and sortedSequence are different.
                                        5. sortedSequence is not a 1-dimensional tensor.
  ```

## aclnnSearchSorteds

- **Parameters:**
  
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnSearchSortedsGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnSearchSorteds` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_searchsorted.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> sortedSequenceShape = {4};
  std::vector<int64_t> sorterShape = {4};
  std::vector<int64_t> outShape = {1};
  void* sortedSequenceDeviceAddr = nullptr;
  void* sorterDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* sortedSequence = nullptr;
  aclTensor* sorter = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> sortedSequenceHostData = {1,3,6,8};
  std::vector<int64_t> sorterHostData = {0,1,2,3};
  std::vector<int64_t> outHostData = {0};

  // Create a sortedSequence aclTensor.
  ret = CreateAclTensor(sortedSequenceHostData, sortedSequenceShape, &sortedSequenceDeviceAddr, aclDataType::ACL_FLOAT, &sortedSequence);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a sorter aclTensor.
  ret = CreateAclTensor(sorterHostData, sorterShape, &sorterDeviceAddr, aclDataType::ACL_INT64, &sorter);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  float value = 5;
  auto self = aclCreateScalar(&value, aclDataType::ACL_FLOAT);
  bool outInt32 = false;
  bool right = false;
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSearchSorteds.
  ret = aclnnSearchSortedsGetWorkspaceSize(sortedSequence, self, outInt32, right, sorter, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSortedsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnSearchSorteds.
  ret = aclnnSearchSorteds(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSorteds failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int64_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyScalar(self);
  aclDestroyTensor(sortedSequence);
  aclDestroyTensor(sorter);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(sortedSequenceDeviceAddr);
  aclrtFree(sorterDeviceAddr);
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
