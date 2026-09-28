# aclnnSWhere

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/select)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

+ Operator function: Selects elements from `self` or `other` based on the condition and returns the elements (broadcasting is supported).
+ The formula is as follows:

$$
out_i=where(self_i,other_i,condition_i)=\begin{cases}
  self_i, & \text{if condition}_i \\
  other_i, & \text{otherwise}
   \end{cases}
$$

## Prototype

Each operator has [two-phase API](../../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSWhereGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSWhere` is called to perform computation.

- `aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSWhere(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSWhereGetWorkspaceSize

* **Parameter description**:
  - condition (aclTensor*, compute input): input condition in the formula, which is an aclTensor on the device. The data type can be UINT8 or BOOL.[Non-Contiguous Tensor](../../../../docs/en/context/non_contiguous_tensor.md) are supported. The shape must meet the broadcast relationship (../../../docs/en/context/broadcast_relationship.md) with self and other.[Non-Contiguous Tensor](../../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../../docs/en/context/data_format.md) can be ND. The number of data dimensions cannot exceed 8.

  - `self` (aclTensor*, computation input): input `self` in the formula. The data type of `self` must meet the data type deduction rules with `other` (see [deduction relationship](../../../../docs/en/context/deduction_relationship.md)). The shape of the aclTensor on the device must meet the [broadcast relationship](../../../../docs/en/context/broadcast_relationship.md) with `other` and `condition`.[Non-Contiguous Tensor](../../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../../docs/en/context/data_format.md) can be ND. The number of data dimensions cannot exceed 8.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT: The data type can be FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128 or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64 or COMPLEX128.

  - `other` (aclTensor*, computation input): input `other` in the formula. The data type of `other` must meet the data type deduction rules with `self` (see [deduction relationship](../../../../docs/en/context/deduction_relationship.md)). The shape of the aclTensor on the device must meet the [broadcast relationship](../../../../docs/en/context/broadcast_relationship.md) with `self` and `condition`.[Non-Contiguous Tensor](../../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../../docs/en/context/data_format.md) can be ND. The number of data dimensions cannot exceed 8.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT: The data type can be FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128 or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64 or COMPLEX128.

  - out (aclTensor *, output): output out in the formula.[Non-Contiguous Tensor](../../../../docs/en/context/non_contiguous_tensor.md) are supported. It is an aclTensor on the device. The shape must be that after self, other, and condition are broadcast.[Non-Contiguous Tensor](../../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../../docs/en/context/data_format.md) can be ND. The number of data dimensions cannot exceed 8.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT: The data type can be FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128 or BFLOAT16.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64 or COMPLEX128.

  - `workspaceSize` (uint64_t \*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

* **Return value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 300px">
  <col style="width: 136px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self, other, condition, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type and dimension of self, other, or condition are not supported.</td>
    </tr>
    <tr>
      <td>Type promotion between <code>self</code> and <code>other</code> cannot be performed.</td>
    </tr>
    <tr>
      <td>The broadcast of self, other, or condition fails to be inferred, or the broadcast result is inconsistent with the shape of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnSWhere

* **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnSWhereGetWorkspaceSize.</td>
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

* **Return value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnSWhere` defaults to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_s_where.h"

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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> conditionShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* conditionDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* condition = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 0, 0, 0, 0, 0, 0, 7};
  std::vector<float> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int8_t> conditionHostData = {false,false,false,false,true,true,true,true};
  std::vector<float> outHostData = {10, 10, 10, 10, 10, 10, 10, 10};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a condition aclTensor.
  ret = CreateAclTensor(conditionHostData, conditionShape, &conditionDeviceAddr, aclDataType::ACL_BOOL, &condition);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSWhere.
  ret = aclnnSWhereGetWorkspaceSize(condition, self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSWhereGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnSWhere.
  ret = aclnnSWhere(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSWhere failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(condition);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(conditionDeviceAddr);
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
