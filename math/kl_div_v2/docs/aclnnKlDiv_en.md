# aclnnKlDiv

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/kl_div_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

* Description: Calculates the KL divergence.
* Formula:
  * Defines loss_pointwise and saves the intermediate result.
    
    $$
    loss\_pointwise_i=\begin{cases}
    NaN & \text{ if }&logTarget=false \text{ and } target_i <= 0,  \\
    target_i * \left ( \log{(target_i)}- self_i  \right )  & \text{ if }& logTarget=false, \\
    \exp^ {target_i} * \left ( target_i- self_i \right )  & \text{ else. }
    \end{cases}
    $$
    
  * Formula for computing `out`:
  
    $$
    out=\begin{cases}
    \bar{loss\_pointwise}  & \text{ if }& reduction= 1, \\
    \sum loss\_pointwise & \text{ elif }& reduction= 2,\\
    \frac{\sum loss\_pointwise}{self.size(0)} & \text{ elif }& reduction= 3,\\
    loss\_pointwise & \text{ else. }
    \end{cases}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnKlDivGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnKlDiv` is called to perform computation.

* `aclnnStatus aclnnKlDivGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, bool logTarget, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnKlDiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnKlDivGetWorkspaceSize

* **Parameters:**
  * `self` (aclTensor*, compute input): `self` in the formula. `aclTensor` on the device. The data type must meet the type deduction rules (for details, see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with target, and the shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with target. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>.
     * <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT, FLOAT16.
  * `target` (aclTensor*, compute input): `target` in the formula. The data type must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with self, and the shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with self. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
     <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>.
     <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT, FLOAT16.
  * `reduction` (int64_t, compute input): `reduction` in the formula, which specifies the operation after loss_pointwise is computed.
    - 0: none, indicating that the reduction operation is not performed.
    - 1: mean, indicating that the mean value of loss_pointwise is computed.
    - 2: sum, indicating the sum of loss_pointwise.
    - 3: batchmean, indicating that the average loss of a batch is computed, which is consistent with the mathematical definition of Kullback-Leibler divergence.
  * `logTarget` (bool, compute input): specifies whether the input target data has been computed using the logarithm.
  * `out` (aclTensor*, compute output): `out` in the formula. The data types of out and self must meet the type deduction rules (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). When `reduction` is 0, the shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with `self` and `target`. When `reduction` is not 0, the shape is fixed at (1,). The [data format](../../../docs/en/context/data_format.md) can be ND.
     <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>.
     <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT, FLOAT16.
  * `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.
* **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ````text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, target, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, target, or out is not supported.
                                       2. The shape of self cannot be broadcast with that of target.
                                       3. The data type of self or target cannot be converted into that of out.
                                       4. The shape of self or target is greater than 8D.
  ````

## aclnnKlDiv

- **Parameters:**
  
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnKlDivGetWorkspaceSize.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnKlDiv` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kl_div.h"

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
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> targetShape = {4, 2};
  std::vector<int64_t> outShape = {1};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> targetHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {1};
  int64_t reduction = 1;
  bool log_target = false;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnKlDiv.
  ret = aclnnKlDivGetWorkspaceSize(self, target, reduction, log_target, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKlDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnKlDiv.
  ret = aclnnKlDiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKlDiv failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
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
