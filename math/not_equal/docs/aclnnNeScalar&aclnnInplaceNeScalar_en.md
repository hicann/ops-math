# aclnnNeScalar&aclnnInplaceNeScalar

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/not_equal)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

* Operator function: Computes whether the element value in `selfRef` is not equal to the value of `other`.
* Formula:

$$
out_i​=(self_i \ne other)?[1]:[0]
$$

$$
selfRef_i​=(selfRef_i \ne other)?[1]:[0]
$$

## Prototype

* `aclnnNeScalar` and `aclnnInplaceNeScalar` implement the same function in different ways. Select a proper operator based on your requirements.
  * `aclnnNeScalar`: An output tensor object needs to be created to store the computation result.
  * `aclnnInplaceNeScalar`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
* Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnNeScalarGetWorkspaceSize` or `aclnnInplaceNeScalarGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the computation process. Then, `aclnnNeScalar` or `aclnnInplaceNeScalar` is called to perform computation.
  * `aclnnStatus aclnnNeScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnNeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  * `aclnnStatus aclnnInplaceNeScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceNeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNeScalarGetWorkspaceSize

* **Parameters:**
  * `self` (aclTensor*, computation input): `self` in the formula, which is an aclTensor on the device. The shape dimensions cannot be greater than 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
    * <term>Atlas training products </term>: The data type can be DOUBLE, FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
  * `other` (aclScalar*, computation input): `other` in the formula, which is an aclScalar on the host.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`.
    * <term>Atlas training products </term>: The data type can be DOUBLE, FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `self`.
  * `out` (aclTensor*, computation output): `out` in the formula, which is an aclTensor on the device. Data types that can be converted from BOOL. For details, see [conversion relationship](../../../docs/en/context/conversion_relationship.md). The shape is the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128.
    * <term>Atlas training products</term>: The data type can be DOUBLE, FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
* **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input self, other, and out are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, other, or out is not supported.
                                        2. The data types of self and other do not meet the type deduction rules.
                                        3. The shape of self is different from that of out.
                                        4. The dimensions of self and out are greater than 8.
  ```

## aclnnNeScalar

* **Parameters:**
  - `workspace` (void\*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnNeScalarGetWorkspaceSize`.
  - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.
* **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceNeScalarGetWorkspaceSize

* **Parameters:**
  * `selfRef` (aclTensor*, computation input/output): `selfRef` in the formula, which is an aclTensor on the device. The shape dimensions cannot be greater than 8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
    * <term>Atlas training products </term>: The data type can be DOUBLE, FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `other`.
  * other (aclScalar*, computation input): input `other` in the formula, which is an aclScalar on the host.
    * <term>Atlas A2 training products/Atlas A2 inference products </term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`.
    * <term>Atlas training products </term>: The data type can be DOUBLE, FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, or COMPLEX128, and must meet the deduction relationship (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) with `selfRef`.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
* **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. selfRef or other is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of selfRef and other are not supported.
                                        2. The data types of selfRef and other do not meet the data type deduction rules.
                                        3. The dimensions of selfRef are greater than 8.
  ```

## aclnnInplaceNeScalar

* **Parameters**
  + `workspace`: address of the workspace to be allocated on the device.
  + `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnInplaceNeScalarGetWorkspaceSize.
  + `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  + `stream` (aclrtStream, input): stream for executing the task.
* **Returns:**
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnNeScalar&aclnnInplaceNeScalar` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ne_scalar.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // (Fixed writing) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of consecutive tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void *selfDeviceAddr = nullptr;
    void *otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclScalar *other = nullptr;
    aclTensor *out = nullptr;
    std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    double otherValue = 1.0f;

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an other aclScalar.
    other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
 
    // aclnnNeScalar API call example
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnNeScalar.
    ret = aclnnNeScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNeScalar.
    ret = aclnnNeScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeScalar failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(selfShape);
    std::vector<double> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // aclnnInplaceNeScalar API call example
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    LOG_PRINT("\ntest aclnnInplaceNeScalar\n");
    // Call the first-phase API of aclnnInplaceNeScalar.
    ret = aclnnInplaceNeScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNeScalar.
    ret = aclnnInplaceNeScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeScalar failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        selfDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyScalar(other);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr); 
    aclrtFree(otherDeviceAddr); 
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
