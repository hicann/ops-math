# aclnnLogicalNot&aclnnInplaceLogicalNot

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/logical_not)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

Description: Computes the element-wise logical NOT of the given input tensor. If no output type is specified, the output tensor is of the Boolean type. If the input tensor is not of the Boolean type, `0` is considered as `False`, and a non-zero value is considered as `True`.

## Prototype

- `aclnnLogicalNot` and `aclnnInplaceLogicalNot` implement the same function in different ways. Select a proper operator based on your requirements.
  - `aclnnLogicalNot`: An output tensor object needs to be created to store the computation result.
  - `aclnnInplaceLogicalNot`: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLogicalNotGetWorkspaceSize` or `aclnnInplaceLogicalNotGetWorkspaceSize` is called to obtain input parameters and compute the required workspace size based on the computation process. Then, `aclnnLogicalNot` or `aclnnInplaceLogicalNot` is called to perform computation.

  * `aclnnStatus aclnnLogicalNotGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnLogicalNot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceLogicalNotGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnInplaceLogicalNot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLogicalNotGetWorkspaceSize

- **Parameters:**
  
  * `self` (aclTensor\*, compute input): input tensor, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, or DOUBLE.
    * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  * `out` (aclTensor\*, compute output): output tensor, aclTensor on the device. The shape is the same as that of `self`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, or DOUBLE.
    * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                    2. The shapes of self and out are inconsistent.
  ```

## aclnnLogicalNot

- **Parameters:**
  
  * `workspace` (void\*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnLogicalNotGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLogicalNotGetWorkspaceSize

- **Parameters:**
  
  * `selfRef` (aclTensor\*, compute input): input tensor, aclTensor on the device. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND.
    * <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type can be BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, or DOUBLE.
    * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, DOUBLE, or BFLOAT16.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef is not supported.
  ```

## aclnnInplaceLogicalNot

- **Parameters:**
  
  * `workspace` (void\*, input): memory address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnInplaceLogicalNotGetWorkspaceSize`.
  * `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - `aclnnLogicalNot` and `aclnnInplaceLogicalNot` default to deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_logical_not.h"
#include <iostream>
#include <vector>

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

    // Call aclrtMemcpy to copy host data to the device memory.
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> selfShape = {3, 3};
    std::vector<int64_t> outShape = {3, 3};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    
    std::vector<int> selfHostData = {0, 1, 0, 2, 0, 3, 4, 0, 0};
    std::vector<int> outHostData ={0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    
    // aclnnLogicalNot API call example
    // 3. Call the first-phase API of aclnnLogicalNot.
    ret = aclnnLogicalNotGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogicalNotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnLogicalNot.
    ret = aclnnLogicalNot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogicalNot failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modification is required based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, 
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
    
    // aclnnInplaceLogicalNot API call example
    // 3. Call the first-phase API of aclnnInplaceLogicalNot.
    ret = aclnnInplaceLogicalNotGetWorkspaceSize(self, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLogicalNotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnInplaceLogicalNot.
    ret = aclnnInplaceLogicalNot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLogicalNot failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modification is required based on the specific API definition.
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, 
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
    
    // 6. Release aclTensor and aclScalar. Modification is required based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    
    // 7. Release device resources.
    aclrtFree(selfDeviceAddr);
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
