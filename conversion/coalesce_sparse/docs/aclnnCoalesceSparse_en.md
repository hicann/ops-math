# aclnnCoalesceSparse

[📄 View Source Code](https://gitcode.com/cann/ops-math/tree/master/conversion/coalesce_sparse)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Adds up the values at the same coordinates to reduce the memory size of the Coo_Tensor.

## Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the aclnnCoalesceSparseGetWorkspaceSize interface to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the aclnnCoalesceSparse interface to perform computation.

```Cpp
aclnnStatus aclnnCoalesceSparseGetWorkspaceSize(
    const aclTensor   *uniqueLen,
    const aclTensor   *uniqueIndices,
    const aclTensor   *indices,
    const aclTensor   *values,
    const aclTensor   *newIndicesOut,
    const aclTensor   *newValuesOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```Cpp
aclnnStatus aclnnCoalesceSparse(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream);
```

## aclnnCoalesceSparseGetWorkspaceSize

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1519px"><colgroup>
  <col style="width: 217px">
  <col style="width: 120px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 145px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>uniqueLen (aclTensor*) </td>
      <td>Input</td>
      <td>Number of indexes after deduplication.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32 or INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>uniqueIndices (aclTensor*) </td>
      <td>Input</td>
      <td>Deduplicated index array.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32 or INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices (aclTensor*) </td>
      <td>Input</td>
      <td>Index array.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The reindexed indices cannot exceed the upper limit of int32.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>values (aclTensor*) </td>
      <td>Input</td>
      <td>Element value corresponding to each coordinate.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>newIndicesOut (aclTensor*) </td>
      <td>Output</td>
      <td>Merged index array.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32 or INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>newValuesOut (aclTensor*) </td>
      <td>Output</td>
      <td>Element value after combination.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
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

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input uniqueLen, uniqueIndices, indices, values, newIndicesOut, or newValuesOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of uniqueLen, uniqueIndices, indices, values, newIndicesOut, or newValuesOut is not supported.</td>
    </tr>
    <tr>
      <td>The number of dimensions of values or newValuesOut exceeds 8.</td>
    </tr>
    <tr>
      <td>The re-indexed indices value cannot exceed the upper limit of int32.</td>
    </tr>
  </tbody></table>

## aclnnCoalesceSparse

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnCoalesceSparseGetWorkspaceSize API.</td>
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
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

None

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_coalesce_sparse.h"

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
    // (Boilerplate) Perform initialization.
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
    // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
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

int main()
{
      // 1. Boilerplate code for device/stream initialization. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> uniqueLenShape = {1};
    std::vector<int64_t> uniqueIndicesShape = {2,2};
    std::vector<int64_t> indexShape = {2,4};
    std::vector<int64_t> valueShape = {4};
    std::vector<int64_t> newIndexShape = {4};
    std::vector<int64_t> newValueShape = {2};
    void* uniqueLenDeviceAddr = nullptr;
    void* uniqueIndicesDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* newIndexDeviceAddr = nullptr;
    void* newValueDeviceAddr = nullptr;
    aclTensor* uniqueLen = nullptr;
    aclTensor* uniqueIndices = nullptr;
    aclTensor* index = nullptr;
    aclTensor* value = nullptr;
    aclTensor* newIndex = nullptr;
    aclTensor* newValue = nullptr;
    std::vector<int32_t> uniqueLenData = {2};
    std::vector<int32_t> uniqueIndicesData = {0, 1, 0, 2};
    std::vector<int32_t> indexData = {0, 0, 1, 1, 0, 0, 2, 2};
    std::vector<float> valueData = {1, 2, 3, 4};
    std::vector<int32_t> newIndexData = {0, 0, 0, 0};
    std::vector<float> newValueData = {0, 0};

    // Create an in aclTensor.
    ret = CreateAclTensor(uniqueLenData, uniqueLenShape, &uniqueLenDeviceAddr, aclDataType::ACL_INT32, &uniqueLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an in aclTensor.
    ret = CreateAclTensor(uniqueIndicesData, uniqueIndicesShape, &uniqueIndicesDeviceAddr, aclDataType::ACL_INT32, &uniqueIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an in aclTensor.
    ret = CreateAclTensor(indexData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an in aclTensor.
    ret = CreateAclTensor(valueData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(newIndexData, newIndexShape, &newIndexDeviceAddr, aclDataType::ACL_INT32, &newIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(newValueData, newValueShape, &newValueDeviceAddr, aclDataType::ACL_FLOAT, &newValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first API of aclnnCoalesceSparse.
    ret = aclnnCoalesceSparseGetWorkspaceSize(uniqueLen, uniqueIndices, index, value, newIndex, newValue, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCoalesceSparseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second phase of aclnnCoalesceSparse.
    ret = aclnnCoalesceSparse(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCoalesceSparse failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(newValueShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), newValueDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(uniqueLen);
    aclDestroyTensor(uniqueIndices);
    aclDestroyTensor(index);
    aclDestroyTensor(value);
    aclDestroyTensor(newIndex);
    aclDestroyTensor(newValue);

    // 7. Free device resources.
    aclrtFree(uniqueLenDeviceAddr);
    aclrtFree(uniqueIndicesDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(newIndexDeviceAddr);
    aclrtFree(newValueDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
