# aclnnSilentCheckV2

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

- Description:
  The **aclnnSilentCheckV2** API is added, which has different computation logic from the [aclnnSilentCheck](../../silent_check/docs/aclnnSilentCheck_en.md) API. The **stepRef** parameter is used to determine **val** and compare it with the new Markov inequality threshold for verification.
- Formula:
  
  - If stepRef == 0:
    - If the current input `val` is inf/-inf/nan, or `val` exceeds `avgRef` × `cThreshL1`, the fault is identified as an L1 fault, and an error log is printed. If the environment variable `npuAsdDetect` is set to 1, `avgRef` and `stepRef` are updated, and then the function returns. Otherwise, inputGradRef is set to 0 and resumable training is triggered.
    - If the current input `val` exceeds `avgRef` × `cThreshL2`, the fault is identified as an L2 fault, and a warning log is printed. `avgRef` and `stepRef` are updated, and then the function returns.
  - If stepRef > 0:
    - If the current input `val` is inf/-inf/nan, or `val` exceeds the Markov inequality threshold (avgRef/(1-beta1)^stepRef) × cThreshL1, the fault is identified as an L1 fault, and an error log is printed. If the environment variable `npuAsdDetect` is set to 2, resumable training is triggered. If the environment variable `npuAsdDetect` is set to 1, `avgRef` and `stepRef` are updated, and then the function returns.
    - If the current input `val` exceeds the Markov inequality threshold (avgRef/(1-beta1)^stepRef) × cThreshL2, the fault is identified as an L2 fault, and a warning log is printed. `avgRef` and `stepRef` are updated, and then the function returns.
    - If neither the L1 fault nor the L2 fault is triggered, the following occurs: If `npuAsdDetect` is set to 3, the feature value of `val` is printed. Otherwise, `avgRef` and `stepRef` are updated, and then the function returns.
  - `stepRef` indicates the number of detection times.
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnSilentCheckV2GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnSilentCheckV2** is called to perform computation.

```Cpp

aclnnStatus aclnnSilentCheckV2GetWorkspaceSize(
    const aclTensor *val, 
    const aclTensor *max, 
    aclTensor *avgRef, 
    aclTensor *inputGradRef, 
    aclTensor *stepRef, 
    aclIntArray *dstSize, 
    aclIntArray *dstStride, 
    aclIntArray *dstOffset, 
    float cThreshL1,  
    float cThreshL2, 
    float beta1, 
    int32_t npuAsdDetect, 
    aclTensor* result,
    uint64_t *workspaceSize, 
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnSilentCheckV2(
    void *workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream stream)
```

## aclnnSilentCheckV2GetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
  <col style="width: 145px">
  </colgroup>
    <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>val</td>
      <td>Input</td>
      <td>Current input value.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>avgRef</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputGradRef</td>
      <td>Input</td>
      <td>Gradient tensor input to the model.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stepRef</td>
      <td>Input</td>
      <td>Current step count.</td>
      <td>Negative numbers are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>The value must be [1].</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstSize</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstStride</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstDffset</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL1</td>
      <td>Input</td>
      <td>Absolute threshold for triggering an L1 fault.</td>
      <td>The recommended value is 1000000.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL2</td>
      <td>Input</td>
      <td>Absolute threshold for triggering an L2 fault.</td>
      <td>The recommended value is 10000 and cThreshL1>cThreshL2.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>Input</td>
      <td>-</td>
      <td>The recommended value is 0.99. The value range is 0&ltbeta1&lt1. It is recommended that the value be close to 1.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>npuAsdDetect</td>
      <td>Input</td>
      <td>Environment variable.</td>
      <td>-</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>result</td>
      <td>Output</td>
      <td>Silence check result returned.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The input val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, dstOffset is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the passed val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, or dstOffset is not supported.</td>
    </tr>
    <tr>
      <td>The shape of the passed val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, or dstOffset does not meet the requirements.</td>
    </tr>
    <tr>
      <td>The range of the passed cThreshL1, cThreshL2, or beta1 does not meet the requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnSilentCheck

- **Parameters:**
  
  <table><thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnSilentCheckGetWorkspaceSize.</td>
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
  
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnSilentCheckV2** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_silent_check_v2.h"

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

int64_t GetShapeSize(std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
  }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // (Fixed writing) Initialize ACL.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(std::vector<T>& hostData, std::vector<int64_t>& shape, void** deviceAddr,
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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> valShape = {1};
    std::vector<int64_t> maxShape = {1};
    std::vector<int64_t> avgRefShape = {1};
    std::vector<int64_t> inputGradRefShape = {2, 3};
    std::vector<int64_t> stepRefShape = {1};
    std::vector<int64_t> resultShape = {1};
    void* valDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* avgRefDeviceAddr = nullptr;
    void* inputGradRefDeviceAddr = nullptr;
    void* stepRefDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    aclTensor* val = nullptr;
    aclTensor* max = nullptr;
    aclTensor* avgRef = nullptr;
    aclTensor* inputGradRef = nullptr;
    aclTensor* stepRef = nullptr;
    aclTensor* result = nullptr;
    std::vector<float> valHostData = {160.0};
    std::vector<float> maxHostData = {400.0};
    std::vector<float> avgRefHostData = {200.0};
    std::vector<float> inputGradRefHostData = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> stepRefHostData = {0};
    std::vector<int64_t> dstSizeData = {2, 3};
    std::vector<int64_t> dstStrideData = {3, 1};
    std::vector<int64_t> dstOffsetData = {0};
    std::vector<int32_t> resultHostData = {0};
    aclIntArray* dstSize = aclCreateIntArray(dstSizeData.data(), 2);
    aclIntArray* dstStride = aclCreateIntArray(dstStrideData.data(), 2);
    aclIntArray* dstOffset = aclCreateIntArray(dstOffsetData.data(), 1);
    float cThreshL1 = 1000000;
    float cThreshL2 = 10000;
    float beta1 = 0.99;
    int32_t npuAsdDetect = 3;

    // Create a val aclTensor.
    ret = CreateAclTensor(valHostData, valShape, &valDeviceAddr, aclDataType::ACL_FLOAT, &val);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a max aclTensor.
    ret = CreateAclTensor(maxHostData, maxShape, &maxDeviceAddr, aclDataType::ACL_FLOAT, &max);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an avgRef aclTensor.
    ret = CreateAclTensor(avgRefHostData, avgRefShape, &avgRefDeviceAddr, aclDataType::ACL_FLOAT, &avgRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an inputGradRef aclTensor.
    ret = CreateAclTensor(inputGradRefHostData, inputGradRefShape, &inputGradRefDeviceAddr, aclDataType::ACL_FLOAT, &inputGradRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a stepRef aclTensor.
    ret = CreateAclTensor(stepRefHostData, stepRefShape, &stepRefDeviceAddr, aclDataType::ACL_INT64, &stepRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a result aclTensor.
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_INT32, &result);
    if (result == nullptr) {
        std::cout << "result is nullptr!" << std::endl;
        return 0;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual host API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnSilentCheckV2.
    ret = aclnnSilentCheckV2GetWorkspaceSize(val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, dstOffset, cThreshL1, cThreshL2, beta1, npuAsdDetect, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheckV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnSilentCheckV2.
    ret = aclnnSilentCheckV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheckV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(val);
    aclDestroyTensor(max);
    aclDestroyTensor(avgRef);
    aclDestroyTensor(inputGradRef);
    aclDestroyTensor(stepRef);
    aclDestroyIntArray(dstSize);
    aclDestroyIntArray(dstStride);
    aclDestroyIntArray(dstOffset);
    aclDestroyTensor(result);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(valDeviceAddr);
    aclrtFree(maxDeviceAddr);
    aclrtFree(avgRefDeviceAddr);
    aclrtFree(inputGradRefDeviceAddr);
    aclrtFree(stepRefDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
