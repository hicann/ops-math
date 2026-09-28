# aclnnCdistBackward

## Supported Products

| Product                                                                                   | Supported|
| :-------------------------------------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT         |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>   |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                      |    ×     |
| <term>Atlas training products</term>                      |    ×     |

## Function

- API function: Implements the backward propagation of aclnnCdist.
- Formulas:

  $$
  \begin{aligned}
  out&=grad \cdot y' \\
  &= grad \cdot \left( \sqrt[p]{\sum (x_1 - x_2)^p} \right)' \\
  &= grad \cdot \frac{1}{p} \times \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p}-1} \times p \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{-(p-1)}{p}} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p} \times (-(p-1))} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \frac{diff^{p-1}}{cdist^{p-1}} \\
  &= grad \cdot \frac{diff \times |diff|^{p-2}}{cdist^{p-1}}
  \end{aligned}
  $$

  - $\mathrm{diff} = x_1 - x_2$: difference between variables.
  - $\mathrm{cdist} = \sqrt[p]{\sum (x_1 - x_2)^p}$: p-norm distance

## Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). You must call aclnnCdistBackwardGetWorkspaceSize to obtain the input parameters, calculate the required workspace size based on the workflow, and then call aclnnCdistBackward to perform the computation.

```Cpp
aclnnStatus aclnnCdistBackwardGetWorkspaceSize(
    const aclTensor *grad,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *cdist,
    float            p,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCdistBackward(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnCdistBackwardGetWorkspaceSize

- **Parameter description**:
  
  <table style="undefined;table-layout: fixed; width: 1510px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 140px">
    <col style="width: 150px">
    <col style="width: 119px">
    <col style="width: 280px">
    <col style="width: 146px">
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
        <td>grad (aclTensor*) </td>
        <td>Input</td>
        <td>Gradient in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2 to 7 dimensions.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1 (aclTensor*) </td>
        <td>Input</td>
        <td> `x1` in the formula.</td>
        <td>The data type is the same as that of grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The dimension is the same as that of grad. Except the last dimension, the shape is the same as that of grad.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (aclTensor*) </td>
        <td>Input</td>
        <td>X2 in the formula.</td>
        <td>The data type is the same as that of grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The dimensions are equal to those of grad. The second-to-last dimension is equal to the last dimension of grad. The last dimension is equal to the last dimension of x1. Other dimensions must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cdist (aclTensor*) </td>
        <td>Input</td>
        <td>cdist in the formula.</td>
        <td>The data type is the same as that of grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The shape is the same as that of grad.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>p (float) </td>
        <td>Attribute</td>
        <td>p in the formula.</td>
        <td>-</td>
        <td>float</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*) </td>
        <td>Output</td>
        <td><code>out</code> in the formula.</td>
        <td>The data type is the same as that of grad.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The shape is the same as that of x1.</td>
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
    </tbody>
    </table>
- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.
  
  <table style="undefined;table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 291px">
    <col style="width: 112px">
    <col style="width: 707px">
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
        <td>The input grad, x1, x2, or cdist is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The data type of grad, x1, x2, or cdist is not supported.</td>
      </tr>
      <tr>
        <td>The shape of grad, x1, x2, or cdist is not supported.</td>
      </tr>
    </tbody>
    </table>

## aclnnCdistBackward

- **Parameter description**:
  
  <table style="undefined;table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
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
        <td>Size of the workspace allocated on the device, which is obtained by the first interface aclnnCdistBackwardGetWorkspaceSize.</td>
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
- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

Deterministic computing is supported by default.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cdist_backward.h"

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

    // Compute the strides of the contiguous tensor.
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
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    int B=5, P=7, Q=9, M=11;
    // 2.Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> gradShape = {B, P, Q};
    std::vector<int64_t> x1Shape = {B, P, M};
    std::vector<int64_t> x2Shape = {B, Q, M};
    std::vector<int64_t> cdistShape = {B, P, Q};
    std::vector<int64_t> outShape = {B, P, M};
    void *gradDeviceAddr = nullptr;
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *cdistDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *grad = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *cdist = nullptr;
    aclTensor *out = nullptr;
    float p = 0.5;
    std::vector<float> gradHostData(B * P * Q, 1);
    std::vector<float> x1HostData(B * P * M, 2);
    std::vector<float> x2HostData(B * Q * M, 4);
    std::vector<float> cdistHostData(B * P * Q, 0.5);
    std::vector<float> outHostData(B * P * M, 1);
    // Create a grad aclTensor.
    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x1 aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x2 aclTensor.
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the cdist aclTensor.
    ret = CreateAclTensor(cdistHostData, cdistShape, &cdistDeviceAddr, aclDataType::ACL_FLOAT, &cdist);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first part of the aclnnCdistBackward API.
    ret = aclnnCdistBackwardGetWorkspaceSize(grad, x1, x2, cdist, p, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second part of the aclnnCdistBackward API.
    ret = aclnnCdistBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(cdist);
    aclDestroyTensor(grad);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(cdistDeviceAddr);
    aclrtFree(gradDeviceAddr);
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
