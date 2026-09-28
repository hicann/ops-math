# aclnnOneHot

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/one_hot)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                      |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>      |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    √    |

## Function

- This API is used to perform one_hot computation on the input self with length n to obtain an output out with the number of elements being n x k, where k is the value of numClasses.

- Formulas:

  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnOneHotGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnOneHot` is called to perform computation.

```Cpp
aclnnStatus aclnnOneHotGetWorkspaceSize(
  const aclTensor* self,
  int              numClasses,
  const aclTensor* onValue,
  const aclTensor* offValue,
  int64_t          axis,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnOneHot(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnOneHotGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 320px">
  <col style="width: 300px">
  <col style="width: 120px">
  <col style="width: 130px">
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
      <td>self (aclTensor*) </td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>-</td>
      <td>UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numClasses (int) </td>
      <td>Input</td>
      <td>Number of classes.</td>
      <td><ul><li>If self is an empty tensor, numClasses must be greater than 0.</li><li>If self is not an empty tensor, numClasses must be greater than or equal to 0. </li><li>If numClasses is 0, an empty tensor is returned. </li><li>If the value of self is greater than numClasses, these elements will be encoded into all offValues.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onValue (aclTensor*) </td>
      <td>Input</td>
      <td>Padding value at the index position, which is the onValue in the formula.</td>
      <td><ul><li>Only the first element value is used for computation. </li><li>The data type is the same as that of out.</li></ul></td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offValue (aclTensor*) </td>
      <td>Input</td>
      <td>Padding value at the non-index position, which is the offValue in the formula.</td>
      <td><ul><li>Only the first element value is used for calculation. </li><li>The data type is the same as that of out.</li></ul></td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t) </td>
      <td>Input</td>
      <td>Insertion dimension of the encoding vector.</td>
      <td><ul><li>The minimum value is -1 and the maximum value is the number of dimensions of self. </li><li>If the value is -1, the encoding vector is inserted into the last dimension of self.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>One-hot tensor, which is the output out in the formula.</td>
      <td>The output has one more dimension than self. The output shape is the same as the shape of self after numClasses is inserted into the axis.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
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
  </tbody>
  </table>

  - <term>Atlas A3 training products/Atlas A3 inference products</term>: The UINT8 data type is not supported.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
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
      <td>The input self, onValue, offValue, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of self, onValue, offValue, or out is not supported.</td>
    </tr>
    <tr>
      <td>The data types of onValue and offValue are inconsistent with that of out.</td>
    </tr>
    <tr>
      <td>self is an empty tensor, and numClasses is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>self is not an empty tensor, and numClasses is less than 0.</td>
    </tr>
    <tr>
      <td>The value of axis is less than -1.</td>
    </tr>
    <tr>
      <td>The value of axis is greater than the number of dimensions of self.</td>
    </tr>
    <tr>
      <td>The number of dimensions of out is not one more than that of self.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with the shape of self after numClasses is inserted into the axis.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self, onValue, offValue, or out exceeds 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnOneHot

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
    <col style="width: 872px">
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
        <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnOneHotGetWorkspaceSize API.</td>
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

- Deterministic computing:
  - `aclnnOneHot` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_one_hot.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
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

    // Calculate the strides of consecutive tensors.
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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
      // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on the API definition.
    std::vector<int64_t> selfShape = {4, 2};
    int numClasses = 4;
    std::vector<int64_t> outShape = {4, 2, 4};
    std::vector<int64_t> onValueShape = {1};
    std::vector<int64_t> offValueShape = {1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* onValueDeviceAddr = nullptr;
    void* offValueDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclTensor* onValue = nullptr;
    aclTensor* offValue = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 3, 2, 1, 0};
    std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> onValueHostData = {1};
    std::vector<int32_t> offValueHostData = {0};
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an onValue aclTensor.
    ret = CreateAclTensor(onValueHostData, onValueShape, &onValueDeviceAddr, aclDataType::ACL_INT32, &onValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an offValue aclTensor.
    ret = CreateAclTensor(offValueHostData, offValueShape, &offValueDeviceAddr, aclDataType::ACL_INT32, &offValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    int64_t axis = -1;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnoneHot.
    ret = aclnnOneHotGetWorkspaceSize(self, numClasses, onValue, offValue, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnOnehot.
    ret = aclnnOneHot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHot failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(onValue);
    aclDestroyTensor(offValue);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(onValueDeviceAddr);
    aclrtFree(offValueDeviceAddr);
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
