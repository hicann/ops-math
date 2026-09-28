# aclnnSearchSorteds

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/search_sorted)

## Supported Products

| Product| Supported|
| :--- | :------: |
| Ascend 950PR/Ascend 950DT| × |
| <term>Atlas A3 training products/Atlas A3 inference products</term>| √ |
| <term>Atlas A2 training products/Atlas A2 inference products</term>| √ |
| <term>Atlas 200I/500 A2 inference products</term>| × |
| <term>Atlas inference products</term>| × |
| <term>Atlas training products</term>| √ |

## Function

- Operator function: searches for the position where a given scalar value (self) should be inserted into a sorted 1D tensor (sortedSequence). Returns a tensor with shape [1], indicating the position where the given scalar value should be inserted into the original tensor. If self is of the tensor type, see the [aclnnSearchSorted](./aclnnSearchSorted.md) document.
- The calculation formula is as follows: Assume that the length of the sequence to be searched is $N$, and the scalar input is $x = self$.
  - When `right=false`, the left insertion point is returned.

  $$
  out=\min\{j\in[0,N]\mid sortedSequence_j\ge x\}
  $$

  - When `right=true`, the right insertion point is returned.

  $$
  out=\min\{j\in[0,N]\mid sortedSequence_j>x\}
  $$

  If no $j$ that meets the condition exists, $N$ is returned.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, the `aclnnSearchSortedsGetWorkspaceSize` API is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, the `aclnnSearchSorteds` API is called to perform computation.

```Cpp
aclnnStatus aclnnSearchSortedsGetWorkspaceSize(
  const aclTensor* sortedSequence,
  const aclScalar* self,
  bool             outInt32,
  bool             right,
  const aclTensor* sorter,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSearchSorteds(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnSearchSortedsGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sortedSequence (aclTensor*) </td>
      <td>Input</td>
      <td>One-dimensional input tensor that has been sorted.</td>
      <td><ul><li>Must be a one-dimensional tensor. </li><li>The data types of <code>self</code> and <code>other</code> must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>). </li><li>In the formula, <code>N</code> indicates the length of <code>sortedSequence</code>, and <code>sortedSequence<sub>j</sub></code> indicates the <code>j</code>th element.</li></ul></td>
      <td>DOUBLE, FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclScalar*) </td>
      <td>Input</td>
      <td>Scalar value whose insertion position is to be searched for.</td>
      <td><ul><li>Scalar on the host. </li><li>The data types of <code>sortedSequence</code> and <code>sortedSequence</code> must meet the requirements described in <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>. </li><li>In the formula, <code>x=self</code>.</li></ul></td>
      <td>DOUBLE, FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64</td>
      <td>-</td>
      <td>Scalar</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outInt32 (bool) </td>
      <td>Input</td>
      <td>Whether to output the INT32 result.</td>
      <td>Data type of the output index.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>right (bool) </td>
      <td>Input</td>
      <td>The left/right insertion point is returned when the equal value is hit.</td>
      <td><ul><li><code>false</code> corresponds to <code>sortedSequence<sub>j</sub> &ge; x</code> in the formula. </li><li><code>true</code> corresponds to <code>sortedSequence<sub>j</sub> &gt; x</code>.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sorter (aclTensor*) </td>
      <td>Input</td>
      <td>Specifies the order of the <code>sortedSequence</code> elements.</td>
      <td><ul><li>The data type must be INT64. </li><li>The shape must be the same as that of <code>sortedSequence</code>. </li><li>When <code>sorter</code> is passed, <code>sortedSequence<sub>j</sub></code> in the formula is assigned values in the order specified by <code>sorter</code>.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>Same as sortedSequence</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>Output result of the insertion position.</td>
      <td><ul><li>The output is the position index. </li><li>The <code>out</code> in the formula corresponds to the output parameter <code>out</code>.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>Generally, it is [1]</td>.
      <td>-</td>
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

- **Return Value**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input sortedSequence, self, or out contains a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of sortedSequence and self are not supported.</td>
    </tr>
    <tr>
      <td>The data type of out is inconsistent with the meaning of outInt32.</td>
    </tr>
    <tr>
      <td>If the data types of sortedSequence and self are different, data type deduction cannot be performed.</td>
    </tr>
    <tr>
      <td>The input sorter is not of the INT64 type.</td>
    </tr>
    <tr>
      <td>The shape of sorter is different from that of sortedSequence.</td>
    </tr>
    <tr>
      <td>sortedSequence is not a 1D tensor.</td>
    </tr>
  </tbody></table>

## aclnnSearchSorteds

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size obtained by the first API <code>aclnnSearchSortedsGetWorkspaceSize</code>.</td>
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

- **Return Value**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description: The default deterministic implementation of `aclnnSearchSorteds` is used.

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

#define LOG_PRINT(message,...)     \
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
