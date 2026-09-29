# aclnnEqual

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/tensor_equal)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description: Determines whether two tensors have the same size and elements, and returns a Boolean value.
- Formula:

  $$
  out = (self == other)  ?  True : False
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnEqualGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnEqual` is called to perform computation.

```Cpp
aclnnStatus aclnnEqualGetWorkspaceSize(
  const aclTensor* self, 
  const aclTensor* other, 
  aclTensor*       out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnEqual(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnEqualGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1549px"><colgroup>
  <col style="width: 168px">
  <col style="width: 136px">
  <col style="width: 258px">
  <col style="width: 271px">
  <col style="width: 311px">
  <col style="width: 116px">
  <col style="width: 142px">
  <col style="width: 147px">
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
      <td>self</td>
      <td>Input</td>
      <td>The first input.</td>
      <td>The data types of <code>self</code> and <code>other</code> must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, BOOL, DOUBLE, INT64, INT16, UINT16, UINT32, UINT64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>The second input.</td>
      <td>The data types of <code>other</code> and <code>self</code> must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, BOOL, DOUBLE, INT64, INT16, UINT16, UINT32, UINT64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output, a 1D BOOL tensor containing a single element.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
  </tbody>
  </table>

  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type cannot be BFLOAT16.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 750px">
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
      <td>The passed <code>self</code> or <code>other</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type deduced from <code>self</code> and <code>other</code> is not supported.</td>
    </tr>
    <tr>
      <td>The number of dimensions of <code>self</code>, <code>other</code>, or <code>out</code> exceeds 8.</td>
    </tr>
    <tr>
      <td>The shape of <code>out</code> is not [1].</td>
    </tr>
  </tbody>
  </table>

## aclnnEqual

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnEqualGetWorkspaceSize</code>.</td>
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

## Constraints

- Deterministic computation:
  - `aclnnEqual` defaults to a deterministic implementation.
- Excessive computation load may cause the operator to time out (AI Core error, errorStr: "timeout or trap error"). This typically occurs when the product of the last two dimensions is less than 16 while the product of the leading dimensions is excessively large.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_equal.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides of the contiguous tensor.
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

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,
    void** selfDeviceAddr, void** otherDeviceAddr, void** outDeviceAddr, aclTensor** self, aclTensor** other,
    aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<char> outHostData = {0};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(otherHostData, otherShape, otherDeviceAddr, aclDataType::ACL_DOUBLE, other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_BOOL, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* other, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnEqualGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqualGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnEqual(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEqual failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(outShape);
  std::vector<char> resultData(size, 0);

  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(char),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;

  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {1};

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(
      selfShape, otherShape, outShape, &selfDeviceAddr, &otherDeviceAddr, &outDeviceAddr, &self, &other, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, other, out, &workspaceAddr, workspaceSize, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

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
