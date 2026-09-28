# aclnnConfusionTranspose

## Supported Products

| Product                                             | Supported|
|:------------------------------------------------| :------: |
| Ascend 950PR/Ascend 950DT         |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>   |    x     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    x     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                      |    x     |
| <term>Atlas training products</term>                      |    x     |

## Function

- API function:

  Fuses the reshape and transpose operations.

- Formulas:
  
  1. When transposeFirst is set to False:

     $$
     y=transpose(reshape(x,shape),perm)
     $$
  2. When transposeFirst is set to True:
  
     $$
     y=reshape(transpose(x,perm),shape)
     $$
     
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnConfusionTransposeGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnConfusionTranspose` is called to perform computation.

```Cpp
aclnnStatus aclnnConfusionTransposeGetWorkspaceSize( 
    const aclTensor    *x, 
    const aclIntArray  *perm, 
    const aclIntArray  *shape, 
    bool                transposeFirst, 
    aclTensor          *out,
    uint64_t           *workspaceSize, 
    aclOpExecutor      **executor)
```

```Cpp 
aclnnStatus aclnnConfusionTranspose(
    void               *workspace, 
    uint64_t            workspaceSize, 
    aclOpExecutor      *executor, 
    aclrtStream         stream)
```
   
## aclnnConfusionTransposeGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-continuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">x (aclTensor*) </td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Input tensor, corresponding to x in the formula.</td>
      <td class="tg-0pky">An empty tensor is supported.</td>
      <td class="tg-0pky">INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">perm (aclIntArray*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Index of each axis before transposition, corresponding to perm in the formula.</td>
      <td class="tg-0pky">1. The elements in this input must be unique and within the range of [0, Number of dimensions of perm – 1].<br>2. When transposeFirst is set to True, the length of perm must be the same as that of x_shape, that is, len(perm) = len(x_shape).<br>3. When transposeFirst is set to False, the length of perm must be the same as that of the attribute input shape, that is, len(perm) = len(shape).</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">shape (aclIntArray*) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Shape size after reshape, corresponding to the shape in the formula.</td>
      <td class="tg-0lax">The product of all dimensions in the shape must be equal to the total number of elements in the input tensor x.</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">transposeFirst (bool) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether to perform the transpose operation first.</td>
      <td class="tg-0lax">If the value is True, transpose is performed first. Otherwise, reshape is performed first.</td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out (aclTensor*) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Indicates the computation result after reshape and transpose.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">Same as the input x.</td>
      <td class="tg-0lax">Same as the input x.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize(uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator execution process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input x or output out is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">The data types of the input x and output out are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0lax">ACLNN_ERR_INNER_NULLPTR</td>
      <td class="tg-0lax">561103</td>
      <td class="tg-0lax">Internal verification error of the API. Generally, this is caused by the fact that the specifications of the input data or attributes are not supported, or the input and output shapes do not meet the requirements described in the parameter description.</td>
    </tr>
  </tbody>
  </table>

## aclnnConfusionTranspose

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnConfusionTransposeGetWorkspaceSize.</td>
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

* Deterministic description: The aclnnConfusionTranspose function is implemented in a deterministic manner by default.

  For example:
      
      Assume that shape_before is the data shape before the reshape operation, and shape_after is the data shape after the reshape operation,

          shape_before = [(ab),(cd),f,(gh)]
          shape_after = [a,(bc),d,e,(fg),h]
      
      The following shape_after is not allowed:

          shape_after_illegal = [a,b,d,e,(fg),(ch)]

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_confusion_transpose.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.

  // Create an input aclTensor.
  aclTensor* x = nullptr;
  std::vector<int64_t> xShape = {2, 4}; 
  std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  void* xDeviceAddr = nullptr;
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create perm.
  aclIntArray* perm = nullptr;
  std::vector<int64_t> permData = {1, 0};
  perm = aclCreateIntArray(permData.data(), permData.size()); 
  CHECK_RET(perm != nullptr, return ret);

  // Create shape.
  aclIntArray* shape = nullptr;
  std::vector<int64_t> shapeData = {2, 4};
  shape = aclCreateIntArray(shapeData.data(), shapeData.size()); 
  CHECK_RET(shape != nullptr, return ret);

  // Create transposeFirst.
  bool transposeFirst = true;

  // Create an output aclTensor.
  std::vector<int64_t> outShape = {2, 4};
  std::vector<float> outHostData(8, 1);
  aclTensor* out = nullptr;
  void* outDeviceAddr = nullptr;
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // Call the first part of the aclnnConfusionTranspose API.
  ret = aclnnConfusionTransposeGetWorkspaceSize(x, perm, shape, transposeFirst, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConfusionTransposeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second part of the aclnnConfusionTranspose API.
  ret = aclnnConfusionTranspose(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConfusionTranspose failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensors. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(xDeviceAddr);
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
