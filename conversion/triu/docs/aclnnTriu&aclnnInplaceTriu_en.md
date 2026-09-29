# aclnnTriu&aclnnInplaceTriu

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/triu)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |     √      |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     ×     |
| <term>Atlas inference products</term>                            |   √     |
| <term>Atlas training products</term>                             |   √     |

## Function

- Description: Sets the elements of the input tensor self (last two dimensions) to 0 in the lower triangular part below the main diagonal. The diagonal parameter can be set to a positive or negative value. The default value is 0. A positive value indicates diagonals above the main diagonal. A negative value indicates diagonals below the main diagonal.
- Formula: In the following formula, i indicates the sequence number of the element in the second-to-last dimension (i is the row index), j indicates the sequence number of the element in the last dimension (j is the column index), and d indicates diagonal. In the two-dimensional coordinate diagram corresponding to (i, j), i+d==j indicates that the element is on the diagonal.

  $$
  For the diagonal and its upper triangular part, that is, i + d < = j, the original value is retained: out_{i, j} = self_{i, j}\\.
  For the lower triangular part, that is, i + d > j, the value is set to zero (excluding the diagonal): out_{i, j} = 0.
  $$

- Example:

  $self = \begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$,
  The result of triu(self, diagonal=0) is as follows:
  $\begin{bmatrix} [9&6&3] \\ [0&2&3] \\ [0&0&1] \end{bmatrix}$;
  Adjust the value of diagonal. The result of triu(self, diagonal=1) is as follows:
  $\begin{bmatrix} [0&6&3] \\ [0&0&3] \\ [0&0&0] \end{bmatrix}$;
  Adjust the value of diagonal to -1. The result of triu(self, diagonal=-1) is as follows:
  $\begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [0&4&1] \end{bmatrix}$.

## Prototype

  - **aclnnTriu** and **aclnnInplaceTriu** implement the same function in different ways. Select a proper operator based on your requirements.
    - **aclnnTriu**: An output tensor object needs to be created to store the computation result.
    - **aclnnInplaceTriu**: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
  - Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTriuGetWorkspaceSize` or `aclnnInplaceTriuGetWorkspaceSize` is called to obtain input parameters and calculate the required workspace size based on the computation process. Then, `aclnnTriu` or `aclnnInplaceTriu` is called to perform computation.

```cpp
aclnnStatus aclnnTriuGetWorkspaceSize(
  const aclTensor* self, 
  int64_t          diagonal, 
  aclTensor*       out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnTriu(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```    

```cpp
aclnnStatus aclnnInplaceTriuGetWorkspaceSize(
  aclTensor*      selfRef, 
  int64_t         diagonal, 
  uint64_t*       workspaceSize, 
  aclOpExecutor** executor)
```  

```cpp
aclnnStatus aclnnInplaceTriu(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
``` 

## aclnnTriuGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 349px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td><code>self</code> in the formula.</td>
      <td> - </td>
      <td>UINT64, INT64, UINT32, INT32, UINT16, INT16, UINT8, INT8, FLOAT16, FLOAT32, DOUBLE, BOOL, BFLOAT16, COMPLEX32, COMPLEX64</td>
      <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
      <td>2–8D and 0D</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonal (int64_t)</td>
      <td>Input</td>
      <td>Diagonal offset.</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>-</td>
      <td>UINT64, INT64, UINT32, INT32, UINT16, INT16, UINT8, INT8, FLOAT16, FLOAT32, DOUBLE, BOOL, BFLOAT16, COMPLEX32, COMPLEX64</td>
      <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
      <td>2–8D and 0D</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - Ascend 950PR/Ascend 950DT: Only COMPLEX32 and COMPLEX64 are supported.
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type cannot be BFLOAT16.

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 230px">
  <col style="width: 127px">
  <col style="width: 851px">
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
      <td><code>self</code> or <code>out</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of <code>self</code> or <code>out</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data format of <code>self</code> or <code>out</code> is private.</td>
    </tr>
    <tr>
      <td>The data types of <code>self</code> and <code>out</code> are different.</td>
    </tr>
    <tr>
      <td>The shapes of <code>self</code> and <code>out</code> are different.</td>
    </tr>
    <tr>
      <td>The data formats of <code>self</code> and <code>out</code> are different.</td>
    </tr>
    <tr>
      <td>The dimension of self is greater than 8 or less than 2.</td>
    </tr>
  </tbody>
  </table>

## aclnnTriu

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 230px">
  <col style="width: 127px">
  <col style="width: 851px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnTriuGetWorkspaceSize</code>.</td>
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

## aclnnInplaceTriuGetWorkspaceSize

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 349px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>selfRef (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor</td>
      <td>-</td>
      <td>UINT64, INT64, UINT32, INT32, UINT16, INT16, UINT8, INT8, FLOAT16, FLOAT32, DOUBLE, BOOL, BFLOAT16, COMPLEX32, COMPLEX64</td>
      <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
      <td>2-8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonal (int64_t)</td>
      <td>Input</td>
      <td>Diagonal offset.</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - Ascend 950PR/Ascend 950DT: Only COMPLEX32 and COMPLEX64 are supported.
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data type cannot be BFLOAT16.

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 230px">
  <col style="width: 127px">
  <col style="width: 851px">
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
      <td><code>selfRef</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of the selfRef parameter is not supported.</td>
    </tr>
    <tr>
      <td>The data format of the selfRef parameter is private.</td>
    </tr>
    <tr>
      <td>The selfRef dimension is greater than 8 or less than 2.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceTriu

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 230px">
  <col style="width: 127px">
  <col style="width: 851px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnInplaceTriuGetWorkspaceSize</code>.</td>
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

- Deterministic computation:
  - `aclnnTriu&aclnnInplaceTriu` defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_triu.h"

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
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  int64_t diagonal = 1;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1.123, -2.001, 303.45, 40009, -50.1234, 60.666, -7.6543,
                                     8000, -9.009, 1024, -11.23345, 12, 1356, -14.99, -15.34023};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnTriu.
  ret = aclnnTriuGetWorkspaceSize(self, diagonal, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriuGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTriu.
  ret = aclnnTriu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriu failed. ERROR: %d\n", ret); return ret);

  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceTriu.
  ret = aclnnInplaceTriuGetWorkspaceSize(self, diagonal, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTriuGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnInplaceTriu.
  ret = aclnnInplaceTriu(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTriu failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr,
                    inplaceSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("inplaceResult[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
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
