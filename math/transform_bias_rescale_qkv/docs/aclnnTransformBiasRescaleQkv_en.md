# aclnnTransformBiasRescaleQkv

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/math/transform_bias_rescale_qkv)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- API function:
  The TransformBiasRescaleQkv operator is an interface used to process the query, key, and value vectors in the multi-head attention mechanism. It is used to adjust the bias and rescale factors of these vectors to optimize the attention computation process.

- Formulas: 
  The element-wise computation process is as follows:

  $$

  q_o=(q_i+q_{bias})/\sqrt{dim\_per\_head}\\

  $$

  $$
  
  k_o=k_i+k_{bias}\\

  $$

  $$
  
    v_o=v_i+v_{bias}

  $$

  In the formula:
  
  - `dim_per_head` indicates the dimension of each attention head.
  - q<sub>o</sub>, k<sub>o</sub>, and v<sub>o</sub> are the output elements of the query, key, and value vectors, respectively.
  - q<sub>i</sub>, k<sub>i</sub>, and v<sub>i</sub> are the input elements of the query, key, and value vectors, respectively.
  - q<sub>bias</sub>, k<sub>bias</sub>, and v<sub>bias</sub> are the input element offsets of the query, key, and value vectors, respectively.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTransformBiasRescaleQkvGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnTransformBiasRescaleQkv` is called to perform computation.

```Cpp
aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qkvBias,
    int64_t          numHeads,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnTransformBiasRescaleQkv(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnTransformBiasRescaleQkvGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1370px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>qkv (aclTensor*) </td>
      <td>Input</td>
      <td>Input tensor, which is q<sub>i</sub>, k<sub>i</sub> and v<sub>i</sub> in the formula.</td>
      <td>The shape is a three-dimensional tensor of {B, T, 3 * num_heads * dim_per_head}. B indicates the batch size, T indicates the sequence length, num_heads indicates the number of attention heads, and dim_per_head indicates the dimension of each attention head.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvBias (aclTensor*) </td>
      <td>Input</td>
      <td>Input tensor, which is q<sub>bias</sub>, k<sub>bias</sub> and v<sub>bias</sub> in the formula.</td>
      <td><ul><li>The shape is a one-dimensional tensor of {3 * num_heads * dim_per_head}. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeads (int64_t) </td>
      <td>Input</td>
      <td>Number of input heads.</td>
      <td>The value must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>qOut (aclTensor*) </td>
      <td>Output</td>
      <td>Output tensor, which is q<sub>o</sub> in the formula.</td>
      <td>The shape is a four-dimensional tensor of {B, num_heads, T, dim_per_head}.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOut (aclTensor*) </td>
      <td>Output</td>
      <td>Output tensor, which is k<sub>o</sub> in the formula.</td>
      <td>The shape is a four-dimensional tensor of {B, num_heads, T, dim_per_head}.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOut (aclTensor*) </td>
      <td>Output</td>
      <td>Output tensor, which is v<sub>o</sub> in the formula.</td>
      <td>The shape is a four-dimensional tensor of {B, num_heads, T, dim_per_head}.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4</td>
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

  <table style="undefined;table-layout: fixed; width: 1145px"><colgroup>
  <col style="width: 295px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed input and output are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td><ul><li>The data types and formats of <code>qkv</code> and <code>qkvBias</code> are not supported. </li><li>The data types of <code>qkv</code> and <code>qkvBias</code> are different. </li><li>The shapes of <code>qkv</code> and <code>qkvBias</code> do not meet the parameter requirements.</li></ul></td>
    </tr>
  </tbody></table>

## aclnnTransformBiasRescaleQkv

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnTransformBiasRescaleQkvGetWorkspaceSize</code>.</td>
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
  - `aclnnTransformBiasRescaleQkv` defaults to a deterministic implementation.

- The data types of the inputs `qkv` and `qkvBias`, and outputs `qOut`, `kOut`, and `vOut` must be the same.
- If the input is **NaN**, the output is also **NaN**. If the input is **Inf**, the output is also **Inf**.
- If the input is `-Inf`, the output is `-Inf`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_transform_bias_rescale_qkv.h"

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
    LOG_PRINT("outPut result[%ld] is: %f\n", i, resultData[i]);
  }
}

void PrintInResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("input[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize.
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
// qkv
int64_t B = 3;
int64_t T = 4;
int64_t n = 3;
int64_t d = 16;
std::vector<int64_t> qkvShape = {B, T, 3 * n * d};
int qkvCount = B * T * 3 * n * d;
std::vector<float> qkvHostData(qkvCount, 1);

for (int i = 0; i < qkvCount; ++i) {
    qkvHostData[i] = i * 1.0;
}

void* qkvDeviceAddr = nullptr;
aclTensor* qkv = nullptr;
// Create an input aclTensor.
ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT, &qkv);
CHECK_RET(ret == ACL_SUCCESS, return ret);

// qkvBias
std::vector<int64_t> qkvBiasShape = {3 * n * d};
std::vector<float> qkvBiasHostData(3 * n * d, 0.5);

void* qkvBiasDeviceAddr = nullptr;
aclTensor* qkvBias = nullptr;
// Create an input aclTensor.
ret = CreateAclTensor(qkvBiasHostData, qkvBiasShape, &qkvBiasDeviceAddr, aclDataType::ACL_FLOAT, &qkvBias);
CHECK_RET(ret == ACL_SUCCESS, return ret);

std::vector<int64_t> outShape = {B, n, T, d};
std::vector<float> outHostData(qkvCount / 3, 1);
aclTensor* outQ = nullptr;
aclTensor* outK = nullptr;
aclTensor* outV = nullptr;
void* outQDeviceAddr = nullptr;
void* outKDeviceAddr = nullptr;
void* outVDeviceAddr = nullptr;

// Create an out aclTensor.
ret = CreateAclTensor(outHostData, outShape, &outQDeviceAddr, aclDataType::ACL_FLOAT, &outQ);
ret = CreateAclTensor(outHostData, outShape, &outKDeviceAddr, aclDataType::ACL_FLOAT, &outK);
ret = CreateAclTensor(outHostData, outShape, &outVDeviceAddr, aclDataType::ACL_FLOAT, &outV);

CHECK_RET(ret == ACL_SUCCESS, return ret);

// 3. Call the CANN operator library API, which needs to be replaced with the actual API.
uint64_t workspaceSize = 16 * 1024 * 1024;
aclOpExecutor* executor;

// LOG_PRINT("qkv input=====");
// PrintInResult(qkvShape, &qkvDeviceAddr);

// LOG_PRINT("qkvBias input=====");
// PrintInResult(qkvBiasShape, &qkvBiasDeviceAddr);

// Call the first-phase API of aclnnTransformBiasRescaleQkv.
ret = aclnnTransformBiasRescaleQkvGetWorkspaceSize(
qkv,
qkvBias,
n,
outQ,
outK,
outV,
&workspaceSize,
&executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkvGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

// Allocate device memory based on workspaceSize computed by the first-phase API.
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
}

// Call the second-phase API of aclnnTransformBiasRescaleQkv.
ret = aclnnTransformBiasRescaleQkv(
workspaceAddr,
workspaceSize,
executor,
stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkv failed. ERROR: %d\n", ret); return ret);

// 4. (Boilerplate) Wait until the task execution is complete.
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
LOG_PRINT("q output=====");
PrintOutResult(outShape, &outQDeviceAddr);

LOG_PRINT("k output=====");
PrintOutResult(outShape, &outKDeviceAddr);


LOG_PRINT("v output=====");
PrintOutResult(outShape, &outVDeviceAddr);

// 6. Release the aclTensor. Modify the code based on the API definition.
aclDestroyTensor(qkv);
aclDestroyTensor(qkvBias);
aclDestroyTensor(outQ);
aclDestroyTensor(outK);
aclDestroyTensor(outV);

// 7. Release device resources. Modify the code based on the API definition.
aclrtFree(qkvDeviceAddr);
aclrtFree(qkvBiasDeviceAddr);

aclrtFree(outQDeviceAddr);
aclrtFree(outKDeviceAddr);
aclrtFree(outVDeviceAddr);

if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();

return 0;
}
```
