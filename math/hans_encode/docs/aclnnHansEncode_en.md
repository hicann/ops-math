# aclnnHansEncode

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

(Optional) Performs PDF statistics on the byte where the exponent bit of the input tensor is located, and performs lossless compression based on the PDF distribution. The compressed result can be stored in the HBM of the device or offloaded to the host.

## Prototype

Each operator is divided into two segments.[two-phase API](../../../docs/en/context/two_phase_api.md). You must call the aclnnHansEncodeGetWorkspaceSize interface to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the aclnnHansEncode interface to perform computation.

- `aclnnStatus aclnnHansEncodeGetWorkspaceSize(const aclTensor *inputTensor, aclTensor *pdfRef, bool statistic, bool reshuff, const aclTensor *mantissaOut, const aclTensor *fixedOut, const aclTensor *varOut, uint64_t *workspaceSize, aclOpExecutor **executor);`
- `aclnnStatus aclnnHansEncode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnHansEncodeGetWorkspaceSize

- **Parameter description**:

  - inputTensor (aclTensor*, computation input): input tensor to be compressed, aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The size of data elements must be a multiple of 64 and greater than or equal to 32768.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
  - pdfRef (aclTensor*, computation input/output): probability density distribution of the bytes where the exponent bits of the inputTensor are located, aclTensor on the device. The data type can be INT32. The shape must be (1, 256).[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
  - statistic (bool, input): whether to perform PDF statistics.
  - reshuff (bool, input): whether to perform memory reshuffling on the encoding result of each core.
  - mantissaOut (aclTensor*, output): mantissa output, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32, which must be the same as that of inputTensor.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
  - fixedOut (aclTensor*, output): first segment of the compressed output, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32, which must be the same as that of inputTensor.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
  - varOut (aclTensor*, output): output after the fixedOut is compressed, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32, which must be the same as that of inputTensor.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 124px">
  <col style="width: 740px">
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
      <td>inputTensor, pdf, mantissaOut, fixedOut, and varOut are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The length of pdf or mantissa is incorrect.</td>
    </tr>
    <tr>
      <td>The accumulated encoding space is too small, and compression overflow may occur. The following formula must be met: (size(fixedOut) + size(mantissaOut)) &gt;= (len(inputTensor) + len(inputTensor) / 64 + 8448 * processCoreDim).</td>
    </tr>
    <tr>
      <td>The size of fixedOut is less than 512 bytes, and the compressed metadata cannot be stored.</td>
    </tr>
    <tr>
      <td>The number of input elements is not a multiple of 64 or is less than 32768.</td>
    </tr>
    <tr>
      <td>The data types of inputTensor, pdf, and outputMantissaTensorOut are not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnHansEncode

- **Parameter description**:

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnEyeGetWorkspaceSize.</td>
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

None

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hans_encode.h"
#include "aclnn_hans_decode.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<float> inputHost(65536, 0);
  std::vector<float> mantissaHost(49152, 0);
  std::vector<float> fixedHost(16384, 0);
  std::vector<float> varHost(16384, 0);
  std::vector<int32_t> pdfHost(256, 0);
  std::vector<float> recoverHost(65536, 0);
  bool statistic = true;
  bool reshuff = false;
  int64_t outHostAddr = -1;
  int64_t outHostLength = 0;

  void* inputAddr = nullptr;
  void* outMantissaAddr = nullptr;
  void* outFixedAddr = nullptr;
  void* outVarAddr = nullptr;
  void* pdfAddr = nullptr;
  void* recoverAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* outMantissa = nullptr;
  aclTensor* outFixed = nullptr;
  aclTensor* pdf = nullptr;
  aclTensor* outVar = nullptr;
  aclTensor* recover = nullptr;
  // Create an out aclTensor.
  ret = CreateAclTensor(inputHost, {1, 65536}, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mantissaHost, {1, 49152}, &outMantissaAddr, aclDataType::ACL_FLOAT, &outMantissa);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(fixedHost, {1, 16384}, &outFixedAddr, aclDataType::ACL_FLOAT, &outFixed);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(varHost, {1, 16384}, &outVarAddr, aclDataType::ACL_FLOAT, &outVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pdfHost, {1, 256}, &pdfAddr, aclDataType::ACL_INT32, &pdf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(recoverHost, {1, 65536}, &recoverAddr, aclDataType::ACL_FLOAT, &recover);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first API of aclnnHansEncode.
  ret = aclnnHansEncodeGetWorkspaceSize(input, pdf, statistic, reshuff, outMantissa, outFixed, outVar, &workspaceSize,
                                        &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansEncodeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second API of aclnnHansEncode.
  ret = aclnnHansEncode(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansEncode failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = 16384 * sizeof(float);
  std::vector<float> resultData(16384, 0);
  ret = aclrtMemcpy(resultData.data(), size, outFixedAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 128; i++) {
    int32_t intVal = *reinterpret_cast<int32_t*>(&resultData[i]);
    LOG_PRINT("result header[%ld] is: %d\n", i, intVal);
  }

  uint64_t workspaceSizeDecode = 0;
  aclOpExecutor* executorDecode;
  // Call the first segment of the aclnnHansDecode API.
  ret = aclnnHansDecodeGetWorkspaceSize(outMantissa, outFixed, outVar, pdf, reshuff, recover, &workspaceSizeDecode,
                                        &executorDecode);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansDecodeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddrDecode = nullptr;
  if (workspaceSizeDecode > 0) {
    ret = aclrtMalloc(&workspaceAddrDecode, workspaceSizeDecode, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second segment of the aclnnHansEncode API.
  ret = aclnnHansDecode(workspaceAddrDecode, workspaceSizeDecode, executorDecode, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansDecode failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  std::vector<float> recoverData(65536, 0);
  ret = aclrtMemcpy(recoverData.data(), 65536 * sizeof(recoverData[0]), recoverAddr, 65536 * sizeof(recoverData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 256; i++) {
    LOG_PRINT("reco[%ld] is: %f org is: %f\n", i, recoverData[i], inputHost[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(outMantissa);
  aclDestroyTensor(outFixed);
  aclDestroyTensor(outVar);
  aclDestroyTensor(pdf);
  aclDestroyTensor(recover);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(inputAddr);
  aclrtFree(outMantissaAddr);
  aclrtFree(outFixedAddr);
  aclrtFree(outVarAddr);
  aclrtFree(pdfAddr);
  aclrtFree(recoverAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (workspaceSizeDecode > 0) {
    aclrtFree(workspaceAddrDecode);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
