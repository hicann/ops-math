# aclnnNpuFormatCast

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/npu_format_cast)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **Operator function**:
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - Performs ND ←→ [NZ](../../../docs/en/context/data_format.md) conversion. C0 is the size of the last dimension in the [NZ](../../../docs/en/context/data_format.md) data format. The calculation method is C0 = 32B / ge::GetSizeByDataType(static_cast `aclDataType` additionalDtype).
    - Performs NCDHW ←→ [NDC1HWC0](../../../docs/en/context/data_format.md) and NCDHW ←→ [FRACTAL_Z_3D](../../../docs/en/context/data_format.md) conversion. C0 is closely related to the micro-architecture, and the value is equal to the cube unit size, for example, 16. C1 is obtained by splitting the C dimension based on C0: C1 = C/C0. If the result is not exactly divided, the last piece of data needs to be padded to C0. Calculation method: C0 = 32B srcDataType (for example, FP16 is 2 bytes)

- **Calculation process**: Based on the input tensor srcTensor, data type `additionalDtype`, and data format dstFormat of the target tensor, `aclnnNpuFormatCastCalculateSizeAndFormat` is called to calculate the shape and actual data format of the converted target tensor dstTensor. The result is used to construct dstTensor. Then, `aclnnNpuFormatCast` is called to convert srcTensor into dstTensor in the actual data format.

## Prototype

First, `aclnnNpuFormatCastCalculateSizeAndFormat` is called to calculate the shape and actual data format of dstTensor. Then, the [two-phase API](../../../docs/en/context/two_phase_api.md) is called. For the two-phase API calls, the `aclnnNpuFormatCastGetWorkSpaceSize` API is called first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then the `aclnnNpuFormatCast` API is called to perform computation.

- `aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat)`

- `aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnNpuFormatCastCalculateSizeAndFormat

- **Parameters**

  - `srcTensor` (aclTensor*, computation input): input tensor, which is an aclTensor on the device. The input data can be contiguous or [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md).
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The [data format](../../../docs/en/context/data_format.md) can be ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D. The data type can be INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32. When the data format is ND, the supported shape dimensions are [2, 6].
  - `dstFormat` (int, computation input): data format of the output tensor.
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The [data format](../../../docs/en/context/data_format.md) can be ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D.

  - `additionalDtype` (int, computation input): basic data type used to infer the C0 size when the data format is converted to FRACTAL_NZ.
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: This parameter supports only the data type of srcTensor.

  - `dstShape` (int64_t**, output parameter): pointer to the shape array of the output dstTensor. The memory to which the pointer points is allocated by this API and released by the caller.

  - `dstShapeSize` (uint64_t*, output parameter): pointer to the size of the shape array of the output dstTensor.

  - `actualFormat` (int*, output parameter): pointer to the actual data format of the output dstTensor.
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The current output [data format](../../../docs/en/context/data_format.md) can be ACL_FORMAT_ND(2), ACL_FORMAT_FRACTAL_NZ(29), ACL_FORMAT_NCDHW(30), ACL_FORMAT_NDC1HWC0(32), or ACL_FRACTAL_Z_3D(33).

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  Input parameter verification. An error is reported in the following scenarios:
  - 161001 (ACLNN_ERR_PARAM_NULLPTR):
  1. The passed srcTensor is a null pointer.
  - 161002 (ACLNN_ERR_PARAM_INVALID):
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    1. The data format of srcTensor is not ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D, and the data type is not INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32.
    2. The data format of dstFormat is not ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D, and the data type is not INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32.
    3. The data type of additionalDtype is not the data type of srcTensor.
    4. The view shape dimension of srcTensor is not in the range of [2, 6] (ND→NZ).
  - 361001(ACLNN_ERR_RUNTIME_ERROR):
  1. The product model is not supported.
  2. The conversion format is not supported.
  ```

## aclnnNpuFormatCastGetWorkspaceSize

- **Parameters**
  - `srcTensor` (aclTensor*, computation input): input tensor, which is an aclTensor on the device. The input data must be contiguous tensors.
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The [data format](../../../docs/en/context/data_format.md) can be ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D. The data type can be INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32.
  - `dstTensor` (aclTensor*, computation input): converted target tensor, which is an aclTensor on the device. Only contiguous tensors are supported.
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The [data format](../../../docs/en/context/data_format.md) can be ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D. The data type can be INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32.
  - `workspaceSize` (uint64_t*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor**, output): operator executor that contains the operator computation process.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  - 161001 (ACLNN_ERR_PARAM_NULLPTR):
  1. The passed srcTensor and dstTensor are null pointers.
  - 161002 (ACLNN_ERR_PARAM_INVALID):
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    1. The data type of srcTensor is not INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32, and the data format is not ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D.
    2. The data type of dstTensor is not INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, or UINT32, and the data format is not ND, NZ, NCDHW, NDC1HWC0, or FRACTAL_Z_3D.
    3. non-contiguous tensor are passed to srcTensor and dstTensor.
  - 361001(ACLNN_ERR_RUNTIME_ERROR):
  1. The product model is not supported.
  ```

## aclnnNpuFormatCast

- **Parameters**

  - `workspace` (void*, input): address of the workspace to be allocated on the device.
  - `workspaceSize` (uint_64, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnNpuFormatCastGetWorkspaceSize`.
  - `executor` (aclOpExecutor*, input): operator executor that contains the operator computation process.
  - `stream` (aclrtStream, input): stream for executing the task.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnNpuFormatCast` defaults to a deterministic implementation.
The input and output support the following data type combinations:
Currently, the following special scenarios are not supported:
- If the data type of `srcTensor` is the same as that of `additionalDtype` and is FLOAT16 or BFLOAT16, and the dimensions are represented as [k, n], the scenario where k is 1 is not supported.
- After this API is called to convert the data format to Ascend affinity [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that can modify the tensor, such as contiguous, pad, and slice, is not supported.
- When any of the last two dimensions of the shape of `srcTensor` is 1, any operation that can modify the tensor, including transpose, is not allowed after the data format is converted to Ascend affinity [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ.

- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

  Parameters of the aclnnNpuFormatCastCalculateSizeAndFormat API:

  | srcTensor | dstFormat                 | additionalDtype              | actualFormat                    |
  | --------- | ------------------------- | ---------------------------- | ------------------------------- |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32                | ACL_FORMAT_FRACTAL_NZ(29)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FORMAT_ND(2) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FORMAT_ND(2) |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FORMAT_NCDHW(30) |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FORMAT_NDC1HWC0(32) |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FRACTAL_Z_3D(33) |

  aclnnNpuFormatCastGetWorkspaceSize API:

  | srcTensor | Data Type of dstTensor| Data Format of dstTensor              |
  | --------- | ----------------- | ------------------------------- |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32     | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32             | ACL_FORMAT_FRACTAL_NZ(29)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FORMAT_ND(2)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FORMAT_NCDHW(30)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FORMAT_NDC1HWC0(32)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FRACTAL_Z_3D(33)       |

  C0 calculation method: $C0=\frac{32B}{size\ of\ Basic type of srcTensor}$

    | Basic type of srcTensor| C0 |
    | --------------- | -- |
    | ACL_FLOAT(0), ACL_INT32(3), and ACL_UINT32(8)    | 8 |
    | ACL_FLOAT16(1) and ACL_BF16(27) | 16 |
    | ACL_INT8(2) and ACL_UINT8(4)   | 32 |

Currently, the following special scenarios are not supported:

- After this API is called to convert the data format to Ascend affinity [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that can modify the tensor, such as contiguous, pad, and slice, is not supported.
- After the data format is converted to Ascend affinity [data format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that can modify the tensor, including transpose, is not allowed.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
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

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
        // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs based on the API definition.
      int64_t k = 64;
      int64_t n = 128;
      int64_t srcDim0 = k;
      int64_t srcDim1 = n;
      int dstFormat = 29;
      aclDataType srcDtype = aclDataType::ACL_INT32;
      aclDataType additionalDtype = aclDataType::ACL_FLOAT16;

      std::vector<int64_t> srcShape = {srcDim0, srcDim1};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(k * n, 1);
      for (size_t i = 0; i < k; i++) {
          for (size_t j = 0; j < n; j++) {
              srcHostData[i * n + j] = (j + 1) % 128;
          }
      }

      std::vector<int32_t> dstTensorHostData(k * n, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // Create a src aclTensor.
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // Calculate the shape and format of the target tensor.
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host.
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. Release dstShape, aclTensor, and aclScalar.
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7. Release device resources.
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
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
      // Change the format of src.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
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

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
        // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs based on the API definition.

      int dstFormat = 32;
      // Change the target format here.
      aclDataType srcDtype = aclDataType::ACL_INT32;
      int additionalDtype = -1;

      // std::vector<int64_t> srcShape = {srcDim0 , srcDim1};
      int64_t N = 1;
      int64_t C = 17;
      int64_t D = 1;
      int64_t H = 2;
      int64_t W = 2;


      std::vector<int64_t> srcShape = {N, C, D, H, W};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(N * C * D * H * W, 1);

      int num = 0;
      for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
              for (int d = 0; d < D; ++d) {
                  for (int h = 0; h < H; ++h) {
                      for (int w = 0; w < W; ++w) {
                          // Arrange data in the row-major order and calculate the linear index.
                          int index = (((n * C + c) * D + d) * H + h) * W + w;
                          srcHostData[index] = num;
                          num++;
                      }
                  }
              }
          }
      }

      std::vector<int32_t> dstTensorHostData(N * C * D * H * W, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // Create a src aclTensor.
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;
      std::cout << "init actualFormat = " << actualFormat << std::endl;
      // Calculate the shape and format of the target tensor.
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, dstFormat, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      std::cout << "actualFormat = " << actualFormat << std::endl;
      std::cout << "&dstShape = " << &dstShape << std::endl;
      std::cout << "dstShape = [ ";
      for (int64_t i = 0; i < dstShapeSize; ++i) {
          std::cout << dstShape[i] << " ";
      }
      std::cout << "]" << std::endl;

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host.
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. Release dstShape, aclTensor, and aclScalar.
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7. Release device resources.
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
