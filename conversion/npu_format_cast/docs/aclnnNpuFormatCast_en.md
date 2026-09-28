# aclnnNpuFormatCast

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/npu_format_cast)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- **API function**:

  - Ascend 950PR/Ascend 950DT:

    Converts the ND [Data Format](../../../docs/en/context/data_format.md) to the FRACTAL_NZ [Data Format](../../../docs/en/context/data_format.md) with the specified C0 size. C0 is the size of the last dimension of the FRACTAL_NZ [Data Format](../../../docs/en/context/data_format.md), and is determined by `additionalDtype`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - Performs ND ←→ [NZ](../../../docs/en/context/data_format.md) conversion. C0 is the size of the last dimension in the [NZ](../../../docs/en/context/data_format.md) data format. The calculation method is as follows: C0 = 32B / ge::GetSizeByDataType(static_cast additionalDtype).
    - Performs NCDHW ←→ [NDC1HWC0](../../../docs/en/context/data_format.md) and NCDHW ←→ [FRACTAL_Z_3D](../../../docs/en/context/data_format.md) conversion. C0 is closely related to the micro-architecture, and the value is equal to the cube unit size, for example, 16. C1 is obtained by splitting the C dimension based on C0: C1 = C/C0. If the result is not exactly divided, the last piece of data needs to be padded to C0. The calculation method is as follows: C0 = 32B / srcDataType (For example, FP16 is 2 bytes.)
- **Computing process**:

  `aclnnNpuFormatCastCalculateSizeAndFormat` calculates the shape and actual data format of the destination tensor dstTensor based on the input tensor srcTensor, data type `additionalDtype`, and data format dstFormat of the destination tensor. Then, `aclnnNpuFormatCast` is called to convert the srcTensor to the destination tensor dstTensor in the actual data format.

## Prototype

First, `aclnnNpuFormatCastCalculateSizeAndFormat` is called to calculate the shape and actual data format of dstTensor. Then, the [two-phase API](../../../docs/en/context/two_phase_api.md) is called. For the two-phase API calls, the `aclnnNpuFormatCastGetWorkSpaceSize` API is called first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then the `aclnnNpuFormatCast` API is called to perform computation.

```c++
aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(
    const aclTensor* srcTensor,
    const int        dstFormat,
    int              additionalDtype,
    int64_t**        dstShape,
    uint64_t*        dstShapeSize,
    int*             actualFormat)
```

```c++
aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(
    const aclTensor* srcTensor,
    aclTensor*       dstTensor,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```c++
aclnnStatus aclnnNpuFormatCast(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnNpuFormatCastCalculateSizeAndFormat

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1665px;">
  <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 200px">
      <col style="width: 290px">
      <col style="width: 300px">
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
      </tr>
  </thead>
  <tbody>
        <tr>
            <td>srcTensor (aclTensor*) </td>
            <td>Input</td>
            <td>Source tensor to be converted.</td>
            <td>-</td>
            <td>INT8, UINT8, INT32, UINT32, FLOAT, FLOAT16, BFLOAT16<sup>2</sup>, FLOAT8_E4M3FN, FLOAT8_E4M3FN<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
            <td>ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D, NCL<sup>2</sup></td>
            <td>2-6</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstFormat (int) </td>
            <td>Input</td>
            <td>Format of the output tensor.</td>
            <td>-</td>
            <td>None</td>
            <td>ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>additionalDtype (int) </td>
            <td>Optional input</td>
            <td>Basic data type used to infer the C0 size when the FRACTAL_NZ format is converted.</td>
            <td>-</td>
            <td>ACL_FLOAT16(1), ACL_BF16(27), INT8(2), ACL_FLOAT8_E4M3FN(36)</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstShape (int64_t**) </td>
            <td>Output</td>
            <td>Pointer to the shape array of the dstTensor output. The memory to which the pointer points is allocated by this API and released by the caller.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>4-8</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstShapeSize (uint64_t*) </td>
            <td>Output</td>
            <td>Pointer to the size of the shape array of the dstTensor output.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualFormat (int*) </td>
            <td>Output</td>
            <td>Pointer to the actual data format of the dstTensor output.</td>
            <td>-</td>
            <td>None</td>
            <td>ACL_FORMAT_ND(2), ACL_FORMAT_FRACTAL_NZ(29), ACL_FORMAT_NCDHW(30), ACL_FORMAT_NDC1HWC0(32), ACL_FRACTAL_Z_3D(33), ACL_FORMAT_FRACTAL_NZ_C0_16(50)<sup>2</sup>, ACL_FORMAT_FRACTAL_NZ_C0_32(51)<sup>2</sup></td>
            <td>None</td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

  - Ascend 950PR/Ascend 950DT:
    - The superscript "1" in the data type column of the preceding table indicates that the data type or format is not supported by the corresponding series.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - The superscript "2" in the data type column of the preceding table indicates that the data type or format is not supported by the corresponding series.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table>
    <thead>
      <tr>
        <th style="width: 291px">Return Value</th>
        <th style="width: 135px">Error Code</th>
        <th style="width: 724px">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td>The input srcTensor is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="5"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="5"> 161002 </td>
        <td>The data format of srcTensor is not ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D or NCL, and the data type is not INT8, UINT8, INT32, UINT32, FLOAT, FLOAT16, BFLOAT16, FLOAT8_E4M3FN or FLOAT4_E2M1.</td>
      </tr>
      <tr>
        <td>The data format of dstFormat is not ND, NZ, NCDHW, NDC1HWC0 or FRACTAL_Z_3D</td>.
      </tr>
      <tr>
        <td>The data type of additionalDtype is not ACL_FLOAT16(1), ACL_BF16(27), INT8(2), or ACL_FLOAT8_E4M3FN(36).</td>
      </tr>
      <tr>
        <td>The view shape dimension of srcTensor is not within the range of [2, 6].</td>
      </tr>
      <tr>
        <td>The input Tensor</td> of srcTensor is empty.
      </tr>
      <tr>
        <td rowspan="2"> ACLNN_ERR_RUNTIME_ERROR </td>
        <td rowspan="2"> 361001 </td>
        <td>The product model is not supported.</td>
      </tr>
      <tr>
        <td>The conversion format is not supported.</td>
      </tr>
    </tbody>
  </table>

## aclnnNpuFormatCastGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1665px;">
  <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 200px">
      <col style="width: 290px">
      <col style="width: 300px">
      <col style="width: 130px">
      <col style="width: 145px">
  </colgroup>
  <thead>
      <tr>
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
          <td>Usage</td>
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimension (Shape)</th>
          <td>Non-contiguous Tensor</td>
      </tr>
  </thead>
  <tbody>
        <tr>
            <td>srcTensor (aclTensor*) </td>
            <td>Input</td>
            <td>Input tensor. Only contiguous tensors are supported.</td>
            <td>-</td>
            <td>INT8, UINT8, INT32, UINT32, FLOAT, FLOAT16, BFLOAT16<sup>2</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
            <td>ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D, NCL<sup>2</sup></td>
            <td>2-6</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstTensor (aclTensor*) </td>
            <td>Input</td>
            <td>Target tensor after conversion. Only contiguous tensors are supported.</td>
            <td>-</td>
            <td>INT8, UINT8, INT32, UINT32, FLOAT, FLOAT16, BFLOAT16<sup>2</sup>, FLOAT8_E4M3FN, FLOAT8_E4M3FN<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
            <td>ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D, ACL_FORMAT_FRACTAL_NZ_C0_16(50)<sup>2</sup>, ACL_FORMAT_FRACTAL_NZ_C0_32(51)<sup>2</sup></td>
            <td>4-8</td>
            <td>-</td>
        </tr>
        <tr>
            <td>workspaceSize (uint64_t*) </td>
            <td>Input</td>
            <td>Size of the workspace to be allocated on the device.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor (aclOpExecutor**) </td>
            <td>Input</td>
            <td>Operator executor that contains the operator computation process.</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

  - Ascend 950PR/Ascend 950DT:

    - The superscript "1" in the data type column of the preceding table indicates that the data type or format is not supported by the corresponding series.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    - The superscript "2" in the data type column of the preceding table indicates that the data type or format is not supported by the corresponding series.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

    <table>
    <thead>
      <tr>
        <th style="width: 291px">Return Value</th>
        <th style="width: 135px">Error Code</th>
        <th style="width: 724px">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td>The input srcTensor and dstTensor are null pointers.</td>
      </tr>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="4"> 161002 </td>
        <td>The data type of srcTensor is not INT8, UINT8, INT32, UINT32, FLOAT, FLOAT16, BFLOAT16, FLOAT8_E4M3FN or FLOAT4_E2M1, and the data format is not ND, NZ, NCDHW, NDC1HWC0, FRACTAL_Z_3D or NCL.</td>
      </tr>
      <tr>
        <td>The data type of dstTensor is not INT8, UINT8, INT32, UINT32, FLOAT, FLOAT16, BFLOAT16, FLOAT8_E4M3FN or FLOAT4_E2M1, and the data format is not ND, NZ, NCDHW, NDC1HWC0 or FRACTAL_Z_3D.</td>
      </tr>
      <tr>
        <td>The input tensors of srcTensor and dstTensor are not contiguous.</td>
      </tr>
      <tr>
        <td>The view shape dimension of srcTensor is not in the range of [2, 6], and the storage shape dimension of dstTensor is not in the range of [4, 8].<sup>2</sup></td>
      </tr>
      <tr>
        <td rowspan="1"> ACLNN_ERR_RUNTIME_ERROR </td>
        <td rowspan="1"> 361001 </td>
        <td>The product model is not supported.</td>
      </tr>
    </tbody>
  </table>

  - Ascend 950PR/Ascend 950DT:

    - The superscript "1" in the data type column in the preceding table indicates the interception type that is not supported by the series.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    - The superscript "2" in the data type column in the preceding table indicates the interception type that is not supported by the series.

## aclnnNpuFormatCast

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnNpuFormatCastGetWorkspaceSize API.</td>
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

- Deterministic computing: The aclnnNpuFormatCast is implemented in deterministic mode by default.

- The input and output support the following data type combinations:

  <details>

  <summary>Ascend 950PR/Ascend 950DT</summary>

    - Parameters of the aclnnNpuFormatCastCalculateSizeAndFormat API:

      | srcTensor | dstFormat                 | additionalDtype              | actualFormat                    |
      | --------- | ------------------------- | ---------------------------- | ------------------------------- |
      | INT8      | ACL_FORMAT_FRACTAL_NZ(29) | ACL_INT8(2)                  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT32     | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1) and ACL_BF16(27)| ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1) and ACL_BF16(27)| ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36) | ACL_FORMAT_FRACTAL_NZ_C0_32(51) |
      | FLOAT16      | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1) | ACL_FORMAT_FRACTAL_NZ(29) |
      | BFLOAT16     | ACL_FORMAT_FRACTAL_NZ(29) | ACL_BF16(27)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT8_E4M3FN     | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36)   | ACL_FORMAT_FRACTAL_NZ(29) |

    - aclnnNpuFormatCastGetWorkspaceSize API:

      | srcTensor | Data Type of dstTensor| dstTensor [Data Format](../../../docs/en/context/data_format.md)              |
      | --------- | ----------------- | ------------------------------- |
      | INT8      | INT8              | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_16(50)/ACL_FORMAT_FRACTAL_NZ_C0_32(51) |
      | FLOAT16   | FLOAT16           | ACL_FORMAT_FRACTAL_NZ(29)       |
      | BFLOAT16  | BFLOAT16          | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT8_E4M3FN  | FLOAT8_E4M3FN          | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT4_E2M1  | FLOAT4_E2M1          | ACL_FORMAT_FRACTAL_NZ_C0_32(51)       |

    - C0 calculation method: $C0=\frac{32B}{size\ of\ additionalDtype}$

      | additionalDtype | C0 |
      | --------------- | -- |
      | ACL_INT8(2)     | 32 |
      | ACL_FLOAT16(1)  | 16 |
      | ACL_BF16(27)    | 16 |
      | ACL_FLOAT8_E4M3FN(36)    | 32 |

    - Currently, the following special scenarios are not supported:
      - If the data type of `srcTensor` is the same as that of `additionalDtype` and is FLOAT16 or BFLOAT16, and the dimensions are represented as [k, n], the scenario where k is 1 is not supported.
      - After this API is called to convert the data format to Ascend affinity [Data Format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that can modify the tensor, such as contiguous, pad, and slice, is not supported.
      - When any of the last two dimensions of the shape of `srcTensor` is 1, any operation that can modify the tensor, including transpose, is not allowed after the data format is converted to Ascend affinity [Data Format](../../../docs/en/context/data_format.md) FRACTAL_NZ.

  </details>

  <details>

  <summary><term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term></summary>

    - Parameters of the aclnnNpuFormatCastCalculateSizeAndFormat API:

      | srcTensor | dstFormat                 | additionalDtype              | actualFormat                    |
      | --------- | ------------------------- | ---------------------------- | ------------------------------- |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32                | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FORMAT_ND(2) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FORMAT_ND(2) |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FORMAT_NCDHW(30) |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FORMAT_NDC1HWC0(32) |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32   | ACL_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32  | ACL_FRACTAL_Z_3D(33) |

    - aclnnNpuFormatCastGetWorkspaceSize API:

      | srcTensor | Data Type of dstTensor| Data Format of dstTensor              |
      | --------- | ----------------- | ------------------------------- |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32     | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32             | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FORMAT_ND(2)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FORMAT_NCDHW(30)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FORMAT_NDC1HWC0(32)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32 | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, and UINT32         | ACL_FRACTAL_Z_3D(33)       |

    - C0 calculation method: $C0=\frac{32B}{size\ of\ Basic type of srcTensor}$

      | Basic type of srcTensor| C0 |
      | --------------- | -- |
      | ACL_FLOAT(0), ACL_INT32(3), and ACL_UINT32(8)    | 8 |
      | ACL_FLOAT16(1) and ACL_BF16(27) | 16 |
      | ACL_INT8(2) and ACL_UINT8(4)   | 32 |

    - Currently, the following special scenarios are not supported:
      - After this API is called to convert the data format to Ascend affinity [Data Format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that can modify the tensor, such as contiguous, pad, and slice, is not supported.
      - After the data format is converted to Ascend affinity [Data Format](../../../docs/en/context/data_format.md) FRACTAL_NZ, any operation that can modify the tensor, including transpose, is not allowed.

  </details>

## Example

- Ascend 950PR/Ascend 950DT:
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
