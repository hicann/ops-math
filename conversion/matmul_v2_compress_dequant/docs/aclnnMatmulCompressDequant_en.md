# aclnnMatmulCompressDequant

[📄 View source code](https://gitcode.com/cann/ops-math/tree/master/conversion/matmul_v2_compress_dequant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: During l@r matrix multiplication, you can use the msModelSlim tool to losslessly compress the r matrix to reduce the memory usage of the r matrix, and then use this API to complete lossless decompression, matrix multiplication, and dequantization.
- Formula:

$$
x2\_unzip = unzip(x2, compressIndex, compressInfo)\\
result=(x1 @ x2\_unzip + bias)*deqScale
$$

`x2` indicates the one-dimensional data of the r matrix after being compressed by the msModelSlim tool. `compressIndex` and `compressInfo` indicate the information related to the compression algorithm. `$x2\_unzip$` is the data after lossless decompression in this API (consistent with the original r matrix data). For details about the compression and calling example of this API, see [Example](#Example).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMatmulCompressDequantGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMatmulCompressDequant` is called to perform computation.

```cpp
aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(
  const aclTensor*   x1, 
  const aclTensor*   x2, 
  const aclTensor*   compressIndex, 
  const aclTensor*   bias, 
  const aclTensor*   deqScale, 
  const aclTensor*   offsetW, 
  int                offsetX, 
  const aclIntArray* compressInfo, 
  aclTensor*         out, 
  uint64_t*          workspaceSize, 
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnMatmulCompressDequant(
  void*           workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor*  executor, 
  aclrtStream     stream)
```

## aclnnMatmulCompressDequantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Left input of matrix multiplication.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Right input of matrix multiplication, which is compressed by the weight_compression module in the msModelSlim tool.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressIndex</td>
      <td>Input</td>
      <td>Compressed index table of the right input of matrix multiplication.</td>
      <td>Obtained from the msModelSlim tool in the example.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>2-dimensional aclTensor in ND format on the device.</td>
      <td>A null pointer can be passed in.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>Two dimensions. The shape can only be (1, n) or (n), where n is the n in the output shape (m, n).</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>Input</td>
      <td>Dequantization parameter.</td>
      <td>The value in the tensor is the UINT64 data of float converted in the following example.</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>Two dimensions. The shape can be (1, n) or (1, 1), where n is the n in the output shape (m, n).</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetW</td>
      <td>Input</td>
      <td>Right input offset of matrix multiplication.</td>
      <td>Currently, only null pointers can be passed in.</td>
      <td>INT8</td>
      <td>-</td>
      <td>Same as $x2\_unzip$.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetX</td>
      <td>Input</td>
      <td>Left input offset of matrix multiplication.</td>
      <td>Currently, only 0 is supported.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressInfo</td>
      <td>Input</td>
      <td>The data type is INT64. It includes the compression tile information tilingN and tilingK (which are obtained after compression by the weight_compression module of the msModelSlim tool and indicate the size of a basic compression tile in the n and k directions of the shape (n, k) before compression), the original shape (two-dimensional and represented by (n, k)) of the x2 matrix before compression, and the identifier of the traversal direction of the compression tile.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Input</td>
      <td>2-dimensional aclTensor on the device</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 303px">
  <col style="width: 118px">
  <col style="width: 726px">
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
      <td>The passed x1, x2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or format of x1 or x2 is not supported.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed for x1 and x2.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to that of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnMatmulCompressDequant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling aclnnMatmulCompressDequantGetWorkspaceSize.</td>
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

- Deterministic computation
  - `aclnnMatmulCompressDequant` defaults to deterministic implementation.

## Example

1. Prepare the data before compression.
    Assume that the input data is generated using the `gen_data.py` script. The following is an example for reference only:

    ```python
    import numpy as np
    import os
    import sys
    from numpy import random

    def write2file(data, path):
      with open(path, 'wb') as f:
          data.tofile(f)

    if not os.path.exists("./data"):
        os.mkdir("./data")

    if len(sys.argv) != 4:
      print("Usage: python gen_data.py m k n")
      sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])

    if m <= 0 or k <= 0 or n <= 0:
      print("Error: m, k and n must be positive integers.")
      sys.exit(1)

    # Randomly generate matrix mat1 with shape (m, k).
    mat1 = random.randn(m, k).astype(np.int8)
    write2file(mat1, "./data/mat1.bin")

    # Randomly generate matrix mat2 with shape (n, k).
    mat2 = random.randint(0, 100, size=(n, k)).astype(np.int8)
    np.save("./data/weight.npy", {'weight': mat2})
    os.chmod("./data/weight.npy", 0o0640)

    # Generate output.
    output = np.random.randn(m, n).astype(np.float16)
    write2file(output, "./data/output.bin")

    # Generate bias.
    bias = random.randn(n).astype(np.float32)
    write2file(bias, "./data/bias.bin")

    # Generate deq_scale.
    deq_scale = random.randn(n).astype(np.float32)
    write2file(deq_scale, "./data/deqScale_ori.bin")
    deq_scale_int64 = np.fromfile("./data/deqScale_ori.bin", dtype=np.int32).astype(np.int64)
    deq_scale_int64.tofile("./data/deqScale.bin")
    ```

    Execute gen_data.py. Assume that the input shapes of mat1 and mat2 are m=512, k=1024, and n=1024.

    ```shell
    python3 gen_data.py 512 1024 1024
    ```

2. Preprocess data.

    **Use the msModelSlim tool to compress the original weight to generate the compressed x2, compressIndex, and compressInfo.**

    When using the following APIs, you need to compile the msModelSlim tool in the CANN package. For details, see `README.md` in the `msmodelslim/pytorch/weight_compression` directory in the [Gitee msit repository](https://gitee.com/ascend/msit/tree/master/msmodelslim).

    ```python
    from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

    compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1)
    compressor = Compressor(compress_config, weight_path=weight_path)

    compress_weight, compress_index, compress_info = compressor.run()
    # Compressed weight, corresponding to x2 of aclnnMatmulCompressDequantGetWorkspaceSize
    compressor.export(compress_weight, './data/weight')
    # Index of the compressed weight, corresponding to compressIndex of aclnnMatmulCompressDequantGetWorkspaceSize
    compressor.export(compress_index, './data/index')
    # Compressed data information, corresponding to compressInfo of aclnnMatmulCompressDequantGetWorkspaceSize
    compressor.export(compress_info, './data/compress_info')

    ```
    
    **Convert the dequantization parameter deqscale of the original float type to obtain the uint64 data required by the aclnn API.**

    The original deqScale is of the float type. It is read as int32 and converted to int64.

    ```python
    import numpy as np
    data = np.fromfile('./deqScale_original.bin', dtype=np.int32).astype(np.int64)
    data.tofile('./deqScale.bin')
    ```

3. Call the aclnn API for computation.

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <acl/acl.h>
#include <aclnnop/aclnn_matmul_compress_dequant.h>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <cstdlib>
#include <string>

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

int ReadBinFileNNop(std::string filePath, void* buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    CHECK_RET(fileStatus == ACL_SUCCESS, LOG_PRINT("Failed to get file %s\n", filePath); return -1);

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    CHECK_RET(file.is_open(), LOG_PRINT("Open file failed.\n"); return -1);

    file.seekg(0, file.end);
    uint64_t binFileBufferLen = file.tellg();
    CHECK_RET(binFileBufferLen > 0,
        std::cout<<"File size is 0.\n";
        file.close();
        return -1);

    file.seekg(0, file.beg);
    file.read(static_cast<char *>(buffer), binFileBufferLen);
    file.close();
    return ACL_SUCCESS;
}

int CreateAclTensor(std::string filePath, const std::vector<int64_t>& shape, int typeSize,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * typeSize;
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMallocHost to allocate memory on the host.
  void* binBufferHost = nullptr;
  ret = aclrtMallocHost(&binBufferHost, size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);

  // Read the file.
  ret = ReadBinFileNNop(filePath, binBufferHost, size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
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

int main(int argc, char* argv[]) {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  if (argc != 6) {
    std::cerr << "Error: Invalid number of arguments. Usage: <program> m k n wCompressedSize indexSize" << std::endl;
    return -1;
  }

  // 2. Construct the inputs and outputs based on the API definition.
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  // wShape is the size of the compressed data of the right matrix.
  int wCompressedSize = atoi(argv[4]);
  // indexShape is the size of the compressed index data.
  int indexSize = atoi(argv[5]);

  if (m <= 0 || k <= 0 || n <= 0 || wCompressedSize <= 0 || indexSize <= 0) {
    std::cerr << "Error: m, k, n, wCompressedSize and indexSize must be positive integers." << std::endl;
    return -1;
  }

  std::vector<int64_t> mat1Shape = {m, k};
  std::vector<int64_t> mat2CompressedShape = {wCompressedSize};
  std::vector<int64_t> indexShape = {indexSize};
  std::vector<int64_t> biasShape = {n};
  std::vector<int64_t> deqScaleShape = {n};
  std::vector<int64_t> outputShape = {m, n};

  std::vector<int64_t> compressInfoHostData = {8, 8, k, n, 1};

  void* mat1DeviceAddr = nullptr;
  void* mat2CompressedDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* deqScaleDeviceAddr = nullptr;
  void* outputDeviceAddr = nullptr;

  aclTensor* mat1 = nullptr;
  aclTensor* mat2Compressed = nullptr;
  aclTensor* index = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* deqScale = nullptr;
  aclTensor* output = nullptr;
  aclIntArray* compressInfo = nullptr;

  std::string rootPath = "./data/";

  // Create a mat1 aclTensor.
  std::string mat1FilePath = rootPath + "mat1.bin";
  ret = CreateAclTensor(mat1FilePath, mat1Shape, sizeof(int8_t), &mat1DeviceAddr, aclDataType::ACL_INT8, &mat1);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create mat1 tensor failed. ERROR: %d\n", ret); return ret);
  // Create a mat2Compressed aclTensor.
  std::string mat2FilePath = rootPath + "weight/weight.dat";
  ret = CreateAclTensor(mat2FilePath, mat2CompressedShape, sizeof(int8_t), &mat2CompressedDeviceAddr,
                        aclDataType::ACL_INT8, &mat2Compressed);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create mat2 tensor failed. ERROR: %d\n", ret); return ret);
  // Create an index aclTensor.
  std::string indexFilePath = rootPath + "index/weight.dat";
  ret = CreateAclTensor(indexFilePath, indexShape, sizeof(int8_t), &indexDeviceAddr, aclDataType::ACL_INT8, &index);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create index tensor failed. ERROR: %d\n", ret); return ret);
  // Create a bias aclTensor.
  std::string biasFilePath = rootPath + "bias.bin";
  ret = CreateAclTensor(biasFilePath, biasShape, sizeof(int32_t), &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create bias tensor failed. ERROR: %d\n", ret); return ret);
  // Create a deqScale aclTensor.
  std::string deqScaleFilePath = rootPath + "deqScale.bin";
  ret = CreateAclTensor(deqScaleFilePath, deqScaleShape, sizeof(int32_t), &deqScaleDeviceAddr, aclDataType::ACL_UINT64,
                        &deqScale);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create deqScale tensor failed. ERROR: %d\n", ret); return ret);
  // Create compressInfo.
  compressInfo = aclCreateIntArray(compressInfoHostData.data(), aclDataType::ACL_INT64);
  // Create an out aclTensor.
  std::string outputFilePath = rootPath + "output.bin";
  ret = CreateAclTensor(outputFilePath, outputShape, 2, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create output tensor failed. ERROR: %d\n", ret); return ret);

  int32_t offsetX = 0;

  // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMm.
  ret = aclnnMatmulCompressDequantGetWorkspaceSize(mat1, mat2Compressed, index, bias, deqScale, nullptr, offsetX, compressInfo, output, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompressDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMm.
  ret = aclnnMatmulCompressDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompressDequant failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modification is required based on the specific API definition.
  auto size = GetShapeSize(outputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modification is required based on the specific API definition.
  aclDestroyTensor(mat1);
  aclDestroyTensor(mat2Compressed);
  aclDestroyTensor(index);
  aclDestroyTensor(bias);
  aclDestroyTensor(deqScale);
  aclDestroyTensor(output);
  aclDestroyIntArray(compressInfo);

  // 7. Release hardware resources. Modification is required based on the specific API definition.
  aclrtFree(mat1DeviceAddr);
  aclrtFree(mat2CompressedDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(deqScaleDeviceAddr);
  aclrtFree(outputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
