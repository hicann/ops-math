# aclnnCalculateMatmulWeightSizeV2
## 产品支持情况
[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/conversion/trans_data)


| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

- 算子功能：
  在Matmul算子ND格式输入下，计算如果要转换到NZ格式下需要占用的空间大小（单位为元素个数），该接口仅仅用于判断对weight Tensor预处理需要使用多少size才可使Matmul算子执行性能最优。
  例如：
  
  - 输入【510， 510】Float16/Bfloat16：该函数出于性能角度考虑，会将shape变化为【512，512】
因此函数会将引用输入修改为262144
  
  - 输入【510， 270】INT8：该函数出于性能角度考虑，会将shape变化为【512，288】
因此函数会将引用输入修改为147456
  
- 计算公式：

  $$
  Float16/Bfloat16:
  result = Align(Shapesize[0], 16) * Align(Shapesize[1], 16)
  $$

  $$
  INT8：
  result = Align(Shapesize[0], 16) * Align(Shapesize[1], 32)
  $$

## 函数原型

`aclnnStatus aclnnCalculateMatmulWeightSizeV2(const aclIntArray *tensorShape, aclDataType dataType, uint64_t *weightTensorSize)`

## aclnnCalculateMatmulWeightSizeV2

- **参数说明：**

  * tensorShape(aclIntArray *, 计算输入)：用于表达该次Matmul载入权重矩阵的Shape，公式中的Shapesize，Host侧的aclIntArray，输入shape支持2-6维，即（batch，n，k），其中batch表示权重矩阵的批次大小，支持0-4维，n表示单个batch权重矩阵第1维的大小，k表示单个batch权重矩阵第2维的大小，不支持空Array。
  * weightDtype(aclDataType, 计算输入)：weight的Dtype，支持FLOAT16、BFLOAT16、INT8。
  * weightTensorSize(uint64_t \*, 计算输出)：转换为NZ格式所占用的空间大小（单位为元素个数），公式中的result。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 输入是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1.不支持空Tensor的输入空Tensor
                                   2.输入shape的维度不满足要求
                                   3.输入的数据类型不满足要求
  361001(ACLNN_ERR_RUNTIME_ERROR): 产品型号不支持。
  ```

## 约束说明

- 确定性计算：
  - aclnnCalculateMatmulWeightSizeV2默认确定性实现。


## 调用示例

* <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品<term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、 <term>Atlas 推理系列产品 </term> ：
  
  tensorShape为2维的场景示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
  tensorShape为多维（3-6维）的场景示例代码请参考[aclnnQuantMatmulV3](./aclnnQuantMatmulV3.md)。
  伪量化有aclnnWeightQuantBatchMatmulV2和aclnnWeightQuantBatchMatmulV3接口， 这里以aclnnWeightQuantBatchMatmulV2为例

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

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
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));

    const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    size *= sizeof(T);

    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {16, 32};
    std::vector<int64_t> weightShape = {32, 16};
    std::vector<int64_t> yShape = {16, 16};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(512, 1);
    std::vector<int8_t> weightHostData(512, 1);
    std::vector<float> yHostData(256, 0);

    std::vector<int64_t> antiquantScaleShape = {16};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(16, 1);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensorWeight(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xFp16 aclTensor
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 调用TransWeight
    ret = aclnnTransMatmulWeightGetWorkspaceSize(weight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnTransMatmulWeight第二段接口
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    workspaceSize = 0;
    // 调用cast生成FP16的输入
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weight, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 将输出转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

* <term>昇腾910_95 AI处理器</term>: 

  这里以aclnnQuantMatmulWeightNz为例，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> biasShape = {32};
      std::vector<int64_t> offsetShape = {32};
      std::vector<int64_t> scaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<int32_t> biasHostData(32, 1);
      std::vector<float> scaleHostData(32, 1);
      std::vector<float> offsetHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1);  // 实际上是float16半精度方式
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建昇腾亲和数据排布格式的x2 aclTensor
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建quantParam aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建offset aclTensor
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // 调用aclnnTransMatmulWeight第一段接口
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnTransMatmulWeight第二段接口
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // FLOAT数据类型的scale需要提前调用TransQuantParamV2算子的aclnn接口
      // 调用aclnnTransQuantParamV2第一段接口
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // 调用aclnnTransQuantParamV2第二段接口
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnQuantMatmulWeightNz第一段接口
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2, nullptr, quantParam, nullptr, nullptr, nullptr, nullptr,
                                                    bias, transposeX1, transposeX2, 0, out, &workspaceSize,
                                                    &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulWeightNz第二段接口
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```