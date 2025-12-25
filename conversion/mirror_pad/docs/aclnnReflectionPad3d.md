# aclnnReflectionPad3d

[📄 查看源码](https://gitcode.com/cann/ops-math-dev/tree/master/conversion/mirror_pad)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |





## 功能说明

算子功能：3D反射填充。

```
例：
输入tensor([[[[[0,1],
               [2,3]],
              [[4,5],
               [6,7]]]]])
padding([1,1,1,1,1,1])
输出为
([[[[[7,6,7,6],
[5,4,5,4],
[7,6,7,6],
[5,4,5,4]],
[[3,2,3,2],
[1,0,1,0],
[3,2,3,2],
[1,0,1,0]],
[[7,6,7,6],
[5,4,5,4],
[7,6,7,6],
[5,4,5,4]],
[[3,2,3,2],
[1,0,1,0],
[3,2,3,2],
[1,0,1,0]]]]])
```

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnReflectionPad3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReflectionPad3d”接口执行计算。

- `aclnnStatus aclnnReflectionPad3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnReflectionPad3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnReflectionPad3dGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*,计算输入)。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，维度支持四维或五维，在最后三维做pad。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、COMPLEX64、COMPLEX128
     * <term>Atlas 训练系列产品</term>：FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL

  - padding(aclIntArray*,计算输入)：数据类型为INT64，长度为6，数值依次代表左右上下前后需要填充的值。padding前两个数值需小于self最后一维度的数值，中间两个数值需小于self倒数第二维度的数值，后两个数值需小于self倒数第三维度的数值。

  - out(aclTensor*,计算输出)：数据类型、[数据格式](../../../docs/context/数据格式.md)、维度与self一致，out倒数第三维度的数值等于self倒数第三维度的数值加padding后两个值，out倒数第二维度的数值等于self倒数第二维度的数值加padding中间两个值，out最后一维度的数值等于self最后一维度的数值加padding前两个值。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、COMPLEX64、COMPLEX128
     * <term>Atlas 训练系列产品</term>：FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL

  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. Tensor为空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. self、padding和out的数据类型或数据格式不在支持的范围之内。
                                   2. self、padding和out的输入shape在支持范围之外。
                                   3. 五维self为空tensor且存在非batch size维度的大小为0。
                                   4. 四维self不支持为空tensor。
                                   5. padding的数值大于等于self对应维度的值。
                                   6. out后三维度的值不等于self后三维度的值加对应padding。
                                   7. out的shape与实际输出shape不匹配。
  ```

## aclnnReflectionPad3d

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnReflectionPad3dGetWorkspaceSize获取。

  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnReflectionPad3d默认确定性实现。

- 如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），场景为最后2轴合轴小于16，前面的轴合轴超大。
- 五维self为空tensor且存在非batch size维度的大小为0。四维self不支持为空tensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_reflection_pad3d.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // 1. 固定写法，device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 4, 4, 4};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> paddingData = {1, 1, 1, 1, 1, 1};
    std::vector<float> outHostData(GetShapeSize(outShape) * 2, 0);

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建padding aclIntArray
    padding = aclCreateIntArray(paddingData.data(), 6);
    CHECK_RET(padding != nullptr, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnReflectionPad3d第一段接口
    ret = aclnnReflectionPad3dGetWorkspaceSize(self, padding, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReflectionPad3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnReflectionPad3d第二段接口
    ret = aclnnReflectionPad3d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReflectionPad3d failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyIntArray(padding);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
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
