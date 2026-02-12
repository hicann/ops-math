# aclnnOneHot

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/one_hot)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：对长度为n的输入self， 经过one_hot的计算后得到一个元素数量为n*k的输出out，其中k的值为numClasses。
  输出的元素满足下列公式：
  
  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnOneHotGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnOneHot”接口执行计算。

```Cpp
aclnnStatus aclnnOneHotGetWorkspaceSize(
  const aclTensor* self, 
  int numClasses, 
  const aclTensor* onValue, 
  const aclTensor* offValue, 
  int64_t axis, 
  aclTensor* out, 
  uint64_t* workspaceSize, 
  aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnOneHot(
  void* workspace, 
  uint64_t workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream stream)
```

## aclnnOneHotGetWorkspaceSize

- **参数说明：**

<table style="undefined;table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 213px">
  <col style="width: 288px">
  <col style="width: 333px">
  <col style="width: 124px">
  <col style="width: 138px">
  <col style="width: 151px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>Device侧的aclTensor。</td>
      <td>shape支持1-8维度。</td>
      <td>UINT8、INT32、INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numClasses</td>
      <td>输入</td>
      <td>表示类别数。</td>
      <td>当self为空Tensor时，numClasses的值需大于0；当self不为空Tensor时，numClasses需大于等于0。若numClasses的值为0，则返回空Tensor。如果self存在元素大于numClasses，这些元素会被编码成全offValue。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onValue</td>
      <td>输入</td>
      <td>表示索引位置的填充值，公式中的onValue，Device侧的aclTensor。</td>
      <td>shape支持1-8维度，且计算时只使用其中第一个元素值进行计算。数据类型与out一致。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offValue</td>
      <td>输入</td>
      <td>表示非索引位置的填充值，公式中的offValue，Device侧的aclTensor。</td>
      <td>shape支持1-8维度，且计算时只使用其中第一个元素值进行计算。数据类型与out一致。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>表示编码向量的插入维度。</td>
      <td>最小值为-1，最大值为self的维度数。若值为-1，编码向量会往self的最后一维插入。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示one-hot张量，公式中的输出out，Device侧的aclTensor。</td>
      <td>最小值为-1，最大值为self的维度数。若值为-1，编码向量会往self的最后一维插入。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持UINT8数据类型。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、onValue、offValue或out为空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self、onValue、offValue或out不在支持的数据类型范围之内。</td>
    </tr>
    <tr>
      <td>onValue、offValue的数据类型与out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self为空Tensor，且numClasses小于等于0。</td>
    </tr>
    <tr>
      <td>self不为空Tensor，且numClasses小于0。</td>
    </tr>
    <tr>
      <td>axis的值小于-1。</td>
    </tr>
    <tr>
      <td>axis的值大于self的维度数量。</td>
    </tr>
    <tr>
      <td>out的维度不比self的维度多1维。</td>
    </tr>
    <tr>
      <td>out的shape与在self的shape在axis轴插入numClasses后的shape不一致。</td>
    </tr>
    <tr>
      <td>self、onValue、offValue或out的维度超过8维。</td>
    </tr>
  </tbody>
  </table>

## aclnnOneHot

- **参数说明：**

<table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
    <col style="width: 872px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnOneHotGetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
    </tbody>
    </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnOneHot默认确定性实现。
- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 当offValue的数据类型为INT64时，其首元素取值仅支持0或1。
  - 输入self大小为selfSize，输出out大小为outSize，ub大小为ubSize，当axis取值为0，满足下列条件的场景暂不支持：
    - selfSize * 3 < ubSize - 16K < outSize * 3 / 2


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_one_hot.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    int numClasses = 4;
    std::vector<int64_t> outShape = {4, 2, 4};
    std::vector<int64_t> onValueShape = {1};
    std::vector<int64_t> offValueShape = {1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* onValueDeviceAddr = nullptr;
    void* offValueDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclTensor* onValue = nullptr;
    aclTensor* offValue = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 3, 2, 1, 0};
    std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> onValueHostData = {1};
    std::vector<int32_t> offValueHostData = {0};
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建onValue aclTensor
    ret = CreateAclTensor(onValueHostData, onValueShape, &onValueDeviceAddr, aclDataType::ACL_INT32, &onValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offValue aclTensor
    ret = CreateAclTensor(offValueHostData, offValueShape, &offValueDeviceAddr, aclDataType::ACL_INT32, &offValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    int64_t axis = -1;
    aclOpExecutor* executor;
    // 调用aclnnoneHot第一段接口
    ret = aclnnOneHotGetWorkspaceSize(self, numClasses, onValue, offValue, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnOnehot第二段接口
    ret = aclnnOneHot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHot failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(onValue);
    aclDestroyTensor(offValue);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(onValueDeviceAddr);
    aclrtFree(offValueDeviceAddr);
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