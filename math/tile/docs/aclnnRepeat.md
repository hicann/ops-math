# aclnnRepeat

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/tile)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

接口功能：对输入tensor沿着repeats中对每个维度指定的复制次数进行复制。示例：
假设输入Tensor为[[a,b],[c,d],[e,f]]，即shape为[3,2]，repeats为(2,4)，则生成的Tensor的shape为[6,8]，值如下所示：

```
>>> x = torch.tensor([[a,b],[c,d],[e,f]])
>>> x.repeat(2,4)
tensor([[a,b,a,b,a,b,a,b],
        [c,d,c,d,c,d,c,d],
        [e,f,e,f,e,f,e,f],
        [a,b,a,b,a,b,a,b],
        [c,d,c,d,c,d,c,d],
        [e,f,e,f,e,f,e,f],
        ])
```
当repeats为(2,4,2)时，即repeats的元素个数大于Tensor中的维度，则输出Tensor等效为如下操作：先将输入Tensor的shape扩张到和repeats个数相同的维度：[1,3,2]，而后按照对应维度和repeats的值进行扩张，即输出Tensor的shape为[2,12,4]，结果如下：
```
>>> x.repeat(2,4,2)
tensor([[[a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f]],

        [[a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f],
         [a,b,a,b],
         [c,d,c,d],
         [e,f,e,f]]])
```
计算时需要满足以下条件：
repeats中参数个数不能少于输入Tensor的维度。
repeats中的值必须大于等于0。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context//两段式接口.md)，必须先调用“aclnnRepeatGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRepeat”接口执行计算。

```cpp
aclnnStatus aclnnRepeatGetWorkspaceSize(
  const aclTensor   *self, 
  const aclIntArray *repeats, 
  aclTensor         *out, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnRepeat(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnRepeatGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px"> 
  </colgroup> 
  <thead> 
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度（shape）</th>
      <th>非连续张量Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT、DOUBLE、FLOAT16、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT32、INT64、UINT16、UINT32、UINT64、BOOL、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>输入</td>
      <td>-</td>
      <td>表示沿每个维度重复输入tensor的次数，参数个数不大于8, 当前不支持对超过4个维度同时做repeat的场景，详细约束请见约束说明。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>-</td>
      <td>-</td>
      <td>与self一致</td>
      <td>ND</td>
      <td>≤8</td>
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
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型不支持支持HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型不支持BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
    
  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1207px"><colgroup>
  <col style="width: 268px">
  <col style="width: 138px">
  <col style="width: 801px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self和out的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self和out的type不匹配。</td>
    </tr>
    <tr>
      <td>参数repeats的参数个数小于输入tensor的维度。</td>
    </tr>
    <tr>
      <td>参数repeats中含有小于等于0的值。</td>
    </tr>
    <tr>
      <td>self的维度数超过8。</td>
    </tr>
    <tr>
      <td>repeats的参数个数超过8。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>kernel执行失败, 中间结果为null。</td>
    </tr>
    <tr>
      <td>同时对超过4个维度做repeat。</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeat

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1143px"><colgroup>
  <col style="width: 158px">
  <col style="width: 140px">
  <col style="width: 845px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRepeatGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRepeat默认确定性实现。

repeat功能内部broadcast的kernel有最大8维度的限制，暂不支持扩维度后超过8维的场景，详细如下：  
  限制1. 当需要对第一根轴进行repeat时，最大支持同时对4个维度进行repeat操作（即repeats的参数非1格式不超过4）。
  ```
   x.repeat(2, 3, 4, 5, 6)  # 不支持，校验报错，第一根轴为repeat为2，同时5个非1repeat参数
   x.repeat(2, 3, 1, 5, 6)  # 支持，第一根轴为repeat为2，同时4个非1repeat参数
  ```
  限制2. 当不需要对第一根轴进行repeat时，最大支持同时对3个维度进行repeat操作（即repeats的参数非1格式不超过3）。
  ```
   x.repeat(1, 3, 4, 5, 6)  # 不支持，校验报错，第一根轴为repeat为1，同时4个非1repeat参数
   x.repeat(1, 3, 1, 5, 6)  # 支持，第一根轴为repeat为1，同时3个非1repeat参数
  ```

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
 } while (0)

#define LOG_PRINT(message, ...)      \
 do {                                \
    printf(message, ##__VA_ARGS__);  \
 } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
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

template <typename T>
void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray) {
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {3, 2};
  std::vector<int64_t> outShape = {2, 12, 4};
  std::vector<int64_t> repeatsArray = {2, 4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclIntArray* repeat = nullptr;
  std::vector<float> selfHostData(GetShapeSize(selfShape) * 2, 1);
  std::vector<float> outHostData(GetShapeSize(outShape) * 2, 1);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建normalizedShape aclIntArray
  aclCreateIntArrayP(repeatsArray, &repeat);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRepeat第一段接口
  ret = aclnnRepeatGetWorkspaceSize(self, repeat, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRepeat第二段接口
  ret = aclnnRepeat(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeat failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyIntArray(repeat);

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
