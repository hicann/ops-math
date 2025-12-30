# Floor

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能: 返回输入Tensor中每个元素向下取整，并将结果回填到输入Tensor中。
- 计算公式：

  $$
  out_{i} =floor(self_{i})
  $$

- **参数说明：**

  * self(aclTensor*, 计算输入) ：待进行floor计算的入参。Device侧的aclTensor，数据类型必须和out一样，[数据格式](common/数据格式.md)支持ND，shape必须和out一样，支持[非连续的Tensor](common/非连续的Tensor.md)，其中UINT64、UINT32、UINT16不支持[非连续的Tensor](common/非连续的Tensor.md)，数据维度不支持8维以上。
    - <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT64、UINT32、UINT16、UINT8。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT64、UINT32、UINT16、UINT8、BFLOAT16。
  * out(aclTensor*, 计算输出)：floor计算的出参。Device侧的aclTensor，数据类型必须和self一样，[数据格式](common/数据格式.md)支持ND，shape必须和self一样， 支持[非连续的Tensor](common/非连续的Tensor.md)，数据维度不支持8维以上。
    - <term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT64、UINT32、UINT16、UINT8。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持DOUBLE、FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT64、UINT32、UINT16、UINT8、BFLOAT16。
  * workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_floor](./examples/test_aclnn_floor.cpp) | 通过[aclnnFloor和aclnnInplaceFloor](./docs/aclnnFloor&aclnnInplaceFloor.md)接口方式调用floor算子  |