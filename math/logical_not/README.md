# LogicalNot

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

算子功能：计算给定输入Tensor的逐元素逻辑非。如果未指定输出类型，输出Tensor是bool类型。如果输入Tensor不是bool类型，则将零视为False，非零视为True。

## 参数说明：**
  
  * self(aclTensor\*, 计算输入)：输入Tensor，Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、UINT8、INT8、INT16、INT32、INT64、FLOAT、FLOAT16、DOUBLE、BFLOAT16。
  * out(aclTensor\*, 计算输出)：输出Tensor，Device侧的aclTensor，shape与`self`一致，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BOOL、UINT8、INT8、INT16、INT32、INT64、FLOAT、FLOAT16、DOUBLE、BFLOAT16。

## 约束说明
无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_not.cpp](./examples/test_aclnn_logical_not.cpp) | 通过[aclnnLogicalNot](./docs/aclnnLogicalNot.md)接口方式调用LogicalNot算子。 |
