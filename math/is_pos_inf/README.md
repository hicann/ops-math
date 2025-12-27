# IsPosInf

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能：判断输入张量的元素是否为正无穷。

- **参数说明：**

  - self(aclTensor*，计算输入)：输入tensor。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>昇腾910_95 AI处理器</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL。
    - <term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL。
  - out(aclTensor*，计算输出)：输出tensor，数据类型支持BOOL，shape需要与self一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_isposinf](./examples/test_aclnn_isposinf.cpp) | 通过[aclnnIsPosInf](./docs/aclnnIsPosInf.md)接口方式调用IsPosInf算子  |