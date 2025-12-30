# LogicalOr

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

算子功能：完成给定输入张量元素的逻辑或运算。当两个输入张量为非bool类型时，0被视为False，非0被视为True。


## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行logical_or计算的入参。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行logical_or计算的入参。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行logical_or计算的出参。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_or](./examples/test_aclnn_logical_or.cpp) | 通过[aclnnLogicalOr](./docs/aclnnLogicalOr&aclnnInplaceLogicalOr.md)接口方式调用LogicalOr算子。 |
| aclnn调用 | [test_aclnn_inplace_logical_or](./examples/test_aclnn_inplace_logical_or.cpp) | 通过[aclnnInplaceLogicalOr](./docs/aclnnLogicalOr&aclnnInplaceLogicalOr.md)接口方式调用LogicalOr算子。       |
