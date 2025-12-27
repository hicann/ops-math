# OnesLike

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：返回形状和类型相同的张量，所有元素都设置为1。

- 示例：

  ```
  输入input：
  tensor([[1, 2],
          [3, 4]])
  输出output：
  tensor([[1, 1],
          [1, 1]])
  ```

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
      <td>input</td>
      <td>输入</td>
      <td>待进行onelike计算的入参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>进行onelike计算的出参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_inplace_one](./examples/test_aclnn_inplace_one.cpp) | 通过[aclnnInplaceOne](./docs/aclnnInplaceOne.md)接口方式调用OnesLike算子。 |
