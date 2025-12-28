# Dot

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：计算两个一维张量的点积结果。

- 计算公式：
  $$
    out = x_{1}*y_{1} + x_{2}*y_{2} + ... + x_{n}*y_{n}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>input_x</td>
      <td>输入</td>
      <td>点积左向量，公式中的输出张量x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_y</td>
      <td>输入</td>
      <td>点积右向量，公式中的输出张量y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>点积输出，公式中的输出张量out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dot](./examples/test_aclnn_dot.cpp) | 通过[aclnnDot](./docs/aclnnDot.md)接口方式调用Dot算子。    |