# Eye

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：返回一个二维张量，该张量的对角线上元素值为1，其余元素值为0。

- 计算公式：
  对于入参self，和比较标量other，gt可以用如下数学公式表示：
  
  $$
  y_{i, j} = 
  \begin{cases}
    1, &if\ i==j \\
    0, &other
  \end{cases}
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
      <td>y</td>
      <td>输出</td>
      <td>对角线输出，公式中的输出张量y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32、INT64、UINT6、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_eye](./examples/test_aclnn_eye.cpp) | 通过[aclnnEye](./docs/aclnnEye.md)接口方式调用Eye算子。    |