# Tanh

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明
- 算子功能：激活函数。返回与输入tensor shape相同的tensor，对输入tensor进行elementwise的计算。
- 计算公式：
$$
tanh(self)=\frac{e^{self}-e^{-self}}{e^{self} + e^{-self}}
$$

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
      <td>x</td>
      <td>输入</td>
      <td>待进行tanh计算的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行tanh计算的出参，公式中的tanh(self)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                            | 说明                                                           |
|--------------|-------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_tanh](./examples/test_aclnn_tanh.cpp) | 通过[aclnnTanh](./docs/aclnnTanh&aclnnInplaceTanh.md)接口方式调用Tanh算子。 |