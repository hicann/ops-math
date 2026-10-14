# FresnelCos

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    ×     |
| <term>Atlas推理系列产品</term>                       |    √     |
| <term>Atlas训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：计算Fresnel余弦积分C(x) = integral_0^x cos(pi*t^2/2) dt。

- 计算公式：

$$C(x) = \int_{0}^{x} \cos\left(\frac{\pi}{2} t^{2}\right) \mathrm{d}t$$

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>待进行Fresnel余弦积分计算的入参，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Fresnel余弦积分计算的出参，公式中的C(x)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持 BFLOAT16、FLOAT16、FLOAT 三种数据类型。
- x的维度数（rank）范围为0~8，支持0维标量输入。
- 支持空Tensor（0元素）输入：Host侧Tiling短路直接返回，输出为同shape空Tensor（自动处理）。
- 支持非连续Tensor输入（Host侧自动处理）。
- 动态shape支持情况：支持动态shape与动态rank（-1未知维度 / -2未知rank）。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_fresnel_cos](./examples/test_geir_fresnel_cos.cpp) | 通过图模式调用FresnelCos算子。 |
