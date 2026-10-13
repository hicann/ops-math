# FresnelSin

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

- 算子功能：计算Fresnel正弦积分 S(x) = integral_0^x sin(pi*t^2/2) dt。

- 计算公式：

$$S(x) = \int_{0}^{x} \sin\left(\frac{\pi}{2} t^{2}\right) \mathrm{d}t$$

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
      <td>待进行Fresnel正弦积分计算的入参，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Fresnel正弦积分计算的出参，公式中的S(x)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持DOUBLE（FP64）数据类型。
- x 的维度数（rank）范围为 0~8，支持 0 维标量输入。
- 支持空 Tensor（0 元素）输入：Host 侧 Tiling 短路直接返回，输出为同 shape 空 Tensor（自动处理）。
- 支持非连续 Tensor 输入（Host 侧自动处理）。
- 动态 shape 支持情况：支持动态 shape 与动态 rank（-1 未知维度 / -2 未知 rank）。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_fresnel_sin](./examples/test_geir_fresnel_sin.cpp) | 通过图模式调用FresnelSin算子。 |
