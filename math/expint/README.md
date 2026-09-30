# Expint

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

- 算子功能：计算指数积分Ei(x) = PV ∫_{-∞}^{x} (eᵗ/t) dt（柯西主值积分），计算公式为：

$$Ei(x) = PV \int_{-\infty}^{x} \frac{e^{t}}{t} \mathrm{d}t$$

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
      <td>待进行指数积分计算的入参，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>指数积分计算的出参，公式中的Ei(x)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持DOUBLE（FP64）数据类型。
- 输入x < 0时输出为NaN，x = 0时输出为-inf，x = +inf时输出为+inf。
- 输出的Shape与数据类型由框架根据输入推导（逐元素计算，输出与输入x一致），数据格式为ND。
- 支持0～8维ND Tensor，包括标量、空Tensor、动态Shape和动态Rank；不支持9维及以上Tensor。
- 非连续Tensor输入由框架自动转换为连续Tensor后执行。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用（静态Shape） | [test_geir_expint](./examples/arch35/test_geir_expint.cpp) | 通过图模式调用Expint算子。 |
| 图模式调用（动态Shape） | [test_geir_expint_dynamic](./examples/arch35/test_geir_expint_dynamic.cpp) | 通过图模式调用Expint算子，支持未知维度(-1)与未知Rank(-2)场景。 |
