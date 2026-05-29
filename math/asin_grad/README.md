# AsinGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：AsinGrad 是 Asin（反正弦）算子的梯度算子，用于深度学习框架自动微分的反向传播阶段。给定正向输入 y 和上游梯度 dy，计算输出梯度 z。
- 计算公式：

  $$
  z_i = \frac{dy_i}{\sqrt{1 - y_i^2}}
  $$

  其中：
  - y：正向 Asin 算子的输入 tensor，值域 [-1, 1]
  - dy：上游传播的梯度（grad_output）
  - z：计算得到的输入梯度（grad_input）

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y</td>
    <td>输入</td>
    <td>正向 Asin 算子的输入 tensor，对应公式中的 y。支持 0-8 维 tensor，支持空 tensor。shape 需要与 dy 的 shape 完全相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>输入</td>
    <td>上游传播的梯度 tensor，对应公式中的 dy。支持 0-8 维 tensor，支持空 tensor。shape 需要与 y 的 shape 完全相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>z</td>
    <td>输出</td>
    <td>计算得到的输入梯度 tensor，对应公式中的 z。shape 与输入相同，数据类型与输入一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- y 和 dy 的数据类型必须一致，z 的数据类型与 y/dy 一致，仅支持 FLOAT、FLOAT16、BFLOAT16。
- y 和 dy 的 shape 必须完全相同，不支持广播。
- 支持 0-8 维 tensor，支持空 tensor（0 元素）。
- y 的值域应在 [-1, 1] 范围内。当 |y| = 1 时，计算结果为 inf；当 |y| > 1 时，计算结果为 NaN。
- FLOAT16 走 native FP16 计算；BFLOAT16 走 BF16 → FP32 中间计算；FLOAT32 直接 FP32 计算。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_asin_grad.cpp](examples/test_geir_asin_grad.cpp) | 通过图模式方式调用AsinGrad算子。 |
