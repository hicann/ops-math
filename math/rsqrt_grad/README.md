# RsqrtGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：Rsqrt算子的反向梯度计算。

- 计算公式：

$$
z = -0.5 \times dy \times y^{3}
$$

其中y为Rsqrt算子的前向输出。

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
      <td>y</td>
      <td>输入</td>
      <td>Rsqrt算子的前向输出，公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>输出梯度，公式中的dy。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>输入梯度，公式中的z，shape与y一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                      | 说明                                                      |
|--------------|-----------------------------------------------------------|---------------------------------------------------------|
| 图模式调用 | [test_geir_rsqrt_grad](./examples/test_geir_rsqrt_grad.cpp) | 通过[算子IR](./op_graph/rsqrt_grad_proto.h)构图方式调用RsqrtGrad算子。 |
