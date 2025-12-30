# ReduceVar

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：计算指定维度(dim)的标准差，这个dim可以是单个维度，维度列表或者None。

- 计算公式：

  $$
  out = \frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2
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
      <td>待进行计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>表示参与计算的维度</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>属性</td>
      <td>修正值</td>
      <td>INT64_ARRAY</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>属性</td>
      <td>是否在输出张量中保留输入张量的维度</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_mean_out</td>
      <td>属性</td>
      <td>判断是否输出均值结果</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>指定维度的方差。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>指定维度的均值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_reduce_var_mean](./examples/test_aclnn_reduce_var_mean.cpp) | 通过[aclnnVarMean](./docs/aclnnVarMean.md)接口方式调用ReduceVar算子。
| aclnn调用 | [test_aclnn_reduce_var](./examples/test_aclnn_reduce_var.cpp) | 通过[aclnnVar](./docs/aclnnVar.md)接口方式调用ReduceVar算子。