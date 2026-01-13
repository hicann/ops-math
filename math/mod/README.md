# Mod

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |

## 功能说明

- 算子功能：返回 self除以other的余数。

- 计算公式：

  对于入参self，和比较标量other，Fmod可以用如下数学公式表示：

  $$
  out_{i} = self_{i} - (other \times\left \lfloor (self_{i}/other) \right \rfloor)
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
      <td>self</td>
      <td>输入</td>
      <td>待进行mod计算的入参，公式中的self_i。</td>
      <td>DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行mod计算的入参，公式中的other。</td>
      <td>DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行mod计算的出参，公式中的out_i。</td>
      <td>DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. 数据类型需满足数据类型推导规则，推导后的数据类型需在支持的数据类型范围内。
2. self和out的shape必须一致。
3. 数据维度不支持8维以上。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fmod_scalar](./examples/test_aclnn_fmod_scalar.cpp) | 通过aclnnFmodScalar接口方式调用Mod算子（标量版本）。 |
| aclnn调用 | [test_aclnn_fmod_tensor](./examples/test_aclnn_fmod_tensor.cpp) | 通过aclnnFmodTensor接口方式调用Mod算子（张量版本）。 |
| 图模式调用 | [算子IR](./op_graph/mod_proto.h)   | 通过算子IR构图方式调用Mod算子。 |
