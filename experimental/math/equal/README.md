# Equal

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |

## 功能说明

- 算子功能：对两个输入张量进行逐元素相等比较，并根据广播后的形状输出BOOL张量。
- 计算公式：

$$
out_i = (self_i == other_i)
$$

- 浮点比较采用数值语义：`+0.0`与`-0.0`相等，`NaN`与任意值（包括`NaN`）均不相等，同号无穷大相等。

## 参数说明

<table style="table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待比较的第一个输入张量。</td>
      <td>数据类型与other一致；shape与other满足broadcast关系。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、INT32、UINT32、INT64、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待比较的第二个输入张量。</td>
      <td>数据类型与self一致；shape与self满足broadcast关系。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、INT32、UINT32、INT64、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>逐元素比较结果。</td>
      <td>shape为self与other广播后的公共shape。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- self与other的数据类型必须一致。
- self与other的shape必须满足broadcast关系。
- out的数据类型必须为BOOL，shape必须与输入广播后的公共shape一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_eq_tensor.cpp](./examples/test_aclnn_eq_tensor.cpp) | 通过[aclnnEqTensor/aclnnInplaceEqTensor](./docs/aclnnEqTensor&aclnnInplaceEqTensor.md)接口调用Equal算子。 |
| aclnn调用 | [test_aclnn_eq_scalar.cpp](./examples/test_aclnn_eq_scalar.cpp) | 通过[aclnnEqScalar/aclnnInplaceEqScalar](./docs/aclnnEqScalar&aclnnInplaceEqScalar.md)接口调用Equal算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| gcw_rESKWmgp | 个人开发者 | Equal | 2026/08/27 | 完成A2/A3 Ascend C实现、数据类型适配、广播支持及性能优化。 |
