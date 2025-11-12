# MaskedSelectV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出，形成一个新的一维张量。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1043px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 320px">
<col style="width: 96px">
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
    <td>输入张量</td>
    <td>输入张量，shape需要与mask满足broadcast关系，支持非连续的Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mask</td>
    <td>输入张量</td>
    <td>布尔掩码张量，shape要和self满足broadcast关系，支持非连续的Tensor。</td>
    <td>UINT8、BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>输出一维张量，元素个数为mask和self广播后的维度大小，不支持非连续的Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- self和mask的shape必须能够进行broadcast操作。
- out的shape必须是一维，且元素个数等于self和mask广播后的维度大小。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_masked_select](./examples/test_aclnn_masked_select_v3.cpp) | 通过[aclnnMaskedSelect](docs/aclnnMaskedSelect.md)接口方式调用MaskedSelect算子。 |