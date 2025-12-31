# FloorDiv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：完成除法计算，对余数向下取整。
- 计算公式：

  $$
  out_i = floor(\frac{self_i}{other_i})
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入张量self_i</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量other_i</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量out_i</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_floor_divide](./examples/test_aclnn_floor_divide.cpp) | 通过[aclnnFloorDivide和aclnnFloorDivide](./docs/aclnnFloorDivide&aclnnInplaceFloorDivide.md)接口方式调用FloorDiv算子  |