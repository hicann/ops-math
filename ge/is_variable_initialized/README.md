# IsVariableInitialized

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

检查张量是否已初始化。IsVariableInitialized算子接收输入张量x，输出标量布尔值y，表示输入张量是否已初始化。InferShape将输出shape设置为标量（0维），InferDataType将输出数据类型设置为DT_BOOL。

## 参数说明

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>输入张量。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>标量布尔张量，表示输入张量是否已初始化。形状为标量（0维），数据类型为DT_BOOL。</td>
      <td>TensorType({DT_BOOL})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 输出y为标量（0维张量），数据类型固定为DT_BOOL。

## 调用说明

IsVariableInitialized算子为图引擎基础算子，通常用于检查变量是否已初始化，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_is_variable_initialized](./examples/test_geir_is_variable_initialized.cpp) | 通过[算子IR](./op_graph/is_variable_initialized_proto.h)构图方式调用IsVariableInitialized算子 |
