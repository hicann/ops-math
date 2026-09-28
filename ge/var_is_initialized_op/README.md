# VarIsInitializedOp

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

检查张量（变量）是否已被初始化。输出为bool标量，指示输入变量是否已初始化：输入需为Variable节点，图引擎在编译期通过VarIsInitializedOpPass查询Variable的初始化状态并将本算子节点改写为常量输出。

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
      <td>待检查的变量张量（需为Variable节点输出）。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>bool标量，指示变量是否已初始化（与输入shape/dtype无关）。</td>
      <td>TensorType({DT_BOOL})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 本算子为实验性功能（EXPERIMENTAL），请勿在生产环境使用。
- 输入需为Variable节点：图引擎VarIsInitializedOpPass在编译期查询Variable初始化状态并改写本算子节点。
- 输出恒为bool标量，与输入的shape和数据类型无关。

## 调用说明

VarIsInitializedOp算子为图引擎基础算子，通常与Variable配合使用：`Variable → VarIsInitializedOp`，查询变量的初始化状态。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_var_is_initialized_op](./examples/test_geir_var_is_initialized_op.cpp) | 通过[算子IR](./op_graph/var_is_initialized_op_proto.h)构图方式调用VarIsInitializedOp算子 |
