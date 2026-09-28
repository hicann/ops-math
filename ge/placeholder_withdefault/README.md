# PlaceholderWithDefault

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

为一个张量插入带默认值的占位符。PlaceholderWithDefault算子接收输入张量x，输出y的形状和数据类型与输入x一致，由InferShape将输入shape复制到输出，InferDataType将输入dtype复制到输出。shape属性用于指定张量的形状信息。

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
      <td>shape</td>
      <td>属性</td>
      <td>必选。张量的形状。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>创建的占位张量，形状和数据类型与输入x一致。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

PlaceholderWithDefault算子为图引擎基础算子，通常作为图中占位节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_placeholder_withdefault](./examples/test_geir_placeholder_withdefault.cpp) | 通过[算子IR](./op_graph/placeholder_withdefault_proto.h)构图方式调用PlaceholderWithDefault算子 |
