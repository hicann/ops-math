# FileConstant

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

创建文件常量张量，用于处理存储在文件中的超大权重数据。根据shape和dtype属性推导输出张量y的形状和数据类型。输出形状与shape属性一致，输出数据类型与dtype属性一致。

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
      <td>file_path</td>
      <td>属性</td>
      <td>记录文件路径的字符串。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>file_id</td>
      <td>属性</td>
      <td>记录文件ID的字符串。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>属性</td>
      <td>必选。数据形状。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>必选。数据类型。</td>
      <td>Type</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>文件常量张量，形状与shape属性一致，数据类型与dtype属性一致。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

FileConstant算子为图引擎基础算子，通常作为图中文件常量节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_file_constant](./examples/test_geir_file_constant.cpp) | 通过[算子IR](./op_graph/file_constant_proto.h)构图方式调用FileConstant算子 |
