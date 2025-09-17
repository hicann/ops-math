# Abs

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|昇腾910_95 AI处理器|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|√|
|Atlas 200/300/500 推理产品|×|

## 功能说明

- 算子功能：为输入张量的每一个元素取绝对值。

- 计算公式：

$$
out_i=|input_i|
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>待进行abs计算的入参，公式中的input_i。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行abs计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），场景为最后2轴合轴小于16，前面的轴合轴超大。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_abs](./examples/test_aclnn_abs.cpp) | 通过[aclnnAbs](./docs/aclnnAbs.md)接口方式调用Abs算子。 |
| 图模式调用 | [test_geir_abs](./examples/test_geir_abs.cpp)   | 通过[算子IR](./op_graph/abs_proto.h)构图方式调用Abs算子。 |
