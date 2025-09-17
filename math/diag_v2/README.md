# DiagV2

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|昇腾910_95 AI处理器|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|√|
|Atlas 200/300/500 推理产品|×|

## 功能说明

- 算子功能：根据输入的二维张量，提取由diagonal指定的对角线元素。
  
  如果diagonal = 0，则选择主对角线；
  
  如果diagonal > 0，则在主对角线的上方；
  
  如果diagonal < 0，则在主对角线的下方。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>UINT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL、COMPLEX32、COMPLEX128、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>可选属性</td>
      <td><ul><li>表示选择对角线的位置。</li><li>默认值为0。</li></td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>UINT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL、COMPLEX32、COMPLEX128、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_diag_v2](./examples/test_geir_diag_v2.cpp)   | 通过[算子IR](./op_graph/diag_v2_proto.h)构图方式调用DiagV2算子。 |
