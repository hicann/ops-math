# HansEncode

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|
|Atlas 200/300/500 推理产品|×|

## 功能说明

- 算子功能：对张量的指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩，压缩后的结果存储在device的内存上或offload到Host侧。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 310px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>input_tensor</td>
      <td>输入</td>
      <td>表示输入的待压缩张量，数据元素大小仅支持64的倍数且大于等于32768。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pdf</td>
      <td>输入</td>
      <td>表示inputTensor的指数位所在字节的概率密度分布，shape要求为(1, 256)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pdf</td>
      <td>输出</td>
      <td>表示inputTensor的指数位所在字节的概率密度分布，shape要求为(1, 256)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mantissa</td>
      <td>输出</td>
      <td>表示输出的尾数部分。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>fixed</td>
      <td>输出</td>
      <td>表示压缩的第一段输出。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>表示压缩超过fixedOut后的输出。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>statistic</td>
      <td>可选属性</td>
      <td><ul><li>表示是否进行pdf统计。</li><li>默认值为false。</li></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reshuff</td>
      <td>可选属性</td>
      <td><ul><li>表示是否对各核编码后的结果进行内存重整。</li><li>默认值为false。</li></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hans_encode](./examples/test_aclnn_hans_encode.cpp) | 通过[aclnnHansEncode](./docs/aclnnHansEncode.md)接口方式调用HansEncode算子。    |
