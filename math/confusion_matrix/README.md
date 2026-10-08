# ConfusionMatrix

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算分类任务的混淆矩阵。给定真实标签labels和预测标签predictions，统计每个(label, prediction)对的出现次数，生成num_classes×num_classes的矩阵。当weights不为空时，每次匹配的贡献为对应的权重值而非1。

- 计算公式：

  如果指定了weights，则

  $$
  y[labels_i, predictions_i] = y[labels_i, predictions_i] + weights_i
  $$

  否则：

  $$
  y[labels_i, predictions_i] = y[labels_i, predictions_i] + 1
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>labels</td>
      <td>输入</td>
      <td>真实标签，1维tensor，取值范围[0, num_classes)。</td>
      <td>INT8、INT32、UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predictions</td>
      <td>输入</td>
      <td>预测标签，1维tensor，shape和dtype与labels一致。</td>
      <td>INT8、INT32、UINT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>每个样本的权重，可为空。shape与labels一致。</td>
      <td>FLOAT、FLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>num_classes</td>
      <td>属性</td>
      <td>类别数量，取值范围[1, 4096]。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>输出tensor的数据类型。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>混淆矩阵，shape为[num_classes, num_classes]。</td>
      <td>FLOAT、FLOAT16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- labels和predictions的shape必须一致。
- labels和predictions的dtype必须一致。
- weights不为空时，其shape必须与labels一致。
- labels和predictions中的值必须在[0, num_classes)范围内。
- num_classes取值范围为[1, 4096]。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| 图模式调用 | [test_geir_confusion_matrix](./examples/test_geir_confusion_matrix.cpp) | 通过[算子IR](./op_graph/confusion_matrix_proto.h)构图方式调用ConfusionMatrix算子。 |
