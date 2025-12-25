# CoalesceSparse
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    √     |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √     |

## 功能说明

- 算子功能：将相同坐标点的value进行累加求和，进而减少Coo_Tensor的内存大小。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
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
      <td>unique_len</td>
      <td>输入</td>
      <td>去重后的索引数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>unique_indices</td>
      <td>输入</td>
      <td>去重后的索引数组。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引数组。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>每个坐标对应的元素值。</td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_indices</td>
      <td>输出</td>
      <td>合并后的索引数组。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_values</td>
      <td>输出</td>
      <td>合并后的元素值。</td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

重索引后的indices值不能超过int32上限。


## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式调用 | [test_geir_coalesce_sparse](./examples/test_geir_coalesce_sparse.cpp)   | 通过[算子IR](./op_graph/coalesce_sparse_proto.h)构图方式调用CoalesceSparse算子。 |
