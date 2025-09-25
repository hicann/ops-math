# ConcatD
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    √     |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √     |

## 功能说明

- 算子功能：将tensors中所有tensor按照维度dim进行级联，除了dim对应的维度以外的维度必须一致。

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
      <td>x</td>
      <td>输入</td>
      <td>需要级联的tensor列表。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16、DOUBLE、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concat_dim</td>
      <td>属性</td>
      <td>需要级联的维度。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>N</td>
      <td>属性</td>
      <td>指定要级联的tensor个数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16、DOUBLE、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* x列表中元素的数据类型和数据格式不在支持的范围之内。
* x列表中无法做数据类型推导。
* 推导出的数据类型无法转换为指定输出out的类型。
* 非级联维度shape不一致。
* dim超过x维度范围。


## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_cat](examples/test_aclnn_cat.cpp) | 通过[aclnnCat](docs/aclnnCat.md)接口方式调用ConcatD算子。 |

