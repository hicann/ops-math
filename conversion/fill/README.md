# Fill

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- **算子功能**：对张量进行填充操作，支持非连续的Tensor操作。
- **InplaceFillScalar**：使用标量值填充整个张量。
- **InplaceFillTensor**：使用张量值填充整个张量（value张量必须是0维或size=1的1维张量）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 922px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 308px">
<col style="width: 102px">
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
    <td>dims</td>
    <td>输入</td>
    <td>一维张量。表示输出张量的形状。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>零维标量。指定用于填充返回张量的值。数据类型需要与dims的数据类型满足数据类型推导规则</td>
    <td>INT8、INT32、INT64、FLOAT、FLOAT16、BOOL、BFLOAT16</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出张量。</td>
    <td>INT8、INT32、INT64、FLOAT、FLOAT16、BOOL、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 调用说明

| 调用方式  | 样例代码                                              | 说明                                                         |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_fill.cpp](./examples/test_aclnn_fill.cpp) | 通过[aclnnInplaceFillScalar](docs/aclnnInplaceFillScalar.md)和[aclnnInplaceFillTensor](docs/aclnnInplaceFillTensor.md)接口方式调用InplaceFillScalar算子。 |

