# AddV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量`x1`和`x2`执行逐元素加法，支持广播，兼容TensorFlow AddV2语义。

- 版本说明：名称中的V2用于对应TensorFlow AddV2图节点，并不表示加法或广播规则发生变化。在AddV2与Add共同支持的数据类型范围内，两者的数学语义一致；AddV2的数据类型和调用通路以本文为准。

- 计算公式：

$$
y_i=x_{1,i}+x_{2,i}
$$

其中 $i$ 遍历广播结果的全部元素，$x_{1,i}$ 和 $x_{2,i}$ 表示广播后对应位置的元素。

- 广播示例：

```text
x1 shape (3, 4), x2 shape (1, 4) -> y shape (3, 4)
x1 shape (3, 1), x2 shape (1, 4) -> y shape (3, 4)
```

广播时从末尾维度向前对齐，每组对应维度的长度必须相等，或至少有一路的维度长度为1。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>x1</td>
      <td>输入</td>
      <td>加法运算的第一个输入张量，对应公式中的x1。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>加法运算的第二个输入张量，对应公式中的x2。数据类型与x1一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>加法运算的输出张量，对应公式中的y。数据类型与x1一致，shape为x1与x2的广播结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

### 产品差异说明

<table><thead>
  <tr>
    <th>产品</th>
    <th>数据类型</th>
    <th>静态shape能力</th>
    <th>动态shape能力</th>
    <th>shape/rank及空Tensor限制</th>
  </tr></thead>
<tbody>
  <tr>
    <td><term>Ascend 950PR/Ascend 950DT</term></td>
    <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT16、UINT8、INT8、INT64、COMPLEX64</td>
    <td>输入ND→输出ND</td>
    <td>输入ND→输出ND；支持固定rank动态shape和dynamic rank</td>
    <td>rank取值范围为[1, 8]；支持空Tensor，输出y的shape仍必须是两路输入的广播结果</td>
  </tr>
  <tr>
    <td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br>
      <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></td>
    <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT64</td>
    <td>输入ND→输出ND</td>
    <td>输入ND→输出ND；支持固定rank动态shape，不支持dynamic rank</td>
    <td>rank取值范围为[0, 8]，rank为0时表示标量；空Tensor仅支持两路输入shape相同，或其中一路输入为单元素张量的广播场景</td>
  </tr>
  <tr>
    <td><term>Atlas 200I/500 A2 推理产品</term><br>
      <term>Atlas 推理系列产品</term><br>
      <term>Atlas 训练系列产品</term></td>
    <td>FLOAT16、FLOAT、INT32、INT64</td>
    <td>输入ND→输出ND</td>
    <td>输入ND→输出ND；支持固定rank动态shape，不支持dynamic rank</td>
    <td>rank取值范围为[0, 8]，rank为0时表示标量；空Tensor仅支持两路输入shape相同，或其中一路输入为单元素张量的广播场景</td>
  </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_add_v2](./examples/test_geir_add_v2.cpp) | 通过[算子IR](./op_graph/add_v2_proto.h)构图方式调用AddV2算子。 |

本算子不提供同名aclnn接口，仅支持上表的GE图模式调用。
